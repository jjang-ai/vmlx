# SPDX-License-Identifier: Apache-2.0
"""MCP policy contracts for server-side tool visibility and execution."""

import asyncio
import json
import pytest

from vmlx_engine.mcp.manager import MCPClientManager
from vmlx_engine.mcp.types import (
    MCPConfig,
    MCPPolicy,
    MCPServerConfig,
    MCPServerState,
    MCPTool,
)


class _FakeClient:
    def __init__(self, config, tools):
        self.config = config
        self.name = config.name
        self.tools = tools
        self.is_connected = True
        self.calls = []

    def get_status(self):
        from vmlx_engine.mcp.types import MCPServerStatus

        return MCPServerStatus(
            name=self.name,
            state=MCPServerState.CONNECTED,
            transport=self.config.transport,
            tools_count=len(self.tools),
        )

    async def call_tool(self, tool_name, arguments, timeout=None):
        from vmlx_engine.mcp.types import MCPToolResult

        self.calls.append((tool_name, arguments, timeout))
        return MCPToolResult(tool_name=tool_name, content={"ok": True})


def _manager_with_fake_tools() -> MCPClientManager:
    config = MCPConfig(
        servers={
            "fs": MCPServerConfig(
                name="fs",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            ),
            "web": MCPServerConfig(
                name="web",
                command="uvx",
                args=["mcp-server-fetch"],
            ),
        }
    )
    manager = MCPClientManager(config)
    manager._clients = {
        "fs": _FakeClient(
            config.servers["fs"],
            [
                MCPTool("fs", "read_file", "read", {"type": "object"}),
                MCPTool("fs", "write_file", "write", {"type": "object"}),
            ],
        ),
        "web": _FakeClient(
            config.servers["web"],
            [MCPTool("web", "fetch", "fetch", {"type": "object"})],
        ),
    }
    return manager


def test_mcp_policy_filters_servers_and_tools_before_openai_schema_merge():
    manager = _manager_with_fake_tools()
    policy = MCPPolicy(
        enabled_servers={"fs"},
        enabled_tools={"fs__read_file"},
        disabled_tools={"fs__write_file"},
    )

    tools = manager.get_all_tools(policy=policy)
    openai_tools = manager.get_all_tools_openai(policy=policy)

    assert [tool.full_name for tool in tools] == ["fs__read_file"]
    assert [tool["function"]["name"] for tool in openai_tools] == ["fs__read_file"]


@pytest.mark.asyncio
async def test_mcp_policy_rejects_disabled_tool_execution_server_side():
    manager = _manager_with_fake_tools()
    policy = MCPPolicy(enabled_servers={"fs"}, disabled_tools={"fs__write_file"})

    result = await manager.execute_tool(
        "fs__write_file",
        {"path": "x", "content": "blocked"},
        policy=policy,
    )

    assert result.is_error is True
    assert "disabled by MCP policy" in (result.error_message or "")
    assert manager._clients["fs"].calls == []


@pytest.mark.asyncio
async def test_mcp_manager_start_and_stop_use_same_task_for_client_contexts():
    manager = MCPClientManager(MCPConfig())

    class TaskBoundClient:
        def __init__(self, name):
            self.name = name
            self.is_connected = False
            self.tools = []
            self.config = type("Config", (), {"enabled": True})()
            self.connect_task = None
            self.mismatched_disconnect_task = False

        async def connect(self):
            self.connect_task = asyncio.current_task()
            self.is_connected = True
            return True

        async def disconnect(self):
            self.mismatched_disconnect_task = asyncio.current_task() is not self.connect_task
            self.is_connected = False

    manager._clients = {
        "a": TaskBoundClient("a"),
        "b": TaskBoundClient("b"),
    }

    await manager.start()
    await manager.stop()

    assert not any(c.mismatched_disconnect_task for c in manager._clients.values())


def test_mcp_policy_status_marks_effective_tools_and_redacts_server_config():
    manager = _manager_with_fake_tools()
    policy = MCPPolicy(enabled_servers={"fs"}, enabled_tools={"fs__read_file"})

    status = manager.get_policy_status(policy=policy)

    assert status["servers"][0]["name"] == "fs"
    assert status["servers"][0]["enabled"] is True
    assert status["servers"][0]["command_redacted"] == "npx"
    assert status["tools"][0]["name"] == "fs__read_file"
    assert status["tools"][0]["effective"] is True
    assert status["tools"][1]["name"] == "fs__write_file"
    assert status["tools"][1]["effective"] is False


def test_mcp_policy_status_redacts_remote_url_query_secrets():
    manager = MCPClientManager(
        MCPConfig(
            servers={
                "remote": MCPServerConfig(
                    name="remote",
                    transport="http",
                    url="https://example.test/mcp?token=real-token&safe=ok&api_key=real-key&password=real-pass",
                    headers={"Authorization": "Bearer real-secret"},
                )
            }
        )
    )
    manager._clients["remote"] = _FakeClient(
        manager.config.servers["remote"],
        [MCPTool("remote", "search", "search", {"type": "object"})],
    )

    status = manager.get_policy_status()

    assert status["servers"][0]["url_redacted"] == (
        "https://example.test/mcp?token=<redacted>&safe=ok&api_key=<redacted>&password=<redacted>"
    )
    assert "real-token" not in str(status)
    assert "real-key" not in str(status)
    assert "real-pass" not in str(status)
    assert "real-secret" not in str(status)
    assert status["servers"][0]["header_keys"] == ["Authorization"]


def test_cli_and_server_expose_mcp_policy_startup_flags():
    import inspect

    import vmlx_engine.cli as cli
    import vmlx_engine.server as server

    cli_source = inspect.getsource(cli)
    server_source = inspect.getsource(server)

    for flag in (
        "--mcp-enabled-servers",
        "--mcp-disabled-servers",
        "--mcp-enabled-tools",
        "--mcp-disabled-tools",
    ):
        assert flag in cli_source
        assert flag in server_source

    for env_name in (
        "VLLM_MLX_MCP_ENABLED_SERVERS",
        "VLLM_MLX_MCP_DISABLED_SERVERS",
        "VLLM_MLX_MCP_ENABLED_TOOLS",
        "VLLM_MLX_MCP_DISABLED_TOOLS",
    ):
        assert env_name in cli_source
        assert env_name in server_source


def test_mcp_schema_merge_drops_mcp_tools_that_collide_with_request_tools():
    import vmlx_engine.server as server

    mcp_tools = [
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "fs__read_file"}},
    ]
    request_tools = [
        server.ToolDefinition(type="function", function={"name": "read_file"}),
    ]

    merged = server._drop_colliding_mcp_tools(mcp_tools, request_tools)

    assert [tool["function"]["name"] for tool in merged] == ["fs__read_file"]


def test_mcp_specific_tool_choice_filters_mcp_tools_by_nested_or_flat_shape():
    import vmlx_engine.server as server

    nested_tools = [
        {"type": "function", "function": {"name": "fs__read_file"}},
        {"type": "function", "function": {"name": "web__fetch"}},
    ]
    flat_tools = [
        {"type": "function", "name": "fs__read_file"},
        {"type": "function", "name": "web__fetch"},
    ]

    nested = server._filter_tools_for_specific_choice(
        nested_tools,
        {"type": "function", "function": {"name": "web__fetch"}},
    )
    flat = server._filter_tools_for_specific_choice(
        flat_tools,
        {"type": "function", "name": "web__fetch"},
    )

    assert [_tool["function"]["name"] for _tool in nested] == ["web__fetch"]
    assert [_tool["name"] for _tool in flat] == ["web__fetch"]


def test_mcp_required_tool_choice_errors_when_policy_disables_all_tools():
    import pytest
    from fastapi import HTTPException
    import vmlx_engine.server as server

    manager = _manager_with_fake_tools()
    policy = MCPPolicy(enabled_tools={"missing__tool"})

    with pytest.raises(HTTPException) as exc_info:
        server._suppress_tool_parsing_when_no_tools(
            manager.get_all_tools_openai(policy=policy),
            "required",
            "Chat Completions",
        )

    assert exc_info.value.status_code == 400
    assert "tool_choice requires at least one available tool" in exc_info.value.detail


def test_mcp_effective_tools_are_used_for_schema_gated_dsml_repair(monkeypatch):
    import vmlx_engine.server as server

    request = server.ChatCompletionRequest(
        model="dsv4-chain-smoke",
        messages=[{"role": "user", "content": "chain"}],
    )
    object.__setattr__(
        request,
        "_vmlx_effective_tools",
        [
            {
                "type": "function",
                "function": {
                    "name": "smoke__add",
                    "description": "Add two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ],
    )
    malformed_dsml = (
        '<｜DSML｜tool_calls>\n'
        '<｜DSML｜invoke name="smoke__add">\n'
        '<｜DSML｜parameter name="a" string="false">2</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="b" string="false">3</｜DSML｜parameter>\n'
        '</｜DSML｜invinvoke>\n'
        '</｜DSML｜tool_calls>'
    )
    monkeypatch.setattr(server, "_tool_call_parser", "dsml")

    cleaned, calls = server._parse_tool_calls_with_parser(malformed_dsml, request)

    assert cleaned == ""
    assert calls is not None
    assert calls[0].function.name == "smoke__add"
    assert json.loads(calls[0].function.arguments) == {"a": 2, "b": 3}


@pytest.mark.asyncio
async def test_server_mcp_endpoints_use_effective_policy(monkeypatch):
    import vmlx_engine.server as server

    manager = _manager_with_fake_tools()
    policy = MCPPolicy(enabled_servers={"fs"}, enabled_tools={"fs__read_file"})
    monkeypatch.setattr(server, "_mcp_manager", manager)
    monkeypatch.setattr(server, "_mcp_policy", policy, raising=False)

    tools_response = await server.list_mcp_tools()
    execute_response = await server.execute_mcp_tool(
        server.MCPExecuteRequest(
            tool_name="fs__write_file",
            arguments={"path": "x", "content": "blocked"},
        )
    )

    by_name = {tool.name: tool for tool in tools_response.tools}
    assert by_name["fs__read_file"].effective is True
    assert by_name["fs__write_file"].effective is False
    assert execute_response.is_error is True
    assert "disabled by MCP policy" in (execute_response.error_message or "")
