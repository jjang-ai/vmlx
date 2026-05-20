# SPDX-License-Identifier: Apache-2.0
"""MCP policy contracts for server-side tool visibility and execution."""

import asyncio
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
