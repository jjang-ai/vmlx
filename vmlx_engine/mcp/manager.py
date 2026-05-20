# SPDX-License-Identifier: Apache-2.0
"""
MCP Client Manager for handling multiple MCP server connections.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from .client import MCPClient
from .tools import merge_tools, mcp_tools_to_openai, openai_call_to_mcp
from .types import (
    MCPConfig,
    MCPPolicy,
    MCPServerStatus,
    MCPTool,
    MCPToolResult,
)

logger = logging.getLogger(__name__)


_URL_SECRET_QUERY_RE = re.compile(
    r"([?&][^=&#]*(?:key|token|secret|password)[^=&#]*=)[^&#]*",
    re.IGNORECASE,
)


def _redact_url_secrets(url: Optional[str]) -> Optional[str]:
    """Redact common secret-bearing query parameters from diagnostics URLs."""
    if not url:
        return url
    return _URL_SECRET_QUERY_RE.sub(r"\1<redacted>", url)


class MCPClientManager:
    """
    Manages multiple MCP server connections.

    Provides a unified interface for:
    - Connecting to multiple MCP servers
    - Discovering and aggregating tools
    - Executing tool calls
    - Managing connection lifecycle
    """

    def __init__(self, config: MCPConfig):
        """
        Initialize MCP Client Manager.

        Args:
            config: MCP configuration with server definitions
        """
        self.config = config
        self._clients: Dict[str, MCPClient] = {}
        self._started = False
        self._lock = asyncio.Lock()

        # Create clients for each server
        for name, server_config in config.servers.items():
            self._clients[name] = MCPClient(server_config)

    @property
    def is_started(self) -> bool:
        """Check if manager has been started."""
        return self._started

    async def start(self):
        """
        Start the manager and connect to all enabled servers.

        Connections are made in parallel for faster startup.
        """
        async with self._lock:
            if self._started:
                return

            logger.info(
                f"Starting MCP client manager with {len(self._clients)} servers"
            )

            # MCP SDK stdio transports bind async context managers to the
            # entering task. Keep connect/disconnect in the manager task so
            # shutdown can exit those contexts cleanly.
            for client in [c for c in self._clients.values() if c.config.enabled]:
                try:
                    result = await client.connect()
                except Exception as exc:
                    logger.error(f"Failed to connect to '{client.name}': {exc}")
                else:
                    if result:
                        logger.info(f"Connected to '{client.name}'")

            self._started = True

            # Log summary
            connected = sum(1 for c in self._clients.values() if c.is_connected)
            total_tools = sum(len(c.tools) for c in self._clients.values())
            logger.info(
                f"MCP manager started: {connected}/{len(self._clients)} servers, "
                f"{total_tools} tools available"
            )

    async def stop(self):
        """Stop the manager and disconnect from all servers."""
        async with self._lock:
            if not self._started:
                return

            logger.info("Stopping MCP client manager")

            for client in self._clients.values():
                try:
                    await client.disconnect()
                except Exception as exc:
                    logger.warning(f"Failed to disconnect '{client.name}': {exc}")

            self._started = False
            logger.info("MCP client manager stopped")

    def _tool_effective(self, tool: MCPTool, policy: Optional[MCPPolicy]) -> bool:
        if policy is None:
            return True
        return policy.tool_enabled(tool.full_name, tool.server_name, tool.name)

    def _server_effective(self, server_name: str, policy: Optional[MCPPolicy]) -> bool:
        if policy is None:
            return True
        return policy.server_enabled(server_name)

    def get_all_tools(self, policy: Optional[MCPPolicy] = None) -> List[MCPTool]:
        """
        Get all tools from all connected servers.

        Returns:
            List of MCPTool instances
        """
        tools = []
        for client in self._clients.values():
            if client.is_connected and self._server_effective(client.name, policy):
                tools.extend(
                    tool for tool in client.tools if self._tool_effective(tool, policy)
                )
        return tools

    def get_all_tools_openai(
        self,
        policy: Optional[MCPPolicy] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all tools in OpenAI function calling format.

        Returns:
            List of OpenAI-compatible tool definitions
        """
        return mcp_tools_to_openai(self.get_all_tools(policy=policy))

    def get_merged_tools(
        self,
        user_tools: Optional[List[Dict[str, Any]]] = None,
        policy: Optional[MCPPolicy] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get MCP tools merged with user-provided tools.

        User tools take precedence on name conflicts.

        Args:
            user_tools: Optional user-provided tools in OpenAI format

        Returns:
            Combined list of tools in OpenAI format
        """
        return merge_tools(self.get_all_tools(policy=policy), user_tools)

    def get_server_status(self) -> List[MCPServerStatus]:
        """
        Get status of all servers.

        Returns:
            List of MCPServerStatus for each server
        """
        return [client.get_status() for client in self._clients.values()]

    def get_client(self, server_name: str) -> Optional[MCPClient]:
        """
        Get client for a specific server.

        Args:
            server_name: Name of the server

        Returns:
            MCPClient instance or None if not found
        """
        return self._clients.get(server_name)

    def get_policy_status(
        self,
        policy: Optional[MCPPolicy] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return redacted server/tool state with effective policy flags."""
        servers: List[Dict[str, Any]] = []
        tools: List[Dict[str, Any]] = []

        for client in self._clients.values():
            cfg = client.config
            server_enabled = self._server_effective(client.name, policy)
            servers.append(
                {
                    "name": client.name,
                    "state": client.get_status().state.value,
                    "transport": cfg.transport.value,
                    "enabled": server_enabled,
                    "configured": bool(getattr(cfg, "enabled", True)),
                    "tools_count": len(client.tools),
                    "error": client.get_status().error,
                    "last_connected": client.get_status().last_connected,
                    "command_redacted": cfg.command,
                    "url_redacted": _redact_url_secrets(cfg.url),
                    "env_keys": sorted((cfg.env or {}).keys()),
                    "header_keys": sorted((cfg.headers or {}).keys()),
                    "skip_security_validation": bool(
                        getattr(cfg, "skip_security_validation", False)
                    ),
                }
            )
            for tool in client.tools:
                effective = bool(client.is_connected) and self._tool_effective(
                    tool, policy
                )
                tools.append(
                    {
                        "name": tool.full_name,
                        "tool_name": tool.name,
                        "server": tool.server_name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                        "enabled": self._tool_effective(tool, policy),
                        "effective": effective,
                        "transport": cfg.transport.value,
                        "server_state": client.get_status().state.value,
                        "error": client.get_status().error,
                    }
                )

        return {"servers": servers, "tools": tools}

    async def execute_tool(
        self,
        full_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
        policy: Optional[MCPPolicy] = None,
    ) -> MCPToolResult:
        """
        Execute a tool by its full name (server__tool).

        Args:
            full_name: Full tool name with server prefix
            arguments: Tool arguments
            timeout: Optional timeout in seconds

        Returns:
            MCPToolResult with the result or error
        """
        # Parse full name
        server_name, tool_name, _ = openai_call_to_mcp(
            {"function": {"name": full_name, "arguments": "{}"}}
        )

        # If no server prefix, try to find the tool
        if not server_name:
            server_name = self._find_tool_server(full_name)
            tool_name = full_name

        if not server_name:
            return MCPToolResult(
                tool_name=full_name,
                content=None,
                is_error=True,
                error_message=f"Tool '{full_name}' not found in any connected server",
            )

        # Get client
        client = self._clients.get(server_name)
        if not client:
            return MCPToolResult(
                tool_name=full_name,
                content=None,
                is_error=True,
                error_message=f"Server '{server_name}' not found",
            )

        if not client.is_connected:
            return MCPToolResult(
                tool_name=full_name,
                content=None,
                is_error=True,
                error_message=f"Server '{server_name}' is not connected",
            )

        full_tool_name = f"{server_name}__{tool_name}"
        if policy is not None and not policy.tool_enabled(
            full_tool_name, server_name, tool_name
        ):
            return MCPToolResult(
                tool_name=full_name,
                content=None,
                is_error=True,
                error_message=f"Tool '{full_tool_name}' is disabled by MCP policy",
            )

        # Execute tool
        return await client.call_tool(
            tool_name,
            arguments,
            timeout=timeout or self.config.default_timeout,
        )

    async def execute_tool_call(
        self,
        tool_call: Dict[str, Any],
        timeout: Optional[float] = None,
        policy: Optional[MCPPolicy] = None,
    ) -> MCPToolResult:
        """
        Execute a tool call from OpenAI format.

        Args:
            tool_call: OpenAI tool call object
            timeout: Optional timeout in seconds

        Returns:
            MCPToolResult with the result or error
        """
        server_name, tool_name, arguments = openai_call_to_mcp(tool_call)

        if server_name:
            full_name = f"{server_name}__{tool_name}"
        else:
            full_name = tool_name

        return await self.execute_tool(full_name, arguments, timeout, policy=policy)

    def _find_tool_server(self, tool_name: str) -> Optional[str]:
        """
        Find which server has a tool by name.

        Args:
            tool_name: Tool name (without server prefix)

        Returns:
            Server name or None if not found
        """
        for client in self._clients.values():
            if client.is_connected:
                for tool in client.tools:
                    if tool.name == tool_name:
                        return client.name
        return None

    async def refresh_tools(self):
        """Refresh tools from all connected servers."""
        tasks = [
            client.refresh_tools()
            for client in self._clients.values()
            if client.is_connected
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def reconnect(self, server_name: Optional[str] = None):
        """
        Reconnect to server(s).

        Args:
            server_name: Specific server to reconnect, or None for all
        """
        async with self._lock:
            if server_name:
                client = self._clients.get(server_name)
                if client:
                    await client.disconnect()
                    await client.connect()
            else:
                # Reconnect all
                for client in self._clients.values():
                    await client.disconnect()
                    await client.connect()
