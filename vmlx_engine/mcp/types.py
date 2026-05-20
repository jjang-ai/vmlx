# SPDX-License-Identifier: Apache-2.0
"""
Type definitions for MCP client support.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class MCPTransport(str, Enum):
    """Supported MCP transport types."""

    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"  # Streamable HTTP (modern remote MCP servers like Exa)


class MCPServerState(str, Enum):
    """MCP server connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    transport: MCPTransport = MCPTransport.STDIO

    # For stdio transport
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None

    # For SSE / HTTP transport
    url: Optional[str] = None
    # HTTP headers for remote transports (auth tokens, API keys for Exa,
    # GitHub remote MCP, etc.). Ignored for stdio.
    headers: Optional[Dict[str, str]] = None

    # Common options
    enabled: bool = True
    timeout: float = 30.0

    # Security options
    skip_security_validation: bool = False  # WARNING: Only for development!

    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.transport, str):
            self.transport = MCPTransport(self.transport)

        if self.transport == MCPTransport.STDIO:
            if not self.command:
                raise ValueError(
                    f"MCP server '{self.name}': stdio transport requires 'command'"
                )
        elif self.transport in (MCPTransport.SSE, MCPTransport.HTTP):
            if not self.url:
                raise ValueError(
                    f"MCP server '{self.name}': {self.transport.value} transport requires 'url'"
                )

        # Security validation
        self._validate_security()

    def _validate_security(self) -> None:
        """Validate security of the configuration."""
        from .security import validate_mcp_server_config, MCPSecurityError

        if self.skip_security_validation:
            import logging

            logging.getLogger(__name__).warning(
                f"MCP server '{self.name}': Security validation SKIPPED. "
                f"This is dangerous and should only be used in development!"
            )
            return

        try:
            validate_mcp_server_config(
                server_name=self.name,
                command=self.command,
                args=self.args,
                env=self.env,
                url=self.url,
            )
        except MCPSecurityError as e:
            raise ValueError(str(e)) from e


@dataclass
class MCPConfig:
    """Root configuration for MCP client."""

    servers: Dict[str, MCPServerConfig] = field(default_factory=dict)
    max_tool_calls: int = 10
    default_timeout: float = 30.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPConfig":
        """Create config from dictionary."""
        servers = {}
        for name, server_data in data.get("servers", data.get("mcpServers", {})).items():
            server_data["name"] = name
            servers[name] = MCPServerConfig(**server_data)

        return cls(
            servers=servers,
            max_tool_calls=data.get("max_tool_calls", 10),
            default_timeout=data.get("default_timeout", 30.0),
        )


@dataclass
class MCPPolicy:
    """Session-level allow/deny policy for effective MCP tools."""

    enabled_servers: Optional[Set[str]] = None
    disabled_servers: Set[str] = field(default_factory=set)
    enabled_tools: Optional[Set[str]] = None
    disabled_tools: Set[str] = field(default_factory=set)

    @staticmethod
    def _normalize(values: Optional[Any]) -> Optional[Set[str]]:
        if values is None:
            return None
        if isinstance(values, str):
            raw = values.replace("\n", ",").split(",")
        else:
            raw = list(values)
        normalized = {str(v).strip() for v in raw if str(v).strip()}
        return normalized

    @classmethod
    def from_values(
        cls,
        *,
        enabled_servers: Optional[Any] = None,
        disabled_servers: Optional[Any] = None,
        enabled_tools: Optional[Any] = None,
        disabled_tools: Optional[Any] = None,
    ) -> "MCPPolicy":
        return cls(
            enabled_servers=cls._normalize(enabled_servers),
            disabled_servers=cls._normalize(disabled_servers) or set(),
            enabled_tools=cls._normalize(enabled_tools),
            disabled_tools=cls._normalize(disabled_tools) or set(),
        )

    def server_enabled(self, server_name: str) -> bool:
        if server_name in self.disabled_servers:
            return False
        if self.enabled_servers is not None and server_name not in self.enabled_servers:
            return False
        return True

    def tool_enabled(self, full_name: str, server_name: str, tool_name: str) -> bool:
        if not self.server_enabled(server_name):
            return False
        names = {full_name, tool_name}
        if names & self.disabled_tools:
            return False
        if self.enabled_tools is not None and not (names & self.enabled_tools):
            return False
        return True


@dataclass
class MCPTool:
    """Normalized tool representation from MCP server."""

    server_name: str
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Get namespaced tool name (server__tool)."""
        return f"{self.server_name}__{self.name}"

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.full_name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class MCPToolResult:
    """Result from a tool execution."""

    tool_name: str
    content: Any
    is_error: bool = False
    error_message: Optional[str] = None

    def to_message(self, tool_call_id: str) -> Dict[str, Any]:
        """Convert to OpenAI tool result message format."""
        if self.is_error:
            content = f"Error: {self.error_message}"
        elif isinstance(self.content, str):
            content = self.content
        else:
            import json
            try:
                content = json.dumps(self.content, default=str)
            except Exception:
                content = str(self.content)

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }


@dataclass
class MCPServerStatus:
    """Status of an MCP server connection."""

    name: str
    state: MCPServerState
    transport: MCPTransport
    tools_count: int = 0
    error: Optional[str] = None
    last_connected: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "name": self.name,
            "state": self.state.value,
            "transport": self.transport.value,
            "tools_count": self.tools_count,
            "error": self.error,
            "last_connected": self.last_connected,
        }
