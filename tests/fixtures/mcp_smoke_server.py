# SPDX-License-Identifier: Apache-2.0
"""Tiny local MCP stdio server used by live MCP policy smoke tests."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("vmlx-mcp-smoke")


@mcp.tool()
def echo(text: str) -> str:
    """Return the provided text."""
    return text


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    mcp.run()
