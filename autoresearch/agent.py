"""
Autoresearch agent using vmlx_engine as the backend.

This agent uses the vmlx_engine SimpleEngine with MLX models for
Apple Silicon GPU acceleration. Tools are executed locally without
requiring LangSmith or external APIs.

Usage:
    python agent.py "What is 25 * 37?"
"""

import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path so vmlx_engine imports work
VMLX_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(VMLX_ROOT))

# vmlx_engine imports
from vmlx_engine.api import parse_tool_calls
from vmlx_engine.engine.simple import SimpleEngine

# ---------------------------------------------------------------------------
# Tools (add, remove, or modify these)
# ---------------------------------------------------------------------------


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports basic arithmetic, powers, roots, and common math functions."""
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units. Supports temperature (C/F/K), distance (km/mi/m/ft), and weight (kg/lb)."""
    conversions = {
        ("km", "mi"): lambda v: v * 0.621371,
        ("mi", "km"): lambda v: v * 1.60934,
        ("m", "ft"): lambda v: v * 3.28084,
        ("ft", "m"): lambda v: v * 0.3048,
        ("kg", "lb"): lambda v: v * 2.20462,
        ("lb", "kg"): lambda v: v * 0.453592,
        ("c", "f"): lambda v: v * 9 / 5 + 32,
        ("f", "c"): lambda v: (v - 32) * 5 / 9,
        ("c", "k"): lambda v: v + 273.15,
        ("k", "c"): lambda v: v - 273.15,
        ("f", "k"): lambda v: (v - 32) * 5 / 9 + 273.15,
        ("k", "f"): lambda v: (v - 273.15) * 9 / 5 + 32,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.2f} {to_unit}"
    return f"Error: Cannot convert from {from_unit} to {to_unit}"


# Tool registry
def get_tool(name: str) -> Any:
    """Get a tool by name."""
    tools = {t.__name__: t for t in TOOLS}
    return tools.get(name)


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful assistant. Answer questions accurately and concisely.

CRITICAL: When a question involves any of the following, you MUST use the appropriate tool:
- Math: arithmetic, powers, roots, logarithms, trigonometry
- Unit conversions: temperature (C/F/K), distance (km/mi/m/ft), weight (kg/lb)

Available tools:
1. calculator(expression): Evaluate mathematical expressions
   Examples: "25 * 37", "15^2", "sqrt(144)", "log10(1000)", "sin(pi/6)"
   
2. unit_converter(value, from_unit, to_unit): Convert between units
   Examples: (100, "km", "mi"), (32, "f", "c"), (5, "kg", "lb")

Rules:
- ALWAYS use tools for math calculations - never calculate in your head
- Provide ONLY the final answer after using tools
- Be concise and direct

Examples of correct behavior:
User: What is 25 * 37?
Assistant: <tool_call>{"name": "calculator", "arguments": {"expression": "25 * 37"}}</tool_call>
→ Tool returns: 925
Assistant: 925

User: Convert 100 km to miles
Assistant: <tool_call>{"name": "unit_converter", "arguments": {"value": 100, "from_unit": "km", "to_unit": "mi"}}</tool_call>
→ Tool returns: 62.14
Assistant: 62.14 miles"""

MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
TEMPERATURE = 0

# Tool definitions in OpenAI format for the engine
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Supports basic arithmetic, powers, roots, and common math functions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "Convert between common units. Supports temperature (C/F/K), distance (km/mi/m/ft), and weight (kg/lb).",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The value to convert",
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "The source unit (e.g., 'km', 'c', 'kg')",
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "The target unit (e.g., 'mi', 'f', 'lb')",
                    },
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]

TOOLS = [calculator, unit_converter]

# ---------------------------------------------------------------------------
# Build the agent
# ---------------------------------------------------------------------------


class VmlxAgent:
    """Agent wrapper around vmlx_engine SimpleEngine."""

    def __init__(self, model_name: str = MODEL, temperature: float = TEMPERATURE):
        self.engine = SimpleEngine(model_name)
        self.temperature = temperature
        self._started = False

    async def start(self):
        """Start the engine."""
        if not self._started:
            await self.engine.start()
            self._started = True

    async def stop(self):
        """Stop the engine."""
        if self._started:
            await self.engine.stop()
            self._started = False

    async def run(self, question: str) -> dict:
        """
        Run the agent on a question.

        Returns:
            Dict with 'response' and 'tools_used' keys.
        """
        await self.start()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        tools_used = []
        max_iterations = 5

        for _ in range(max_iterations):
            # Get response from model
            output = await self.engine.chat(
                messages=messages,
                max_tokens=512,
                temperature=self.temperature,
                tools=TOOL_DEFINITIONS,
            )

            response_text = output.text

            # Check for tool calls in the response
            cleaned_text, tool_calls = parse_tool_calls(response_text)

            if not tool_calls:
                # No tool calls, return the response
                return {
                    "response": cleaned_text or response_text,
                    "tools_used": tools_used,
                }

            # Execute tool calls
            for tc in tool_calls:
                func = tc.function
                tool_name = func.name
                tools_used.append(tool_name)

                # Parse arguments
                try:
                    args = json.loads(func.arguments) if func.arguments else {}
                except json.JSONDecodeError:
                    args = {}

                # Execute the tool
                tool_func = get_tool(tool_name)
                if tool_func:
                    try:
                        result = tool_func(**args)
                    except Exception as e:
                        result = f"Error executing {tool_name}: {e}"
                else:
                    result = f"Error: Tool '{tool_name}' not found"

                # Add assistant message with tool call
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": func.arguments,
                            },
                        }
                    ],
                })

                # Add tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

        # Max iterations reached
        return {
            "response": "Maximum iterations reached without a final answer.",
            "tools_used": tools_used,
        }


# Global agent instance
_agent: VmlxAgent | None = None


def build_agent() -> VmlxAgent:
    """Build and return the agent."""
    global _agent
    if _agent is None:
        _agent = VmlxAgent()
    return _agent


# ---------------------------------------------------------------------------
# Functions called by run_eval.py
#
# run_agent_with_tools() is the contract with the eval harness. It must
# return a dict that the evaluators can score.
# ---------------------------------------------------------------------------


def run_agent(question: str) -> str:
    """Run the agent on a single question and return the response text."""
    result = run_agent_with_tools(question)
    return result["response"]


def run_agent_with_tools(question: str) -> dict:
    """Run the agent and return response + tool usage info for evaluation."""
    agent = build_agent()

    try:
        # Run the async function
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, use run_coroutine_threadsafe
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, agent.run(question))
                result = future.result()
        else:
            result = loop.run_until_complete(agent.run(question))

        return result
    except Exception as e:
        return {"response": f"ERROR: {e}", "error": True, "tools_used": []}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py <question>")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    print(run_agent(question))
