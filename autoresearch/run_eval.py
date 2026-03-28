"""
Evaluation harness for vmlx autoresearch agents.

Runs the agent against a dataset and scores it using local evaluation.
This version does NOT require LangSmith - all evaluation happens locally.

Usage:
    python run_eval.py
    python run_eval.py --dataset dataset.json --prefix "experiment-1"

Output format (parsed by the experiment loop):
    ---
    avg_correctness: 0.850000
    avg_helpfulness: 0.900000
    avg_tool_usage: 0.750000
    overall_score: 0.833333
    num_examples: 20
    num_errors: 0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# We can optionally use an LLM as judge via vmlx_engine
# or use simple heuristics for evaluation

SCRIPT_DIR = Path(__file__).parent

# Add parent directory (vmlx root) to path for vmlx_engine imports
VMLX_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(VMLX_ROOT))


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CUSTOMIZE 1: Run function — how to call your agent
# ---------------------------------------------------------------------------


def run_agent_for_eval(inputs: dict) -> dict:
    """Call the agent and return outputs for evaluators to score."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from agent import run_agent_with_tools

    try:
        return run_agent_with_tools(inputs["question"])
    except Exception as e:
        return {"response": f"ERROR: {e}", "error": True, "tools_used": []}


# ---------------------------------------------------------------------------
# CUSTOMIZE 2: Evaluators
#
# Each evaluator takes (outputs, example) and returns {"score": number, "comment": str}.
# - outputs = the dict your run function returned
# - example = the dataset example with "inputs" and "outputs"
# ---------------------------------------------------------------------------


def correctness_evaluator(outputs: dict, example: dict) -> dict:
    """Code-based: check if expected answer appears in the response."""
    if outputs.get("error"):
        return {"score": 0, "comment": "Agent returned an error"}

    response = outputs.get("response", "").lower()
    expected = example.get("outputs", {}).get("answer", "").lower()

    if not expected:
        return {"score": 1, "comment": "No expected answer defined"}

    # Simple heuristic: check if expected answer appears in response
    if expected in response:
        return {"score": 1, "comment": f"Expected answer '{expected}' found in response"}

    # Try to extract numeric answer and compare
    import re

    # Extract numbers from both
    response_nums = re.findall(r"-?\d+\.?\d*", response)
    expected_nums = re.findall(r"-?\d+\.?\d*", expected)

    if expected_nums and response_nums:
        # Check if any expected number appears in response
        for en in expected_nums:
            for rn in response_nums:
                try:
                    if abs(float(en) - float(rn)) < 0.01:
                        return {
                            "score": 1,
                            "comment": f"Numeric answer match: {en} ≈ {rn}",
                        }
                except ValueError:
                    continue

    return {"score": 0, "comment": f"Expected '{expected}' not found in response: '{response[:100]}...'"}


def helpfulness_evaluator(outputs: dict, example: dict) -> dict:
    """Code-based: check response length and quality heuristics."""
    if outputs.get("error"):
        return {"score": 0, "comment": "Agent returned an error"}

    response = outputs.get("response", "")
    question = example.get("inputs", {}).get("question", "")

    # Heuristics for helpfulness
    score = 1
    comments = []

    # Should not be too short
    if len(response) < 10:
        score = 0
        comments.append("Response too short")

    # Should not be an error message
    if response.lower().startswith("error"):
        score = 0
        comments.append("Response is an error message")

    # Should contain an answer (ends with punctuation or number)
    if not any(response.rstrip().endswith(c) for c in ".!?"):
        comments.append("Response doesn't end with punctuation")

    # Direct answer check - does it answer the question?
    question_words = set(question.lower().split())
    response_words = set(response.lower().split())
    overlap = len(question_words & response_words)
    if overlap < 2 and len(question_words) > 3:
        comments.append("Response may not address the question")

    comment = "; ".join(comments) if comments else "Response appears helpful"
    return {"score": score, "comment": comment}


def tool_usage_evaluator(outputs: dict, example: dict) -> dict:
    """Code-based: did the agent use tools when expected?"""
    if outputs.get("error"):
        return {"score": 0, "comment": "Agent returned an error"}

    expected_tool_use = example.get("outputs", {}).get("expected_tool_use", None)
    if expected_tool_use is None:
        return {"score": 1, "comment": "No tool usage expectation defined"}

    tools_used = outputs.get("tools_used", [])
    actually_used = len(tools_used) > 0

    if expected_tool_use and not actually_used:
        return {"score": 0, "comment": "Expected tool use but agent didn't use tools"}
    if not expected_tool_use and actually_used:
        return {"score": 0, "comment": f"Agent used tools when not expected: {tools_used}"}
    return {"score": 1, "comment": f"Tool usage matched expectations (tools: {tools_used})"}


# ---------------------------------------------------------------------------
# CUSTOMIZE 3: Which evaluators to run
# ---------------------------------------------------------------------------

EVALUATORS = [correctness_evaluator, helpfulness_evaluator, tool_usage_evaluator]


# ---------------------------------------------------------------------------
# Evaluation infrastructure
# ---------------------------------------------------------------------------


def run_evaluation(dataset_path: str, prefix: str) -> dict[str, Any]:
    """Run evaluation and return summary."""
    examples = load_dataset(dataset_path)

    scores_by_evaluator: dict[str, list[float]] = {}
    num_errors = 0
    num_examples = len(examples)

    for example in examples:
        # Run the agent
        outputs = run_agent_for_eval(example["inputs"])

        if outputs.get("error"):
            num_errors += 1

        # Run evaluators
        for evaluator in EVALUATORS:
            result = evaluator(outputs, example)
            scores_by_evaluator.setdefault(evaluator.__name__, []).append(
                result["score"]
            )

    # Calculate averages
    averages = {}
    for name, scores in scores_by_evaluator.items():
        if scores:
            # Convert evaluator name to metric name
            metric_name = name.replace("_evaluator", "")
            averages[metric_name] = sum(scores) / len(scores)

    overall = sum(averages.values()) / len(averages) if averages else 0

    summary = {f"avg_{k}": v for k, v in averages.items()}
    summary["overall_score"] = overall
    summary["num_examples"] = num_examples
    summary["num_errors"] = num_errors
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run autoresearch agent evaluation")
    parser.add_argument(
        "--dataset",
        default=str(SCRIPT_DIR / "dataset.json"),
        help="Path to dataset JSON",
    )
    parser.add_argument(
        "--prefix",
        default="vmlx-autoresearch",
        help="Experiment prefix",
    )
    args = parser.parse_args()

    summary = run_evaluation(args.dataset, args.prefix)

    print("---")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
