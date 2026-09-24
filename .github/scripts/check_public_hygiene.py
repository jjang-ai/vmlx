"""Reject private working documents in the Git index before publication."""
from pathlib import PurePosixPath
import re
import subprocess
import sys

PRIVATE_PREFIXES = (
    "docs/internal/", "docs/plans/", "docs/runtime/", "docs/superpowers/",
    "research/", "notes/", "botes/", ".agents/", ".claude/", ".codex/",
    "proof-artifacts/", "screen-recordings/", "reports/", "ralph_runner/results/",
)
PRIVATE_NAMES = {
    "AGENTS.md", "CLAUDE.md", "HANDOFF.md", "PLAN.md", "TODO.md",
    "AUDIT_CHECKLIST.md", "INVESTIGATION_LOG.md", "PROMPT.md",
}
DOC_SUFFIXES = {".md", ".rst", ".txt"}


def violations(path: str, content: bytes = b"") -> list[str]:
    p = PurePosixPath(path)
    reasons = []
    if path.startswith(PRIVATE_PREFIXES) or p.name in PRIVATE_NAMES:
        reasons.append("private document or evidence path")
    if p.suffix.lower() in DOC_SUFFIXES:
        text = content.decode("utf-8", errors="replace")
        if re.search(r"(?im)^>?.*(?:required sub-skill|for claude:|internal only|do not publish).*$", text):
            reasons.append("internal document marker")
        if re.search(r"/Users/eric/|/Volumes/Erics|erics-m5-max", text):
            reasons.append("maintainer-specific machine path")
    return reasons


def main() -> int:
    paths = subprocess.check_output(["git", "ls-files", "-z"]).split(b"\0")
    failures = []
    for raw in filter(None, paths):
        path = raw.decode("utf-8")
        # Read staged bytes, not a possibly sanitized working copy.
        content = b""
        if PurePosixPath(path).suffix.lower() in DOC_SUFFIXES:
            content = subprocess.check_output(["git", "show", ":" + path])
        reasons = violations(path, content)
        if reasons:
            failures.append((path, reasons))
    for path, reasons in failures:
        print(path + ": " + "; ".join(reasons), file=sys.stderr)
    if failures:
        return 1
    print("Public document boundary: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
