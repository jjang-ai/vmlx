#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from vmlx_engine.native_mtp_examples.mtp_runtime_common import print_json


ACTIVATED_RE = re.compile(r"MTP path activated", re.IGNORECASE)
ACCEPT_RE = re.compile(
    r"(?:^|\b)(?:MLLM\s+)?MTP\[(?P<request>[^\]]+)\].*?"
    r"\baccept=(?P<accepted>\d+)(?:/(?P<drafted>\d+))?",
    re.IGNORECASE,
)
FINISH_RE = re.compile(
    r"MLLM\s+MTP\[(?P<request>[^\]]+)\]\s+finish=(?P<finish>\S+)\s+"
    r"cycles=(?P<cycles>\d+)\s+accepted=(?P<accepted>\d+)/(?P<drafted>\d+)",
    re.IGNORECASE,
)
TEXT_FINISH_RE = re.compile(
    r"MTP\[(?P<request>[^\]]+)\]\s+finish=(?P<finish>\S+)\s+"
    r"tokens=(?P<tokens>\d+)\s+cycles=(?P<cycles>\d+)\s+"
    r"accept=(?P<accepted>\d+)/(?P<drafted>\d+)",
    re.IGNORECASE,
)
DEPTH_RE = re.compile(
    r"MLLM\s+MTP\[(?P<request>[^\]]+)\]\s+accept_by_depth\["
    r"d1=(?P<a1>\d+)/(?P<d1>\d+),d2=(?P<a2>\d+)/(?P<d2>\d+),"
    r"d3=(?P<a3>\d+)/(?P<d3>\d+)\]",
    re.IGNORECASE,
)


def _rate(accepted: int, drafted: int | None) -> float | None:
    if not drafted:
        return None
    return accepted / drafted


def _request_row(requests: dict[str, dict[str, Any]], request_id: str) -> dict[str, Any]:
    return requests.setdefault(request_id, {"request_id": request_id})


def parse_log_lines(lines: Iterable[str]) -> dict[str, Any]:
    activations: list[dict[str, Any]] = []
    requests: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(lines, start=1):
        if ACTIVATED_RE.search(line):
            activations.append({"line": line_number, "text": line.rstrip("\n")})

        finish = FINISH_RE.search(line)
        if finish:
            request = _request_row(requests, finish.group("request"))
            accepted = int(finish.group("accepted"))
            drafted = int(finish.group("drafted"))
            request.update(
                {
                    "finish": finish.group("finish"),
                    "cycles": int(finish.group("cycles")),
                    "accepted_tokens": accepted,
                    "drafted_tokens": drafted,
                    "acceptance_rate": _rate(accepted, drafted),
                }
            )
            continue

        text_finish = TEXT_FINISH_RE.search(line)
        if text_finish:
            request = _request_row(requests, text_finish.group("request"))
            accepted = int(text_finish.group("accepted"))
            drafted = int(text_finish.group("drafted"))
            request.update(
                {
                    "finish": text_finish.group("finish"),
                    "tokens": int(text_finish.group("tokens")),
                    "cycles": int(text_finish.group("cycles")),
                    "accepted_tokens": accepted,
                    "drafted_tokens": drafted,
                    "acceptance_rate": _rate(accepted, drafted),
                }
            )
            continue

        accept = ACCEPT_RE.search(line)
        if accept:
            request = _request_row(requests, accept.group("request"))
            accepted = int(accept.group("accepted"))
            drafted = int(accept.group("drafted")) if accept.group("drafted") else None
            request.setdefault("accept_events", []).append(
                {
                    "line": line_number,
                    "accepted": accepted,
                    "drafted": drafted,
                    "acceptance_rate": _rate(accepted, drafted),
                }
            )

        depth = DEPTH_RE.search(line)
        if depth:
            request = _request_row(requests, depth.group("request"))
            accepted_by_depth = [
                int(depth.group("a1")),
                int(depth.group("a2")),
                int(depth.group("a3")),
            ]
            drafted_by_depth = [
                int(depth.group("d1")),
                int(depth.group("d2")),
                int(depth.group("d3")),
            ]
            request.update(
                {
                    "accepted_by_depth": accepted_by_depth,
                    "drafted_by_depth": drafted_by_depth,
                    "acceptance_by_depth": [
                        _rate(accepted, drafted)
                        for accepted, drafted in zip(accepted_by_depth, drafted_by_depth)
                    ],
                }
            )

    request_rows = list(requests.values())
    total_accepted = 0
    total_drafted = 0
    for row in request_rows:
        if row.get("accepted_tokens") is not None or row.get("drafted_tokens") is not None:
            total_accepted += int(row.get("accepted_tokens") or 0)
            total_drafted += int(row.get("drafted_tokens") or 0)
            continue
        for event in row.get("accept_events") or []:
            total_accepted += int(event.get("accepted") or 0)
            total_drafted += int(event.get("drafted") or 0)
    return {
        "mtp_path_activated": bool(activations),
        "acceptance_telemetry_present": bool(request_rows),
        "activation_events": activations,
        "request_count": len(request_rows),
        "requests": request_rows,
        "totals": {
            "accepted_tokens": total_accepted,
            "drafted_tokens": total_drafted,
            "acceptance_rate": _rate(total_accepted, total_drafted),
        },
    }


def read_lines(paths: list[Path]) -> list[str]:
    if not paths:
        return list(sys.stdin)
    lines: list[str] = []
    for path in paths:
        lines.extend(path.read_text(encoding="utf-8", errors="replace").splitlines())
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse native MTP activation and acceptance telemetry logs."
    )
    parser.add_argument("log", nargs="*", type=Path, help="Log files; stdin if omitted")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-active", action="store_true")
    parser.add_argument("--require-acceptance", action="store_true")
    args = parser.parse_args(argv)

    report = parse_log_lines(read_lines(args.log))
    if args.json:
        print_json(report)
    else:
        print(f"MTP path activated: {report['mtp_path_activated']}")
        print(f"Requests with telemetry: {report['request_count']}")
        print(f"Totals: {report['totals']}")
        for row in report["requests"]:
            print(row)
    if args.require_active and not report["mtp_path_activated"]:
        print("Missing required native MTP activation log: MTP path activated", file=sys.stderr)
        return 2
    if args.require_acceptance and not report["acceptance_telemetry_present"]:
        print("Missing required native MTP acceptance telemetry: MTP[...] accept=...", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
