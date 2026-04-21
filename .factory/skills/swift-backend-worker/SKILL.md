---
name: swift-backend-worker
description: Swift backend worker for vMLX server features — routes, engine logic, MLX tensor operations
---

# Swift Backend Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use this skill for features that modify:
- HTTP route handlers in `Sources/vMLXServer/Routes/`
- Engine/streaming logic in `Sources/vMLXEngine/`
- Core generation/evaluation in `Sources/vMLXLMCommon/Evaluate.swift`
- SSE encoding in `Sources/vMLXServer/SSEEncoder.swift`
- Package dependencies in `Package.swift`

## Required Skills

- `test-driven-development` — For writing tests before implementation. MUST be invoked before writing any implementation code.
- `systematic-debugging` — For investigating build failures, test failures, or unexpected behavior.

## Work Procedure

### 1. Understand the Feature
- Read the feature description in `features.json`
- Read `AGENTS.md` for mission boundaries and conventions
- Read `.factory/library/architecture.md` for system design context
- Read relevant source files to understand current implementation

### 2. Write Tests First (TDD)
- Invoke the `test-driven-development` skill
- Write failing tests that exercise the new behavior
- For route features: add e2e harness cases or curl-based assertions
- For core logic: add unit tests (even if `swift test` currently fails, write the tests)
- Tests must fail before implementation begins (red phase)

### 3. Implement the Feature
- Make the smallest change that makes tests pass (green phase)
- Follow existing code style and patterns in the codebase
- For performance-critical code (logprobs, prefill):
  - Ensure normal-path (no logprobs/echo) is completely unchanged
  - Use batched tensor operations, not per-token loops
  - Use `argPartition` instead of `argSort` for top-K
  - Document performance implications in comments
- For route changes:
  - Use existing JSON serialization pattern (`[String: Any]` + `JSONSerialization`)
  - Use existing error response pattern (`errorJSON()`)
  - Handle missing/invalid parameters gracefully

### 4. Build and Typecheck
- Run `swift build --product vmlxctl`
- Fix all compilation errors
- Run `scripts/lint-typed-mlx-scalars.sh` if modifying MLX array code

### 5. Manual Verification
- Start the server: `.build/debug/vmlxctl serve --model <model-path> --port 8080`
- Test the feature with `curl` requests
- For streaming: capture SSE output and verify chunk structure
- Verify edge cases: empty input, invalid params, no model loaded
- Stop the server cleanly after testing

### 6. E2E Harness Verification (if applicable)
- If the feature affects endpoints tested by the e2e harness, run relevant harness cases
- Command: `tests/e2e/harness.sh <model-path> 8080 <suite>`
- Document which cases pass/fail

### 7. Commit
- Commit with a descriptive message following the repo's commit style
- Include the feature ID in the commit message if applicable

## Example Handoff

```json
{
  "salientSummary": "Implemented GET /tokenizer_info and POST /tokenize/detokenize endpoints. Added tokenizer access via ModelContainer.perform(). All endpoints return correct metadata and handle no-model state with 503.",
  "whatWasImplemented": "Three new OpenAI-compatible routes in OpenAIRoutes.swift: /tokenizer_info returns eos_token/bos_token/pad_token/chat_template; /tokenize encodes text to token IDs; /detokenize decodes token IDs to text. Tokenizer accessed via engine.loaded?.perform { $0.tokenizer }. Error handling for no-model (503) and bad requests (400).",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      { "command": "swift build --product vmlxctl", "exitCode": 0, "observation": "Build succeeded with no warnings" },
      { "command": "curl -s http://localhost:8080/v1/tokenizer_info", "exitCode": 0, "observation": "Returned JSON with eos_token, bos_token, pad_token, chat_template" },
      { "command": "curl -s -X POST http://localhost:8080/v1/tokenize -d '{\"text\":\"hello\"}'", "exitCode": 0, "observation": "Returned tokens array [non-empty]" },
      { "command": "curl -s -X POST http://localhost:8080/v1/detokenize -d '{\"tokens\":[1234]}'", "exitCode": 0, "observation": "Returned decoded text" }
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      { "file": "tests/e2e/harness.sh", "cases": [{ "name": "tokenizer_info", "verifies": "GET /tokenizer_info returns metadata" }, { "name": "tokenize_roundtrip", "verifies": "tokenize then detokenize preserves text" }] }
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- The feature requires an API endpoint or data model that doesn't exist yet and is not in the preconditions
- Requirements are ambiguous or contradictory (e.g., performance constraints conflict with functionality)
- Build fails due to missing dependencies that cannot be resolved
- The feature touches off-limits resources (see AGENTS.md)
- A pre-existing bug blocks implementation and is outside the feature scope
