# lm-evaluation-harness Integration Notes

## Compatibility Status

vMLX is fully compatible with `EleutherAI/lm-evaluation-harness` v0.4.11 via the `local-completions` backend with `tokenizer_backend=remote`.

## Tested Configurations

- **generate_until tasks**: gsm8k (exit 0, valid metrics)
- **loglikelihood tasks**: hellaswag, record (exit 0, valid metrics)
- **batch_size=1**: Required for correct loglikelihood scoring. All tested tasks produce valid results.
- **tokenizer_backend=remote**: Successfully calls `/tokenizer_info`, `/tokenize`, `/detokenize`

## Required Server Configuration

```bash
# Start vMLX server
vmlxctl serve --model <model-path> --port 8080

# Run lm-eval (loglikelihood tasks)
lm-eval --model local-completions \
  --model_args "base_url=http://localhost:8080/v1/completions,model=<model-name>,tokenizer_backend=remote" \
  --tasks hellaswag --limit 5 --batch_size 1

# Run lm-eval (generate_until tasks)
lm-eval --model local-completions \
  --model_args "base_url=http://localhost:8080/v1/completions,model=<model-name>,tokenizer_backend=remote" \
  --tasks gsm8k --limit 5 --batch_size 1
```

## Known Limitations

### 1. Batch Size > 1 Incorrect for Loglikelihood

The `/v1/completions` handler joins multiple prompts with `\n` for batching. This is incorrect for independent loglikelihood scoring because each prompt should be evaluated independently. **Always use `batch_size=1`** for loglikelihood tasks.

### 2. Legacy Completions Chat Template Bypass

For loglikelihood fast-path requests (`max_tokens=1` + `echo=true` + `logprobs`), the server uses `rawPrompt` mode which tokenizes the prompt text directly without applying the chat template. This ensures correct token counts for lm-eval's `parse_logprobs()` slicing.

For generate_until tasks (non-fast-path streaming), the prompt still goes through the chat template in the streaming pipeline. This changes the model's context but does not break functionality — the model still generates text, just with chat formatting applied.

### 3. Tokenizer Endpoint Paths

lm-eval's `RemoteTokenizer` and `check_remote_tokenizer_support()` expect tokenizer endpoints at the server root (without `/v1/` prefix). vMLX registers both paths:
- `/v1/tokenizer_info` and `/tokenizer_info`
- `/v1/tokenize` and `/tokenize`
- `/v1/detokenize` and `/detokenize`

### 4. Tokenize Request Key Compatibility

lm-eval sends `{"prompt": "text"}` to `/tokenize` (not `{"text": "text"}`). vMLX accepts both `text` and `prompt` keys.

### 5. Detokenize Response Key Compatibility

lm-eval reads the `prompt` key from the `/detokenize` response (not `text`). vMLX returns both keys: `{"text": "...", "prompt": "..."}`.

### 6. Prompt as Token ID Array

lm-eval sends `"prompt": [int, int, ...]` (array of token IDs) for loglikelihood tasks when `tokenized_requests=true` (default). vMLX detokenizes these back to text and uses `rawPrompt` mode for the fast-path.

### 7. pad_token Accuracy

The Tokenizer protocol does not expose `padToken`. The `/tokenizer_info` endpoint returns `unknownToken` as `pad_token`. For models where `pad_token ≠ unk_token`, this will be incorrect.
