# Benchmarks

Performance benchmarks for vmlx-engine on Apple Silicon.

## Benchmark Types

- [LLM Benchmarks](llm.md) - Text generation performance
- [Image Benchmarks](image.md) - Image understanding performance
- [Video Benchmarks](video.md) - Video understanding performance

## Quick Commands

```bash
# LLM benchmark
vmlx-engine-bench --model mlx-community/Qwen3-0.6B-8bit

# Image benchmark
vmlx-engine-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# Video benchmark
vmlx-engine-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --video
```

## Hardware

All benchmarks run on:
- **Chip**: Apple M4 Max
- **Memory**: 128 GB unified memory

Results will vary on different Apple Silicon chips.

## Contributing Benchmarks

If you have a different Apple Silicon chip, please share your results:

```bash
vmlx-engine-bench --model mlx-community/Qwen3-0.6B-8bit --output results.json
```

Open an issue with your results at [GitHub Issues](https://github.com/jjang-ai/vmlx/issues).
