# go_llama3 – Agent Context

## Project Overview

**go_llama3** is a high-performance Llama 3.2 text inference engine written in pure Go. It runs CPU-based autoregressive generation with no external runtime dependencies. Supports loading from safetensor directories or single GGUF files.

## Architecture

### Packages

| Path | Purpose |
|-----|---------|
| `cmd/go_llama3/` | CLI entry point; loads model, runs generation, prints stats |
| `internal/config/` | Parses `config.json` (hidden size, layers, vocab, RoPE, etc.) |
| `internal/loader/` | Safetensor shard loading; GGUF file loading (F32/F16/BF16) |
| `internal/tokenizer/` | BPE tokenizer from `tokenizer.json` or GGUF metadata |
| `internal/math/` | SIMD ops via go-highway (DotProduct, RMSNorm, Softmax, MatVec, MatMul) |
| `internal/layers/` | Embedding, RMSNorm, RoPE, GQA attention, SwiGLU FFN |
| `internal/model/` | Model load, forward pass, KV cache, generation with sampling |
| `internal/sampling/` | Temperature, top-k, top-p sampling |

### Model flow

1. **Load**: `model.Load` or `LoadWithOptions` → GGUF or safetensor directory
2. **Generate**: `GenerateWithStats` → encode prompt → prefill (forward on prompt) → decode (incremental with KV cache) → sampling
3. **Forward**: Embedding → N × (RMSNorm → Attention → residual → RMSNorm → FFN → residual) → final norm → LM head

### Performance

- **Parallelism**: Goroutines over tokens in Embedding, RMSNorm, Attention, FFN; over rows in MatVec/MatMul
- **SIMD**: go-highway for vectorized math; `GOEXPERIMENT=simd` recommended on AMD64
- **KV cache**: Incremental decoding; prefill once, then decode per token

## Tech Stack

- **Go 1.26+**
- **Dependencies**: `github.com/ajroetker/go-highway`, `golang.org/x/sys/cpu`, `github.com/nlpodyssey/safetensors`

## Model Formats

- **Safetensors**: Directory with `config.json`, `tokenizer.json`, `model.safetensors.index.json` + shards
- **GGUF**: Single `.gguf` file (F32/F16); tokenizer from GGUF metadata or `tokenizer.json` in same dir

## CLI

```bash
go build -o go_llama3 ./cmd/go_llama3
./go_llama3 --model <path> --prompt <text> [--max-tokens 64] [--verbose] [--stats true]
```

Flags: `--model`, `--prompt`, `--max-tokens`, `--temperature`, `--top-k`, `--top-p`, `--stats`, `--verbose`

## Agent Guidance

- **After each feature**: Build the CLI with `go build ./cmd/go_llama3` or `go build -o go_llama3 ./cmd/go_llama3`
- **Tests**: Run `go test ./...`; JSON generation tests require `GO_LLAMA3_MODEL_PATH` when a model is present
- **Conventions**: Keep internal packages under `internal/`; use `float32` for weights and activations; log via optional `Logf` callbacks
