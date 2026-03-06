# go_llama3

High-performance Llama 3.2 text inference engine in pure Go. Loads safetensor weights, implements GQA, RoPE, RMSNorm, and uses go-highway SIMD for matrix/vector operations.

## Requirements

- Go 1.26+
- Llama 3.2 model files (1B or 3B, FP32/BF16)

## Model Setup

Place a Llama 3.2 model in a directory with:

- `config.json`
- `tokenizer.json`
- `model.safetensors.index.json` (or `model.safetensors` for single-file checkpoints)
- `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`, etc.

Download models from [Meta Llama](https://llama.meta.com/) or [Hugging Face](https://huggingface.co/meta-llama).

## Running Local Inference

### Build

```bash
# With SIMD acceleration (recommended on AMD64)
GOEXPERIMENT=simd go build -o go_llama3 ./cmd/go_llama3

# Without SIMD (scalar fallback)
go build -o go_llama3 ./cmd/go_llama3
```

### Run

```bash
./go_llama3 --model /path/to/model --prompt "Hello, how are you?"
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to model directory |
| `--prompt` | (required) | Text prompt to run |
| `--max-tokens` | 64 | Maximum tokens to generate |
| `--temperature` | 0.8 | Sampling temperature (lower = more deterministic) |
| `--top-k` | 40 | Top-k sampling |
| `--top-p` | 0.95 | Top-p (nucleus) sampling |

### Example

```bash
./go_llama3 --model ./Llama-3.2-3B-Instruct \
  --prompt "What is the capital of France?" \
  --max-tokens 128 \
  --temperature 0.7
```

### One-liner (no build)

```bash
GOEXPERIMENT=simd go run ./cmd/go_llama3 --model /path/to/model --prompt "Your prompt here"
```

## Integration Tests

JSON generation tests require a model. Set `GO_LLAMA3_MODEL_PATH` to skip when absent:

```bash
GO_LLAMA3_MODEL_PATH=/path/to/model go test ./internal/model -run JSON -v
```

## License

BSD 2-Clause License
