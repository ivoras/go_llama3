package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/ivora/go_llama3/internal/model"
)

func main() {
	var (
		modelPath   string
		prompt      string
		maxTokens   int
		temperature float64
		topK        int
		topP        float64
		showStats   bool
		verbose     bool
	)

	flag.StringVar(&modelPath, "model", "", "Path to model (directory with safetensors, or single .gguf file)")
	flag.StringVar(&prompt, "prompt", "", "Prompt to run")
	flag.IntVar(&maxTokens, "max-tokens", 64, "Maximum generated tokens")
	flag.Float64Var(&temperature, "temperature", 0.8, "Sampling temperature")
	flag.IntVar(&topK, "top-k", 40, "Top-k sampling")
	flag.Float64Var(&topP, "top-p", 0.95, "Top-p (nucleus) sampling")
	flag.BoolVar(&showStats, "stats", true, "Print inference statistics")
	flag.BoolVar(&verbose, "verbose", false, "Verbose logging during model load")
	flag.Parse()

	if modelPath == "" || prompt == "" {
		fmt.Fprintln(os.Stderr, "Usage: go_llama3 --model <path> --prompt <text> [options]")
		fmt.Fprintln(os.Stderr, "  path: model directory (safetensors) or single .gguf file")
		fmt.Fprintln(os.Stderr, "  options: --max-tokens, --temperature, --top-k, --top-p, --stats, --verbose")
		os.Exit(2)
	}

	var loadStats model.LoadStats
	opts := &model.LoadOptions{Stats: &loadStats}
	if verbose {
		opts.Logf = func(format string, args ...any) {
			fmt.Fprintf(os.Stderr, "[load] "+format+"\n", args...)
		}
	}
	m, err := model.LoadWithOptions(modelPath, opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n", err)
		os.Exit(1)
	}

	var stats model.GenerateStats
	stats.LoadTimeMs = loadStats.TotalMs

	genOpts := model.GenerateOptions{
		MaxNewTokens: maxTokens,
		Temperature:  float32(temperature),
		TopK:         topK,
		TopP:         float32(topP),
	}
	if verbose {
		genOpts.Logf = func(format string, args ...any) {
			fmt.Fprintf(os.Stderr, "[prompt] "+format+"\n", args...)
		}
	}
	out, err := m.GenerateWithStats(prompt, genOpts, &stats)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(out)

	if showStats {
		fmt.Fprintln(os.Stderr)
		fmt.Fprintf(os.Stderr, "--- Load stats ---\n")
		fmt.Fprintf(os.Stderr, "  Source:          %s\n", loadStats.Source)
		fmt.Fprintf(os.Stderr, "  Config:          %.1f ms\n", loadStats.ConfigMs)
		fmt.Fprintf(os.Stderr, "  Tokenizer:       %.1f ms\n", loadStats.TokenizerMs)
		fmt.Fprintf(os.Stderr, "  Init:            %.1f ms\n", loadStats.InitMs)
		fmt.Fprintf(os.Stderr, "  Tensors:         %.1f ms\n", loadStats.TensorsMs)
		fmt.Fprintf(os.Stderr, "  Total load:      %.1f ms\n", loadStats.TotalMs)
		fmt.Fprintln(os.Stderr)
		fmt.Fprintf(os.Stderr, "--- Inference stats ---\n")
		fmt.Fprintf(os.Stderr, "  Prompt encode:   %.1f ms\n", stats.EncodeMs)
		fmt.Fprintf(os.Stderr, "  Prefill:          %.1f ms\n", stats.PrefillMs)
		fmt.Fprintf(os.Stderr, "  First token:      %.1f ms\n", stats.FirstTokenMs)
		fmt.Fprintf(os.Stderr, "  Total time:       %.1f ms\n", stats.TotalMs)
		fmt.Fprintf(os.Stderr, "  Prompt tokens:    %d\n", stats.PromptTokens)
		fmt.Fprintf(os.Stderr, "  Generated:        %d tokens\n", stats.NumGenerated)
		fmt.Fprintf(os.Stderr, "  Throughput:       %.2f tok/s\n", stats.TokensPerSecond)
	}
}
