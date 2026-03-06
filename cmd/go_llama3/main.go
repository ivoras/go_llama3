package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/ivora/go_llama3/internal/model"
)

func main() {
	var (
		modelDir    string
		prompt      string
		maxTokens   int
		temperature float64
		topK        int
		topP        float64
	)

	flag.StringVar(&modelDir, "model", "", "Path to model directory")
	flag.StringVar(&prompt, "prompt", "", "Prompt to run")
	flag.IntVar(&maxTokens, "max-tokens", 64, "Maximum generated tokens")
	flag.Float64Var(&temperature, "temperature", 0.8, "Sampling temperature")
	flag.IntVar(&topK, "top-k", 40, "Top-k sampling")
	flag.Float64Var(&topP, "top-p", 0.95, "Top-p (nucleus) sampling")
	flag.Parse()

	if modelDir == "" || prompt == "" {
		fmt.Fprintln(os.Stderr, "Usage: go_llama3 --model <model_dir> --prompt <text> [--max-tokens 64 --temperature 0.8 --top-k 40 --top-p 0.95]")
		os.Exit(2)
	}

	m, err := model.LoadFromDir(modelDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n", err)
		os.Exit(1)
	}

	out, err := m.Generate(prompt, model.GenerateOptions{
		MaxNewTokens: maxTokens,
		Temperature:  float32(temperature),
		TopK:         topK,
		TopP:         float32(topP),
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(out)
}
