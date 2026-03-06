package model

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ivora/go_llama3/internal/config"
	"github.com/ivora/go_llama3/internal/layers"
	"github.com/ivora/go_llama3/internal/loader"
	mathops "github.com/ivora/go_llama3/internal/math"
	"github.com/ivora/go_llama3/internal/sampling"
	"github.com/ivora/go_llama3/internal/tokenizer"
)

type TransformerLayer struct {
	InputNormWeight []float32
	PostNormWeight  []float32
	Attention       layers.AttentionWeights
	FFN             layers.FFNWeights
}

type KVCache struct {
	Layers []layers.KVLayerCache
}

func NewKVCache(numLayers int) *KVCache {
	return &KVCache{
		Layers: make([]layers.KVLayerCache, numLayers),
	}
}

type Model struct {
	Config    config.Config
	Tokenizer *tokenizer.Tokenizer
	Rope      *layers.RoPE

	Embedding layers.Embedding
	Layers    []TransformerLayer
	FinalNorm []float32
	LMHead    []float32 // [vocab, hidden]
}

// LoadOptions configures model loading with optional logging and stats.
type LoadOptions struct {
	// Logf is called with progress messages during load. If nil, no logging.
	Logf func(format string, args ...any)
	// Stats receives phase breakdown. If nil, no stats are collected.
	Stats *LoadStats
}

// LoadStats holds per-phase timing and size metrics for model loading.
type LoadStats struct {
	Source      string  // "gguf" or "safetensors"
	ConfigMs    float64 // Config/metadata parse
	TokenizerMs float64 // Tokenizer load
	InitMs      float64 // GGUF open or safetensor index
	TensorsMs   float64 // Weight loading
	TotalMs     float64 // Total load time
	TotalBytes  int64   // Total bytes read
}

// Load loads a model from path. Path may be:
// - A directory containing config.json, tokenizer.json, and safetensor files
// - A single .gguf file (tokenizer.json must be in the same directory)
func Load(path string) (*Model, error) {
	return LoadWithOptions(path, nil)
}

// LoadWithOptions loads a model with optional logging and stats.
func LoadWithOptions(path string, opts *LoadOptions) (*Model, error) {
	logf := func(string, ...any) {}
	if opts != nil && opts.Logf != nil {
		logf = opts.Logf
	}
	var stats *LoadStats
	if opts != nil && opts.Stats != nil {
		stats = opts.Stats
	}
	start := time.Now()
	if strings.HasSuffix(strings.ToLower(path), ".gguf") {
		m, err := loadFromGGUF(path, logf, stats)
		if err != nil {
			return nil, err
		}
		if stats != nil {
			stats.TotalMs = time.Since(start).Seconds() * 1000
		}
		return m, nil
	}
	m, err := loadFromDir(path, logf, stats)
	if err != nil {
		return nil, err
	}
	if stats != nil {
		stats.TotalMs = time.Since(start).Seconds() * 1000
	}
	return m, nil
}

func loadFromGGUF(ggufPath string, logf func(string, ...any), stats *LoadStats) (*Model, error) {
	logf("Loading GGUF from %s", ggufPath)
	t0 := time.Now()
	gguf, err := loader.NewGGUF(ggufPath)
	if err != nil {
		return nil, err
	}
	t1 := time.Now()
	initMs := t1.Sub(t0).Seconds() * 1000
	if stats != nil {
		stats.Source = "gguf"
		stats.InitMs = initMs
	}
	logf("  GGUF opened in %.1f ms", initMs)

	cfg, err := loader.ConfigFromGGUF(gguf.Metadata())
	if err != nil {
		return nil, fmt.Errorf("gguf config: %w", err)
	}
	modelCfg := cfg.ToConfig()
	t2 := time.Now()
	if stats != nil {
		stats.ConfigMs = t2.Sub(t1).Seconds() * 1000
	}
	logf("  Config: %d layers, %d hidden, vocab %d", modelCfg.NumHiddenLayers, modelCfg.HiddenSize, modelCfg.VocabSize)

	tokPath := filepath.Join(gguf.TokenizerPath(), "tokenizer.json")
	tok, err := tokenizer.LoadFromGGUF(gguf.Metadata(), tokPath)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}
	t3 := time.Now()
	tokMs := t3.Sub(t2).Seconds() * 1000
	if stats != nil {
		stats.TokenizerMs = tokMs
	}
	logf("  Tokenizer loaded in %.1f ms", tokMs)

	m := &Model{
		Config:    modelCfg,
		Tokenizer: tok,
		Rope:      layers.NewRoPE(modelCfg.MaxPositionEmbeddings, modelCfg.HeadDim(), modelCfg.RopeTheta),
		Layers:    make([]TransformerLayer, modelCfg.NumHiddenLayers),
	}

	t4 := time.Now()
	logf("  Loading weights...")
	emb, err := gguf.LoadTensor("model.embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	m.Embedding = layers.Embedding{
		Weights:   emb.Data,
		VocabSize: modelCfg.VocabSize,
		Hidden:    modelCfg.HiddenSize,
	}

	finalNorm, err := gguf.LoadTensor("model.norm.weight")
	if err != nil {
		return nil, err
	}
	m.FinalNorm = finalNorm.Data

	lmHead, err := gguf.LoadTensor("lm_head.weight")
	if err != nil {
		m.LMHead = emb.Data
	} else {
		m.LMHead = lmHead.Data
	}

	for i := 0; i < modelCfg.NumHiddenLayers; i++ {
		layerPrefix := fmt.Sprintf("model.layers.%d", i)
		inNorm, err := gguf.LoadTensor(layerPrefix + ".input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		postNorm, err := gguf.LoadTensor(layerPrefix + ".post_attention_layernorm.weight")
		if err != nil {
			return nil, err
		}
		q, err := gguf.LoadTensor(layerPrefix + ".self_attn.q_proj.weight")
		if err != nil {
			return nil, err
		}
		k, err := gguf.LoadTensor(layerPrefix + ".self_attn.k_proj.weight")
		if err != nil {
			return nil, err
		}
		v, err := gguf.LoadTensor(layerPrefix + ".self_attn.v_proj.weight")
		if err != nil {
			return nil, err
		}
		o, err := gguf.LoadTensor(layerPrefix + ".self_attn.o_proj.weight")
		if err != nil {
			return nil, err
		}
		gate, err := gguf.LoadTensor(layerPrefix + ".mlp.gate_proj.weight")
		if err != nil {
			return nil, err
		}
		up, err := gguf.LoadTensor(layerPrefix + ".mlp.up_proj.weight")
		if err != nil {
			return nil, err
		}
		down, err := gguf.LoadTensor(layerPrefix + ".mlp.down_proj.weight")
		if err != nil {
			return nil, err
		}

		m.Layers[i] = TransformerLayer{
			InputNormWeight: inNorm.Data,
			PostNormWeight:  postNorm.Data,
			Attention: layers.NewAttentionWeights(
				modelCfg.HiddenSize,
				modelCfg.NumAttentionHeads,
				modelCfg.NumKeyValueHeads,
				q.Data, k.Data, v.Data, o.Data,
			),
			FFN: layers.FFNWeights{
				GateW:      gate.Data,
				UpW:        up.Data,
				DownW:      down.Data,
				HiddenSize: modelCfg.HiddenSize,
				InterSize:  modelCfg.IntermediateSize,
			},
		}
		if (i+1)%8 == 0 || i == modelCfg.NumHiddenLayers-1 {
			logf("  Loaded layers 0..%d/%d", i+1, modelCfg.NumHiddenLayers)
		}
	}

	tensorsMs := time.Since(t4).Seconds() * 1000
	if stats != nil {
		stats.TensorsMs = tensorsMs
	}
	logf("  Weights loaded in %.1f ms", tensorsMs)
	return m, nil
}

func loadFromDir(modelDir string, logf func(string, ...any), stats *LoadStats) (*Model, error) {
	logf("Loading model from directory %s", modelDir)
	t0 := time.Now()
	cfg, err := config.LoadFromModelDir(modelDir)
	if err != nil {
		return nil, err
	}
	t1 := time.Now()
	if stats != nil {
		stats.Source = "safetensors"
		stats.ConfigMs = t1.Sub(t0).Seconds() * 1000
	}
	logf("  Config: %d layers, %d hidden, vocab %d", cfg.NumHiddenLayers, cfg.HiddenSize, cfg.VocabSize)

	tok, err := tokenizer.LoadFromModelDir(modelDir)
	if err != nil {
		return nil, err
	}
	t2 := time.Now()
	tokMs := t2.Sub(t1).Seconds() * 1000
	if stats != nil {
		stats.TokenizerMs = tokMs
	}
	logf("  Tokenizer loaded in %.1f ms", tokMs)

	ldr, err := loader.New(modelDir)
	if err != nil {
		return nil, err
	}
	t3 := time.Now()
	initMs := t3.Sub(t2).Seconds() * 1000
	if stats != nil {
		stats.InitMs = initMs
	}
	logf("  Safetensor index loaded in %.1f ms", initMs)

	m := &Model{
		Config:    cfg,
		Tokenizer: tok,
		Rope:      layers.NewRoPE(cfg.MaxPositionEmbeddings, cfg.HeadDim(), cfg.RopeTheta),
		Layers:    make([]TransformerLayer, cfg.NumHiddenLayers),
	}

	t4 := time.Now()
	logf("  Loading weights...")
	emb, err := ldr.LoadTensor("model.embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	m.Embedding = layers.Embedding{
		Weights:   emb.Data,
		VocabSize: cfg.VocabSize,
		Hidden:    cfg.HiddenSize,
	}

	finalNorm, err := ldr.LoadTensor("model.norm.weight")
	if err != nil {
		return nil, err
	}
	m.FinalNorm = finalNorm.Data

	lmHead, err := ldr.LoadTensor("lm_head.weight")
	if err != nil {
		// Some checkpoints tie LM head to embedding.
		m.LMHead = emb.Data
	} else {
		m.LMHead = lmHead.Data
	}

	for i := 0; i < cfg.NumHiddenLayers; i++ {
		layerPrefix := fmt.Sprintf("model.layers.%d", i)

		inNorm, err := ldr.LoadTensor(layerPrefix + ".input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		postNorm, err := ldr.LoadTensor(layerPrefix + ".post_attention_layernorm.weight")
		if err != nil {
			return nil, err
		}

		q, err := ldr.LoadTensor(layerPrefix + ".self_attn.q_proj.weight")
		if err != nil {
			return nil, err
		}
		k, err := ldr.LoadTensor(layerPrefix + ".self_attn.k_proj.weight")
		if err != nil {
			return nil, err
		}
		v, err := ldr.LoadTensor(layerPrefix + ".self_attn.v_proj.weight")
		if err != nil {
			return nil, err
		}
		o, err := ldr.LoadTensor(layerPrefix + ".self_attn.o_proj.weight")
		if err != nil {
			return nil, err
		}

		gate, err := ldr.LoadTensor(layerPrefix + ".mlp.gate_proj.weight")
		if err != nil {
			return nil, err
		}
		up, err := ldr.LoadTensor(layerPrefix + ".mlp.up_proj.weight")
		if err != nil {
			return nil, err
		}
		down, err := ldr.LoadTensor(layerPrefix + ".mlp.down_proj.weight")
		if err != nil {
			return nil, err
		}

		m.Layers[i] = TransformerLayer{
			InputNormWeight: inNorm.Data,
			PostNormWeight:  postNorm.Data,
			Attention: layers.NewAttentionWeights(
				cfg.HiddenSize,
				cfg.NumAttentionHeads,
				cfg.NumKeyValueHeads,
				q.Data, k.Data, v.Data, o.Data,
			),
			FFN: layers.FFNWeights{
				GateW:      gate.Data,
				UpW:        up.Data,
				DownW:      down.Data,
				HiddenSize: cfg.HiddenSize,
				InterSize:  cfg.IntermediateSize,
			},
		}
		if (i+1)%8 == 0 || i == cfg.NumHiddenLayers-1 {
			logf("  Loaded layers 0..%d/%d", i+1, cfg.NumHiddenLayers)
		}
	}

	tensorsMs := time.Since(t4).Seconds() * 1000
	if stats != nil {
		stats.TensorsMs = tensorsMs
	}
	logf("  Weights loaded in %.1f ms", tensorsMs)
	return m, nil
}

// LoadFromDir loads a model from a directory (config.json, tokenizer.json, safetensors).
func LoadFromDir(modelDir string) (*Model, error) {
	return loadFromDir(modelDir, func(string, ...any) {}, nil)
}

func (m *Model) Forward(tokens []int32, startPos int, kvCache *KVCache) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("forward requires at least one token")
	}
	seqLen := len(tokens)
	x, err := m.Embedding.Forward(tokens)
	if err != nil {
		return nil, err
	}
	for i := 0; i < m.Config.NumHiddenLayers; i++ {
		ly := m.Layers[i]

		attnIn, err := layers.ApplyRMSNormSequence(x, seqLen, m.Config.HiddenSize, ly.InputNormWeight, float32(m.Config.RMSNormEps))
		if err != nil {
			return nil, err
		}
		var cache *layers.KVLayerCache
		if kvCache != nil && i < len(kvCache.Layers) {
			cache = &kvCache.Layers[i]
		}
		attnOut, err := ly.Attention.Forward(attnIn, seqLen, m.Rope, startPos, cache)
		if err != nil {
			return nil, err
		}
		mathops.AddInPlace(x, attnOut) // residual

		ffnIn, err := layers.ApplyRMSNormSequence(x, seqLen, m.Config.HiddenSize, ly.PostNormWeight, float32(m.Config.RMSNormEps))
		if err != nil {
			return nil, err
		}
		ffnOut, err := ly.FFN.Forward(ffnIn, seqLen)
		if err != nil {
			return nil, err
		}
		mathops.AddInPlace(x, ffnOut) // residual
	}

	finalNormed, err := layers.ApplyRMSNormSequence(x, seqLen, m.Config.HiddenSize, m.FinalNorm, float32(m.Config.RMSNormEps))
	if err != nil {
		return nil, err
	}

	// Last-token logits.
	last := finalNormed[(seqLen-1)*m.Config.HiddenSize : seqLen*m.Config.HiddenSize]
	logits := make([]float32, m.Config.VocabSize)
	if err := mathops.MatVec(logits, m.LMHead, last, m.Config.VocabSize, m.Config.HiddenSize); err != nil {
		return nil, err
	}
	return logits, nil
}

type GenerateOptions struct {
	MaxNewTokens int
	Temperature  float32
	TopK         int
	TopP         float32
	// Logf is called with progress messages during generation. If nil, no logging.
	Logf func(format string, args ...any)
}

// GenerateStats holds inference timing and throughput metrics.
type GenerateStats struct {
	LoadTimeMs       float64 // Model load duration in milliseconds
	EncodeMs         float64 // Prompt tokenization time
	PrefillMs        float64 // Prefill (forward on prompt tokens) time
	FirstTokenMs     float64 // Time to first generated token (prefill + first decode)
	TotalMs          float64 // Total generation time in milliseconds
	NumGenerated     int     // Number of tokens generated
	TokensPerSecond  float64 // Generated tokens per second
	PromptTokens     int     // Number of prompt tokens
}

func (m *Model) Generate(prompt string, opts GenerateOptions) (string, error) {
	out, err := m.GenerateWithStats(prompt, opts, nil)
	return out, err
}

// GenerateWithStats runs generation with incremental KV cache and populates stats if non-nil.
func (m *Model) GenerateWithStats(prompt string, opts GenerateOptions, stats *GenerateStats) (string, error) {
	if opts.MaxNewTokens <= 0 {
		opts.MaxNewTokens = 64
	}
	if opts.Temperature <= 0 {
		opts.Temperature = 1.0
	}
	logf := func(string, ...any) {}
	if opts.Logf != nil {
		logf = opts.Logf
	}

	logf("Processing prompt (%d chars)...", len(prompt))
	t0 := time.Now()
	toks := m.Tokenizer.Encode(prompt)
	encodeMs := time.Since(t0).Seconds() * 1000
	if len(toks) == 0 {
		return "", fmt.Errorf("prompt produced no tokens")
	}
	logf("  Encoded %d tokens in %.1f ms", len(toks), encodeMs)
	if stats != nil {
		stats.PromptTokens = len(toks)
		stats.EncodeMs = encodeMs
	}

	generated := make([]int32, 0, opts.MaxNewTokens)
	kvCache := NewKVCache(m.Config.NumHiddenLayers)

	startTotal := time.Now()
	var firstTokenAt time.Duration

	logf("  Prefilling...")
	t1 := time.Now()
	logits, err := m.Forward(toks, 0, kvCache)
	if err != nil {
		return "", err
	}
	prefillMs := time.Since(t1).Seconds() * 1000
	logf("  Prefill complete in %.1f ms", prefillMs)
	if stats != nil {
		stats.PrefillMs = prefillMs
	}

	next := sampling.Sample(logits, sampling.Options{
		Temperature: opts.Temperature,
		TopK:        opts.TopK,
		TopP:        opts.TopP,
	})
	firstTokenAt = time.Since(startTotal)
	generated = append(generated, next)

	// Decode: incremental forward with KV cache.
	ctx := append(append([]int32(nil), toks...), next)
	for i := 1; i < opts.MaxNewTokens; i++ {
		startPos := len(ctx) - 1
		logits, err = m.Forward(ctx[startPos:], startPos, kvCache)
		if err != nil {
			return "", err
		}
		next = sampling.Sample(logits, sampling.Options{
			Temperature: opts.Temperature,
			TopK:        opts.TopK,
			TopP:        opts.TopP,
		})
		ctx = append(ctx, next)
		generated = append(generated, next)
	}

	totalDuration := time.Since(startTotal)
	if stats != nil {
		stats.FirstTokenMs = firstTokenAt.Seconds() * 1000
		stats.TotalMs = totalDuration.Seconds() * 1000
		stats.NumGenerated = len(generated)
		stats.PromptTokens = len(toks)
		if stats.TotalMs > 0 {
			stats.TokensPerSecond = float64(stats.NumGenerated) / (stats.TotalMs / 1000)
		}
	}
	return m.Tokenizer.Decode(generated), nil
}

// HasModelFiles returns true if path points to a loadable model (directory with config+tokenizer, or .gguf file).
func HasModelFiles(path string) bool {
	if strings.HasSuffix(strings.ToLower(path), ".gguf") {
		if st, err := os.Stat(path); err != nil || !st.Mode().IsRegular() {
			return false
		}
		tokPath := filepath.Join(filepath.Dir(path), "tokenizer.json")
		if _, err := os.Stat(tokPath); err != nil {
			return false
		}
		return true
	}
	paths := []string{
		filepath.Join(path, "config.json"),
		filepath.Join(path, "tokenizer.json"),
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err != nil {
			return false
		}
	}
	return true
}
