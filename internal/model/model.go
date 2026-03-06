package model

import (
	"fmt"
	"os"
	"path/filepath"

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

func LoadFromDir(modelDir string) (*Model, error) {
	cfg, err := config.LoadFromModelDir(modelDir)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadFromModelDir(modelDir)
	if err != nil {
		return nil, err
	}
	ldr, err := loader.New(modelDir)
	if err != nil {
		return nil, err
	}

	m := &Model{
		Config:    cfg,
		Tokenizer: tok,
		Rope:      layers.NewRoPE(cfg.MaxPositionEmbeddings, cfg.HeadDim(), cfg.RopeTheta),
		Layers:    make([]TransformerLayer, cfg.NumHiddenLayers),
	}

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
	}

	return m, nil
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
}

func (m *Model) Generate(prompt string, opts GenerateOptions) (string, error) {
	if opts.MaxNewTokens <= 0 {
		opts.MaxNewTokens = 64
	}
	if opts.Temperature <= 0 {
		opts.Temperature = 1.0
	}
	toks := m.Tokenizer.Encode(prompt)
	if len(toks) == 0 {
		return "", fmt.Errorf("prompt produced no tokens")
	}
	generated := make([]int32, 0, opts.MaxNewTokens)
	ctx := append([]int32(nil), toks...)

	for i := 0; i < opts.MaxNewTokens; i++ {
		logits, err := m.Forward(ctx, 0, nil)
		if err != nil {
			return "", err
		}
		next := sampling.Sample(logits, sampling.Options{
			Temperature: opts.Temperature,
			TopK:        opts.TopK,
			TopP:        opts.TopP,
		})
		ctx = append(ctx, next)
		generated = append(generated, next)
	}
	return m.Tokenizer.Decode(generated), nil
}

func HasModelFiles(modelDir string) bool {
	paths := []string{
		filepath.Join(modelDir, "config.json"),
		filepath.Join(modelDir, "tokenizer.json"),
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err != nil {
			return false
		}
	}
	return true
}
