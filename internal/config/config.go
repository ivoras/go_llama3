package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Config holds the model architecture information from config.json.
type Config struct {
	Architectures         []string `json:"architectures"`
	HiddenSize            int      `json:"hidden_size"`
	IntermediateSize      int      `json:"intermediate_size"`
	NumAttentionHeads     int      `json:"num_attention_heads"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	NumKeyValueHeads      int      `json:"num_key_value_heads"`
	VocabSize             int      `json:"vocab_size"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	RopeTheta             float64  `json:"rope_theta"`
	RMSNormEps            float64  `json:"rms_norm_eps"`
}

func (c Config) HeadDim() int {
	if c.NumAttentionHeads == 0 {
		return 0
	}
	return c.HiddenSize / c.NumAttentionHeads
}

func (c Config) GQARatio() int {
	if c.NumKeyValueHeads == 0 {
		return 0
	}
	return c.NumAttentionHeads / c.NumKeyValueHeads
}

func (c Config) Validate() error {
	if c.HiddenSize <= 0 {
		return fmt.Errorf("invalid hidden_size: %d", c.HiddenSize)
	}
	if c.IntermediateSize <= 0 {
		return fmt.Errorf("invalid intermediate_size: %d", c.IntermediateSize)
	}
	if c.NumAttentionHeads <= 0 {
		return fmt.Errorf("invalid num_attention_heads: %d", c.NumAttentionHeads)
	}
	if c.NumHiddenLayers <= 0 {
		return fmt.Errorf("invalid num_hidden_layers: %d", c.NumHiddenLayers)
	}
	if c.NumKeyValueHeads <= 0 {
		return fmt.Errorf("invalid num_key_value_heads: %d", c.NumKeyValueHeads)
	}
	if c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
		return fmt.Errorf("num_attention_heads must be divisible by num_key_value_heads")
	}
	if c.HiddenSize%c.NumAttentionHeads != 0 {
		return fmt.Errorf("hidden_size must be divisible by num_attention_heads")
	}
	if c.VocabSize <= 0 {
		return fmt.Errorf("invalid vocab_size: %d", c.VocabSize)
	}
	if c.MaxPositionEmbeddings <= 0 {
		return fmt.Errorf("invalid max_position_embeddings: %d", c.MaxPositionEmbeddings)
	}
	if c.RMSNormEps <= 0 {
		return fmt.Errorf("invalid rms_norm_eps: %f", c.RMSNormEps)
	}
	if c.RopeTheta == 0 {
		c.RopeTheta = 10000.0
	}
	return nil
}

func Load(path string) (Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("read config %q: %w", path, err)
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return Config{}, fmt.Errorf("unmarshal config %q: %w", path, err)
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000.0
	}
	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func LoadFromModelDir(modelDir string) (Config, error) {
	return Load(filepath.Join(modelDir, "config.json"))
}
