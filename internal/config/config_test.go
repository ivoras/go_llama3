package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	payload := `{
		"hidden_size": 16,
		"intermediate_size": 32,
		"num_attention_heads": 4,
		"num_hidden_layers": 2,
		"num_key_value_heads": 2,
		"vocab_size": 128,
		"max_position_embeddings": 1024,
		"rope_theta": 10000.0,
		"rms_norm_eps": 0.00001
	}`
	if err := os.WriteFile(path, []byte(payload), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if cfg.HeadDim() != 4 {
		t.Fatalf("HeadDim got %d", cfg.HeadDim())
	}
	if cfg.GQARatio() != 2 {
		t.Fatalf("GQARatio got %d", cfg.GQARatio())
	}
}
