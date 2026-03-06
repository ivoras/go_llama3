package loader

import (
	"testing"
)

func TestGGUFNameMapping(t *testing.T) {
	tests := []struct {
		hfName   string
		ggufName string
	}{
		{"model.embed_tokens.weight", "token_embd.weight"},
		{"lm_head.weight", "output.weight"},
		{"model.norm.weight", "norm.weight"},
		{"model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight"},
		{"model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"},
		{"model.layers.5.mlp.down_proj.weight", "blk.5.ffn_down.weight"},
	}
	for _, tt := range tests {
		got := ggufNameForHF(tt.hfName)
		if got != tt.ggufName {
			t.Errorf("ggufNameForHF(%q) = %q, want %q", tt.hfName, got, tt.ggufName)
		}
	}
}
