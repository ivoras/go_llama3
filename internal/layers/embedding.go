package layers

import "fmt"

type Embedding struct {
	Weights   []float32
	VocabSize int
	Hidden    int
}

func (e Embedding) Forward(tokens []int32) ([]float32, error) {
	out := make([]float32, len(tokens)*e.Hidden)
	for i, tok := range tokens {
		t := int(tok)
		if t < 0 || t >= e.VocabSize {
			return nil, fmt.Errorf("token id out of range: %d", t)
		}
		src := t * e.Hidden
		dst := i * e.Hidden
		copy(out[dst:dst+e.Hidden], e.Weights[src:src+e.Hidden])
	}
	return out, nil
}
