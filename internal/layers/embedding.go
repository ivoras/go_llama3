package layers

import (
	"fmt"
	"runtime"
	"sync"
)

const embeddingParallelThreshold = 8

type Embedding struct {
	Weights   []float32
	VocabSize int
	Hidden    int
}

func (e Embedding) Forward(tokens []int32) ([]float32, error) {
	n := len(tokens)
	for i, tok := range tokens {
		if t := int(tok); t < 0 || t >= e.VocabSize {
			return nil, fmt.Errorf("token id out of range: %d", t)
		}
		_ = i
	}
	out := make([]float32, n*e.Hidden)
	if n >= embeddingParallelThreshold {
		var wg sync.WaitGroup
		nWorkers := runtime.NumCPU()
		if nWorkers > n {
			nWorkers = n
		}
		chunk := (n + nWorkers - 1) / nWorkers
		for w := 0; w < nWorkers; w++ {
			iStart := w * chunk
			iEnd := iStart + chunk
			if iEnd > n {
				iEnd = n
			}
			if iStart >= iEnd {
				continue
			}
			wg.Add(1)
			go func(is, ie int) {
				defer wg.Done()
				for i := is; i < ie; i++ {
					t := int(tokens[i])
					src := t * e.Hidden
					dst := i * e.Hidden
					copy(out[dst:dst+e.Hidden], e.Weights[src:src+e.Hidden])
				}
			}(iStart, iEnd)
		}
		wg.Wait()
		return out, nil
	}
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
