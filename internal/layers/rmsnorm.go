package layers

import (
	mathops "github.com/ivora/go_llama3/internal/math"
)

func ApplyRMSNormSequence(x []float32, seqLen, hidden int, weight []float32, eps float32) ([]float32, error) {
	out := make([]float32, len(x))
	for t := 0; t < seqLen; t++ {
		src := x[t*hidden : (t+1)*hidden]
		dst := out[t*hidden : (t+1)*hidden]
		if err := mathops.RMSNorm(dst, src, weight, eps); err != nil {
			return nil, err
		}
	}
	return out, nil
}
