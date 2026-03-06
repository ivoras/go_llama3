package layers

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	mathops "github.com/ivora/go_llama3/internal/math"
)

type FFNWeights struct {
	GateW      []float32 // [intermediate, hidden]
	UpW        []float32 // [intermediate, hidden]
	DownW      []float32 // [hidden, intermediate]
	HiddenSize int
	InterSize  int
}

const ffnParallelThreshold = 4

func (f FFNWeights) Forward(x []float32, seqLen int) ([]float32, error) {
	if len(x) != seqLen*f.HiddenSize {
		return nil, fmt.Errorf("ffn input shape mismatch")
	}
	out := make([]float32, len(x))
	if seqLen >= ffnParallelThreshold {
		var wg sync.WaitGroup
		nWorkers := runtime.NumCPU()
		if nWorkers > seqLen {
			nWorkers = seqLen
		}
		chunk := (seqLen + nWorkers - 1) / nWorkers
		for w := 0; w < nWorkers; w++ {
			tStart := w * chunk
			tEnd := tStart + chunk
			if tEnd > seqLen {
				tEnd = seqLen
			}
			if tStart >= tEnd {
				continue
			}
			wg.Add(1)
			go func(ts, te int) {
				defer wg.Done()
				gate := make([]float32, f.InterSize)
				up := make([]float32, f.InterSize)
				prod := make([]float32, f.InterSize)
				for t := ts; t < te; t++ {
					token := x[t*f.HiddenSize : (t+1)*f.HiddenSize]
					_ = mathops.MatVec(gate, f.GateW, token, f.InterSize, f.HiddenSize)
					_ = mathops.MatVec(up, f.UpW, token, f.InterSize, f.HiddenSize)
					for i := range gate {
						sig := 1.0 / (1.0 + math.Exp(-float64(gate[i])))
						gate[i] = float32(float64(gate[i]) * sig)
						prod[i] = gate[i] * up[i]
					}
					target := out[t*f.HiddenSize : (t+1)*f.HiddenSize]
					_ = mathops.MatVec(target, f.DownW, prod, f.HiddenSize, f.InterSize)
				}
			}(tStart, tEnd)
		}
		wg.Wait()
		return out, nil
	}
	gate := make([]float32, f.InterSize)
	up := make([]float32, f.InterSize)
	prod := make([]float32, f.InterSize)
	for t := 0; t < seqLen; t++ {
		token := x[t*f.HiddenSize : (t+1)*f.HiddenSize]
		if err := mathops.MatVec(gate, f.GateW, token, f.InterSize, f.HiddenSize); err != nil {
			return nil, err
		}
		if err := mathops.MatVec(up, f.UpW, token, f.InterSize, f.HiddenSize); err != nil {
			return nil, err
		}
		for i := range gate {
			sig := 1.0 / (1.0 + math.Exp(-float64(gate[i])))
			gate[i] = float32(float64(gate[i]) * sig)
			prod[i] = gate[i] * up[i]
		}
		target := out[t*f.HiddenSize : (t+1)*f.HiddenSize]
		if err := mathops.MatVec(target, f.DownW, prod, f.HiddenSize, f.InterSize); err != nil {
			return nil, err
		}
	}
	return out, nil
}
