package layers

import (
	"runtime"
	"sync"

	mathops "github.com/ivora/go_llama3/internal/math"
)

const rmsNormParallelThreshold = 4

func ApplyRMSNormSequence(x []float32, seqLen, hidden int, weight []float32, eps float32) ([]float32, error) {
	out := make([]float32, len(x))
	if seqLen >= rmsNormParallelThreshold {
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
				for t := ts; t < te; t++ {
					src := x[t*hidden : (t+1)*hidden]
					dst := out[t*hidden : (t+1)*hidden]
					_ = mathops.RMSNorm(dst, src, weight, eps)
				}
			}(tStart, tEnd)
		}
		wg.Wait()
		return out, nil
	}
	for t := 0; t < seqLen; t++ {
		src := x[t*hidden : (t+1)*hidden]
		dst := out[t*hidden : (t+1)*hidden]
		if err := mathops.RMSNorm(dst, src, weight, eps); err != nil {
			return nil, err
		}
	}
	return out, nil
}
