package layers

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	mathops "github.com/ivora/go_llama3/internal/math"
)

const attentionParallelThreshold = 4

type KVLayerCache struct {
	K []float32 // [cached_seq, numKVHeads, headDim]
	V []float32 // [cached_seq, numKVHeads, headDim]
}

func (c *KVLayerCache) CachedSeq(numKVHeads, headDim int) int {
	if c == nil || numKVHeads == 0 || headDim == 0 {
		return 0
	}
	return len(c.K) / (numKVHeads * headDim)
}

type AttentionWeights struct {
	Wq []float32 // [hidden, hidden]
	Wk []float32 // [kv_hidden, hidden]
	Wv []float32 // [kv_hidden, hidden]
	Wo []float32 // [hidden, hidden]

	HiddenSize  int
	NumHeads    int
	NumKVHeads  int
	HeadDim     int
	ScaleFactor float32
}

func (a AttentionWeights) Forward(x []float32, seqLen int, rope *RoPE, startPos int, cache *KVLayerCache) ([]float32, error) {
	if len(x) != seqLen*a.HiddenSize {
		return nil, fmt.Errorf("attention input shape mismatch")
	}

	kvHidden := a.NumKVHeads * a.HeadDim
	qAll := make([]float32, seqLen*a.HiddenSize)
	kAll := make([]float32, seqLen*kvHidden)
	vAll := make([]float32, seqLen*kvHidden)

	parallelTokens := seqLen >= attentionParallelThreshold
	if parallelTokens {
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
					token := x[t*a.HiddenSize : (t+1)*a.HiddenSize]
					_ = mathops.MatVec(qAll[t*a.HiddenSize:(t+1)*a.HiddenSize], a.Wq, token, a.HiddenSize, a.HiddenSize)
					_ = mathops.MatVec(kAll[t*kvHidden:(t+1)*kvHidden], a.Wk, token, kvHidden, a.HiddenSize)
					_ = mathops.MatVec(vAll[t*kvHidden:(t+1)*kvHidden], a.Wv, token, kvHidden, a.HiddenSize)
				}
			}(tStart, tEnd)
		}
		wg.Wait()
		for t := 0; t < seqLen; t++ {
			pos := startPos + t
			for h := 0; h < a.NumHeads; h++ {
				qHead := qAll[t*a.HiddenSize+h*a.HeadDim : t*a.HiddenSize+(h+1)*a.HeadDim]
				rope.Apply(qHead, pos)
			}
			for h := 0; h < a.NumKVHeads; h++ {
				kHead := kAll[t*kvHidden+h*a.HeadDim : t*kvHidden+(h+1)*a.HeadDim]
				rope.Apply(kHead, pos)
			}
		}
	} else {
		for t := 0; t < seqLen; t++ {
			token := x[t*a.HiddenSize : (t+1)*a.HiddenSize]
			if err := mathops.MatVec(qAll[t*a.HiddenSize:(t+1)*a.HiddenSize], a.Wq, token, a.HiddenSize, a.HiddenSize); err != nil {
				return nil, err
			}
			if err := mathops.MatVec(kAll[t*kvHidden:(t+1)*kvHidden], a.Wk, token, kvHidden, a.HiddenSize); err != nil {
				return nil, err
			}
			if err := mathops.MatVec(vAll[t*kvHidden:(t+1)*kvHidden], a.Wv, token, kvHidden, a.HiddenSize); err != nil {
				return nil, err
			}
		}
		for t := 0; t < seqLen; t++ {
			pos := startPos + t
			for h := 0; h < a.NumHeads; h++ {
				qHead := qAll[t*a.HiddenSize+h*a.HeadDim : t*a.HiddenSize+(h+1)*a.HeadDim]
				rope.Apply(qHead, pos)
			}
			for h := 0; h < a.NumKVHeads; h++ {
				kHead := kAll[t*kvHidden+h*a.HeadDim : t*kvHidden+(h+1)*a.HeadDim]
				rope.Apply(kHead, pos)
			}
		}
	}

	// Concatenate cached K/V with current token block.
	var fullK, fullV []float32
	var totalSeq int
	if cache != nil && len(cache.K) > 0 && len(cache.V) > 0 {
		totalSeq = cache.CachedSeq(a.NumKVHeads, a.HeadDim) + seqLen
		fullK = make([]float32, len(cache.K)+len(kAll))
		fullV = make([]float32, len(cache.V)+len(vAll))
		copy(fullK, cache.K)
		copy(fullV, cache.V)
		copy(fullK[len(cache.K):], kAll)
		copy(fullV[len(cache.V):], vAll)
	} else {
		totalSeq = seqLen
		fullK = kAll
		fullV = vAll
	}

	// Update cache for incremental decoding.
	if cache != nil {
		cache.K = append(cache.K[:0], fullK...)
		cache.V = append(cache.V[:0], fullV...)
	}

	groupSize := a.NumHeads / a.NumKVHeads
	attnOut := make([]float32, seqLen*a.HiddenSize)
	score := make([]float32, totalSeq)
	if parallelTokens {
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
				localScore := make([]float32, totalSeq)
				for t := ts; t < te; t++ {
					globalPos := startPos + t
					for qh := 0; qh < a.NumHeads; qh++ {
						kvh := qh / groupSize
						qVec := qAll[t*a.HiddenSize+qh*a.HeadDim : t*a.HiddenSize+(qh+1)*a.HeadDim]
						for s := 0; s < totalSeq; s++ {
							if s > globalPos {
								localScore[s] = float32(-1e9)
								continue
							}
							kBase := s*a.NumKVHeads*a.HeadDim + kvh*a.HeadDim
							kVec := fullK[kBase : kBase+a.HeadDim]
							localScore[s] = mathops.DotProduct(qVec, kVec) * a.ScaleFactor
						}
						mathops.SoftmaxInplace(localScore)
						outHead := attnOut[t*a.HiddenSize+qh*a.HeadDim : t*a.HiddenSize+(qh+1)*a.HeadDim]
						for i := range outHead {
							outHead[i] = 0
						}
						for s := 0; s < totalSeq; s++ {
							vBase := s*a.NumKVHeads*a.HeadDim + kvh*a.HeadDim
							vVec := fullV[vBase : vBase+a.HeadDim]
							w := localScore[s]
							for i := 0; i < a.HeadDim; i++ {
								outHead[i] += w * vVec[i]
							}
						}
					}
				}
			}(tStart, tEnd)
		}
		wg.Wait()
	} else {
		for t := 0; t < seqLen; t++ {
			globalPos := startPos + t
			for qh := 0; qh < a.NumHeads; qh++ {
				kvh := qh / groupSize
				qVec := qAll[t*a.HiddenSize+qh*a.HeadDim : t*a.HiddenSize+(qh+1)*a.HeadDim]
				for s := 0; s < totalSeq; s++ {
					if s > globalPos {
						score[s] = float32(-1e9)
						continue
					}
					kBase := s*a.NumKVHeads*a.HeadDim + kvh*a.HeadDim
					kVec := fullK[kBase : kBase+a.HeadDim]
					score[s] = mathops.DotProduct(qVec, kVec) * a.ScaleFactor
				}
				mathops.SoftmaxInplace(score)
				outHead := attnOut[t*a.HiddenSize+qh*a.HeadDim : t*a.HiddenSize+(qh+1)*a.HeadDim]
				for i := range outHead {
					outHead[i] = 0
				}
				for s := 0; s < totalSeq; s++ {
					vBase := s*a.NumKVHeads*a.HeadDim + kvh*a.HeadDim
					vVec := fullV[vBase : vBase+a.HeadDim]
					w := score[s]
					for i := 0; i < a.HeadDim; i++ {
						outHead[i] += w * vVec[i]
					}
				}
			}
		}
	}

	// Output projection Wo.
	projected := make([]float32, seqLen*a.HiddenSize)
	if parallelTokens {
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
					tokenOut := projected[t*a.HiddenSize : (t+1)*a.HiddenSize]
					tokenIn := attnOut[t*a.HiddenSize : (t+1)*a.HiddenSize]
					_ = mathops.MatVec(tokenOut, a.Wo, tokenIn, a.HiddenSize, a.HiddenSize)
				}
			}(tStart, tEnd)
		}
		wg.Wait()
	} else {
		for t := 0; t < seqLen; t++ {
			tokenOut := projected[t*a.HiddenSize : (t+1)*a.HiddenSize]
			tokenIn := attnOut[t*a.HiddenSize : (t+1)*a.HiddenSize]
			if err := mathops.MatVec(tokenOut, a.Wo, tokenIn, a.HiddenSize, a.HiddenSize); err != nil {
				return nil, err
			}
		}
	}
	return projected, nil
}

func NewAttentionWeights(hiddenSize, numHeads, numKVHeads int, wq, wk, wv, wo []float32) AttentionWeights {
	headDim := hiddenSize / numHeads
	return AttentionWeights{
		Wq:          wq,
		Wk:          wk,
		Wv:          wv,
		Wo:          wo,
		HiddenSize:  hiddenSize,
		NumHeads:    numHeads,
		NumKVHeads:  numKVHeads,
		HeadDim:     headDim,
		ScaleFactor: float32(1.0 / math.Sqrt(float64(headDim))),
	}
}
