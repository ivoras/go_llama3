package sampling

import (
	"math/rand"
	"sort"

	mathops "github.com/ivora/go_llama3/internal/math"
)

type Options struct {
	Temperature float32
	TopK        int
	TopP        float32
}

type idxProb struct {
	Idx  int
	Prob float32
}

func Sample(logits []float32, opts Options) int32 {
	if len(logits) == 0 {
		return 0
	}
	temp := opts.Temperature
	if temp <= 0 {
		temp = 1.0
	}

	probs := make([]float32, len(logits))
	copy(probs, logits)
	invTemp := float32(1.0) / temp
	for i := range probs {
		probs[i] *= invTemp
	}
	mathops.SoftmaxInplace(probs)

	cands := make([]idxProb, 0, len(probs))
	for i, p := range probs {
		cands = append(cands, idxProb{Idx: i, Prob: p})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].Prob > cands[j].Prob })

	if opts.TopK > 0 && opts.TopK < len(cands) {
		cands = cands[:opts.TopK]
	}
	if opts.TopP > 0 && opts.TopP < 1.0 {
		sum := float32(0)
		cut := len(cands)
		for i, c := range cands {
			sum += c.Prob
			if sum >= opts.TopP {
				cut = i + 1
				break
			}
		}
		cands = cands[:cut]
	}

	// Renormalize and sample.
	total := float32(0)
	for _, c := range cands {
		total += c.Prob
	}
	if total == 0 {
		return int32(cands[0].Idx)
	}
	r := rand.Float32() * total
	acc := float32(0)
	for _, c := range cands {
		acc += c.Prob
		if r <= acc {
			return int32(c.Idx)
		}
	}
	return int32(cands[len(cands)-1].Idx)
}
