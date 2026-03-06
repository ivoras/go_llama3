package layers

import "math"

type RoPE struct {
	HeadDim int
	Base    float64
	Cos     [][]float32
	Sin     [][]float32
}

func NewRoPE(maxPos, headDim int, base float64) *RoPE {
	if base == 0 {
		base = 10000.0
	}
	half := headDim / 2
	cos := make([][]float32, maxPos)
	sin := make([][]float32, maxPos)
	for pos := 0; pos < maxPos; pos++ {
		cos[pos] = make([]float32, half)
		sin[pos] = make([]float32, half)
		for i := 0; i < half; i++ {
			theta := float64(pos) / math.Pow(base, float64(2*i)/float64(headDim))
			cos[pos][i] = float32(math.Cos(theta))
			sin[pos][i] = float32(math.Sin(theta))
		}
	}
	return &RoPE{
		HeadDim: headDim,
		Base:    base,
		Cos:     cos,
		Sin:     sin,
	}
}

// Apply rotates one head vector in-place for a given position.
func (r *RoPE) Apply(v []float32, pos int) {
	if pos < 0 || pos >= len(r.Cos) {
		return
	}
	half := r.HeadDim / 2
	c := r.Cos[pos]
	s := r.Sin[pos]
	for i := 0; i < half; i++ {
		x1 := v[i]
		x2 := v[i+half]
		v[i] = x1*c[i] - x2*s[i]
		v[i+half] = x1*s[i] + x2*c[i]
	}
}
