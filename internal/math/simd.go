package mathops

import (
	"fmt"
	"math"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/algo"
	"golang.org/x/sys/cpu"
)

// CPUFeatures captures runtime SIMD capability detection.
type CPUFeatures struct {
	AVX2 bool
	FMA  bool
	NEON bool
}

func DetectCPU() CPUFeatures {
	return CPUFeatures{
		AVX2: cpu.X86.HasAVX2,
		FMA:  cpu.X86.HasFMA,
		NEON: cpu.ARM64.HasASIMD,
	}
}

// DotProduct computes the dot product of a and b using SIMD (go-highway).
func DotProduct(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	lanes := hwy.NumLanes[float32]()
	sum := float32(0)
	i := 0
	for i+lanes <= n {
		va := hwy.Load(a[i:])
		vb := hwy.Load(b[i:])
		sum += hwy.ReduceSum(hwy.Mul(va, vb))
		i += lanes
	}
	for i < n {
		sum += a[i] * b[i]
		i++
	}
	return sum
}

// RMSNorm applies RMS normalization: dst = x * weight / sqrt(mean(x^2) + eps).
func RMSNorm(dst, x, weight []float32, eps float32) error {
	if len(dst) != len(x) || len(x) != len(weight) {
		return fmt.Errorf("rmsnorm length mismatch dst=%d x=%d weight=%d", len(dst), len(x), len(weight))
	}
	n := len(x)
	if n == 0 {
		return nil
	}
	lanes := hwy.NumLanes[float32]()

	// Sum of squares using SIMD.
	ss := float32(0)
	i := 0
	for i+lanes <= n {
		v := hwy.Load(x[i:])
		sq := hwy.Mul(v, v)
		ss += hwy.ReduceSum(sq)
		i += lanes
	}
	for i < n {
		ss += x[i] * x[i]
		i++
	}

	scale := float32(1.0 / math.Sqrt(float64(ss/float32(n)+eps)))
	scaleVec := hwy.Set[float32](scale)

	// dst = x * scale * weight
	i = 0
	for i+lanes <= n {
		vx := hwy.Load(x[i:])
		vw := hwy.Load(weight[i:])
		v := hwy.Mul(hwy.Mul(vx, scaleVec), vw)
		hwy.Store(v, dst[i:])
		i += lanes
	}
	for i < n {
		dst[i] = x[i] * scale * weight[i]
		i++
	}
	return nil
}

// SoftmaxInplace applies softmax in-place using SIMD (go-highway).
func SoftmaxInplace(x []float32) {
	if len(x) == 0 {
		return
	}
	lanes := hwy.NumLanes[float32]()

	// Find max using SIMD.
	maxVal := x[0]
	i := 0
	for i+lanes <= len(x) {
		v := hwy.Load(x[i:])
		m := hwy.ReduceMax(v)
		if m > maxVal {
			maxVal = m
		}
		i += lanes
	}
	for i < len(x) {
		if x[i] > maxVal {
			maxVal = x[i]
		}
		i++
	}

	// x = exp(x - max)
	maxVec := hwy.Set[float32](maxVal)
	i = 0
	for i+lanes <= len(x) {
		v := hwy.Load(x[i:])
		sub := hwy.Sub(v, maxVec)
		hwy.Store(sub, x[i:])
		i += lanes
	}
	for i < len(x) {
		x[i] -= maxVal
		i++
	}
	algo.ExpTransformFloat32(x, x)

	// Sum
	sum := float32(0)
	i = 0
	for i+lanes <= len(x) {
		v := hwy.Load(x[i:])
		sum += hwy.ReduceSum(v)
		i += lanes
	}
	for i < len(x) {
		sum += x[i]
		i++
	}
	if sum == 0 {
		return
	}
	inv := 1 / sum
	invVec := hwy.Set[float32](inv)

	// Scale
	i = 0
	for i+lanes <= len(x) {
		v := hwy.Load(x[i:])
		hwy.Store(hwy.Mul(v, invVec), x[i:])
		i += lanes
	}
	for i < len(x) {
		x[i] *= inv
		i++
	}
}

// MatMul computes C = A x B where:
// A is (m x n), B is (n x p), C is (m x p), all row-major.
func MatMul(a, b []float32, m, n, p int) ([]float32, error) {
	if len(a) != m*n {
		return nil, fmt.Errorf("matmul invalid A size got=%d want=%d", len(a), m*n)
	}
	if len(b) != n*p {
		return nil, fmt.Errorf("matmul invalid B size got=%d want=%d", len(b), n*p)
	}
	c := make([]float32, m*p)
	lanes := hwy.NumLanes[float32]()
	const block = 64
	for ii := 0; ii < m; ii += block {
		iMax := min(ii+block, m)
		for kk := 0; kk < n; kk += block {
			kMax := min(kk+block, n)
			for jj := 0; jj < p; jj += block {
				jMax := min(jj+block, p)
				for i := ii; i < iMax; i++ {
					aRow := i * n
					cRow := i * p
					for k := kk; k < kMax; k++ {
						aik := a[aRow+k]
						bRow := k * p
						aikVec := hwy.Set[float32](aik)
						j := jj
						for j+lanes <= jMax {
							cv := hwy.Load(c[cRow+j:])
							bv := hwy.Load(b[bRow+j:])
							hwy.Store(hwy.FMA(aikVec, bv, cv), c[cRow+j:])
							j += lanes
						}
						for j < jMax {
							c[cRow+j] += aik * b[bRow+j]
							j++
						}
					}
				}
			}
		}
	}
	return c, nil
}

func MatVec(out, mat, vec []float32, rows, cols int) error {
	if len(mat) != rows*cols {
		return fmt.Errorf("matvec invalid matrix size")
	}
	if len(vec) != cols {
		return fmt.Errorf("matvec invalid vector size")
	}
	if len(out) != rows {
		return fmt.Errorf("matvec invalid output size")
	}
	for r := 0; r < rows; r++ {
		out[r] = DotProduct(mat[r*cols:(r+1)*cols], vec)
	}
	return nil
}

// AddInPlace adds src to dst element-wise using SIMD.
func AddInPlace(dst, src []float32) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	if n == 0 {
		return
	}
	lanes := hwy.NumLanes[float32]()
	i := 0
	for i+lanes <= n {
		dv := hwy.Load(dst[i:])
		sv := hwy.Load(src[i:])
		hwy.Store(hwy.Add(dv, sv), dst[i:])
		i += lanes
	}
	for i < n {
		dst[i] += src[i]
		i++
	}
}

// MulInPlace multiplies dst by src element-wise using SIMD.
func MulInPlace(dst, src []float32) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	if n == 0 {
		return
	}
	lanes := hwy.NumLanes[float32]()
	i := 0
	for i+lanes <= n {
		dv := hwy.Load(dst[i:])
		sv := hwy.Load(src[i:])
		hwy.Store(hwy.Mul(dv, sv), dst[i:])
		i += lanes
	}
	for i < n {
		dst[i] *= src[i]
		i++
	}
}

// ScaleInPlace multiplies dst by factor using SIMD.
func ScaleInPlace(dst []float32, factor float32) {
	if len(dst) == 0 {
		return
	}
	lanes := hwy.NumLanes[float32]()
	fv := hwy.Set[float32](factor)
	i := 0
	for i+lanes <= len(dst) {
		v := hwy.Load(dst[i:])
		hwy.Store(hwy.Mul(v, fv), dst[i:])
		i += lanes
	}
	for i < len(dst) {
		dst[i] *= factor
		i++
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
