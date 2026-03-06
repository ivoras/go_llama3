package mathops

import "testing"

func TestDotProduct(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	got := DotProduct(a, b)
	if got != 70 {
		t.Fatalf("DotProduct got %v want 70", got)
	}
}

func TestRMSNorm(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 1, 1}
	dst := make([]float32, 4)
	if err := RMSNorm(dst, x, weight, 1e-6); err != nil {
		t.Fatal(err)
	}
	// mean(x^2) = (1+4+9+16)/4 = 7.5, scale = 1/sqrt(7.5+1e-6)
	sum := dst[0] + dst[1] + dst[2] + dst[3]
	if sum < 1e-6 {
		t.Fatalf("RMSNorm produced zeros")
	}
}

func TestSoftmaxInplace(t *testing.T) {
	x := []float32{1, 1, 1}
	SoftmaxInplace(x)
	sum := x[0] + x[1] + x[2]
	if sum < 0.999 || sum > 1.001 {
		t.Fatalf("softmax sum = %v", sum)
	}
}

func BenchmarkDotProduct(b *testing.B) {
	const n = 1024
	a := make([]float32, n)
	bb := make([]float32, n)
	for i := range a {
		a[i] = float32(i) * 0.001
		bb[i] = float32(i) * 0.002
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DotProduct(a, bb)
	}
}
