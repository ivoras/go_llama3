package loader

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/nlpodyssey/safetensors"
)

const (
	indexFileName = "model.safetensors.index.json"
)

// WeightLoader loads model weights. Implemented by Loader (safetensors) and GGUFLoader.
type WeightLoader interface {
	LoadTensor(name string) (Tensor, error)
	TensorNames() []string
}

// NewFromPath creates a WeightLoader from a path. If path is a .gguf file, loads GGUF.
// Otherwise treats path as a model directory with safetensors.
func NewFromPath(path string) (WeightLoader, error) {
	if strings.HasSuffix(strings.ToLower(path), ".gguf") {
		return NewGGUF(path)
	}
	return New(path)
}

type IndexFile struct {
	Metadata  map[string]string `json:"metadata"`
	WeightMap map[string]string `json:"weight_map"`
}

type Tensor struct {
	Name  string
	Shape []int
	Data  []float32
}

type Loader struct {
	modelDir string
	index    IndexFile

	mu     sync.Mutex
	shards map[string]safetensors.SafeTensors
}

func New(modelDir string) (*Loader, error) {
	ldr := &Loader{
		modelDir: modelDir,
		shards:   make(map[string]safetensors.SafeTensors),
	}
	if err := ldr.loadIndex(); err != nil {
		return nil, err
	}
	return ldr, nil
}

func (l *Loader) loadIndex() error {
	indexPath := filepath.Join(l.modelDir, indexFileName)
	raw, err := os.ReadFile(indexPath)
	if err != nil {
		// Single-file models may not have an index; support that case.
		if os.IsNotExist(err) {
			l.index = IndexFile{WeightMap: map[string]string{}}
			return nil
		}
		return fmt.Errorf("read index %q: %w", indexPath, err)
	}
	var idx IndexFile
	if err := json.Unmarshal(raw, &idx); err != nil {
		return fmt.Errorf("unmarshal index %q: %w", indexPath, err)
	}
	if idx.WeightMap == nil {
		idx.WeightMap = map[string]string{}
	}
	l.index = idx
	return nil
}

func (l *Loader) TensorNames() []string {
	names := make([]string, 0, len(l.index.WeightMap))
	for n := range l.index.WeightMap {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

func (l *Loader) LoadTensor(name string) (Tensor, error) {
	shardName := l.index.WeightMap[name]
	if shardName == "" {
		// Fallback for unsharded checkpoints.
		shardName = "model.safetensors"
	}
	st, err := l.loadShard(shardName)
	if err != nil {
		return Tensor{}, err
	}
	tv, ok := st.Tensor(name)
	if !ok {
		return Tensor{}, fmt.Errorf("tensor %q not found in shard %q", name, shardName)
	}

	shape := make([]int, 0, len(tv.Shape()))
	for _, dim := range tv.Shape() {
		shape = append(shape, int(dim))
	}
	data, err := toFloat32(tv.DType(), tv.Data())
	if err != nil {
		return Tensor{}, fmt.Errorf("tensor %q dtype conversion: %w", name, err)
	}
	return Tensor{
		Name:  name,
		Shape: shape,
		Data:  data,
	}, nil
}

func (l *Loader) loadShard(shardName string) (safetensors.SafeTensors, error) {
	l.mu.Lock()
	if st, ok := l.shards[shardName]; ok {
		l.mu.Unlock()
		return st, nil
	}
	l.mu.Unlock()

	path := filepath.Join(l.modelDir, shardName)
	raw, err := os.ReadFile(path)
	if err != nil {
		return safetensors.SafeTensors{}, fmt.Errorf("read shard %q: %w", path, err)
	}
	st, err := safetensors.Deserialize(raw)
	if err != nil {
		return safetensors.SafeTensors{}, fmt.Errorf("deserialize shard %q: %w", path, err)
	}

	l.mu.Lock()
	l.shards[shardName] = st
	l.mu.Unlock()

	return st, nil
}

func toFloat32(dtype safetensors.DType, raw []byte) ([]float32, error) {
	switch dtype {
	case safetensors.F32:
		if len(raw)%4 != 0 {
			return nil, fmt.Errorf("invalid F32 byte length %d", len(raw))
		}
		out := make([]float32, len(raw)/4)
		for i := range out {
			bits := binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
			out[i] = math.Float32frombits(bits)
		}
		return out, nil
	case safetensors.BF16:
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("invalid BF16 byte length %d", len(raw))
		}
		out := make([]float32, len(raw)/2)
		for i := range out {
			bits16 := binary.LittleEndian.Uint16(raw[i*2 : i*2+2])
			bits32 := uint32(bits16) << 16
			out[i] = math.Float32frombits(bits32)
		}
		return out, nil
	case safetensors.F16:
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("invalid F16 byte length %d", len(raw))
		}
		out := make([]float32, len(raw)/2)
		for i := range out {
			h := binary.LittleEndian.Uint16(raw[i*2 : i*2+2])
			out[i] = float16ToFloat32(h)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported tensor dtype %s", dtype.String())
	}
}

// float16ToFloat32 converts IEEE-754 binary16 to float32.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1F
	frac := uint32(h & 0x3FF)

	var bits uint32
	switch exp {
	case 0:
		if frac == 0 {
			bits = sign << 31
		} else {
			// subnormal
			e := int32(-14)
			f := frac
			for (f & 0x400) == 0 {
				f <<= 1
				e--
			}
			f &= 0x3FF
			exp32 := uint32(e + 127)
			frac32 := f << 13
			bits = (sign << 31) | (exp32 << 23) | frac32
		}
	case 0x1F:
		// inf / nan
		bits = (sign << 31) | (0xFF << 23) | (frac << 13)
	default:
		exp32 := exp - 15 + 127
		frac32 := frac << 13
		bits = (sign << 31) | (exp32 << 23) | frac32
	}
	return math.Float32frombits(bits)
}
