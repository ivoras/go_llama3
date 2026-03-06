package loader

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ivora/go_llama3/internal/config"
)

// GGUF magic and version.
const (
	ggufMagic = 0x46554747 // "GGUF" in little-endian
	ggufAlign = 32
)

// GGML type enum (subset we support).
const (
	ggmlTypeF32  = 0
	ggmlTypeF16  = 1
	ggmlTypeBF16 = 30
)

// GGUF value types for metadata.
const (
	ggufTypeUint8   = 0
	ggufTypeInt8    = 1
	ggufTypeUint16  = 2
	ggufTypeInt16   = 3
	ggufTypeUint32  = 4
	ggufTypeInt32   = 5
	ggufTypeFloat32 = 6
	ggufTypeBool    = 7
	ggufTypeString  = 8
	ggufTypeArray   = 9
	ggufTypeUint64  = 10
	ggufTypeInt64   = 11
	ggufTypeFloat64 = 12
)

// GGUFLoader loads weights from a single GGUF file.
type GGUFLoader struct {
	path     string
	meta     map[string]any
	tensors  map[string]ggufTensorInfo
	data     []byte
	dataBase int64 // file offset where tensor data starts
}

type ggufTensorInfo struct {
	Name   string
	Shape  []int
	Type   uint32
	Offset uint64
}

// NewGGUF loads a GGUF file from the given path.
func NewGGUF(path string) (*GGUFLoader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open gguf %q: %w", path, err)
	}
	defer f.Close()

	// Header
	var magic, version uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, err
	}
	if magic != ggufMagic {
		return nil, fmt.Errorf("invalid GGUF magic: got 0x%x, want 0x%x", magic, ggufMagic)
	}
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, err
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("unsupported GGUF version: %d", version)
	}

	var nTensors, nKV uint64
	if err := binary.Read(f, binary.LittleEndian, &nTensors); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &nKV); err != nil {
		return nil, err
	}

	// Parse KV pairs
	meta := make(map[string]any)
	for i := uint64(0); i < nKV; i++ {
		key, err := readGGUFString(f)
		if err != nil {
			return nil, fmt.Errorf("kv %d key: %w", i, err)
		}
		var vtype uint32
		if err := binary.Read(f, binary.LittleEndian, &vtype); err != nil {
			return nil, err
		}
		val, err := readGGUFValue(f, vtype, uint64(version))
		if err != nil {
			return nil, fmt.Errorf("kv %q: %w", key, err)
		}
		meta[key] = val
	}

	// Parse tensor infos
	alignment := ggufAlign
	if a, ok := meta["general.alignment"].(uint32); ok {
		alignment = int(a)
	}
	if alignment < 1 {
		alignment = 1
	}

	tensors := make(map[string]ggufTensorInfo)
	var dataOffset int64
	for i := uint64(0); i < nTensors; i++ {
		name, err := readGGUFString(f)
		if err != nil {
			return nil, fmt.Errorf("tensor %d name: %w", i, err)
		}
		var nDims uint32
		if err := binary.Read(f, binary.LittleEndian, &nDims); err != nil {
			return nil, err
		}
		if nDims > 4 {
			return nil, fmt.Errorf("tensor %q: invalid n_dims %d", name, nDims)
		}
		var ne [4]uint64
		for d := uint32(0); d < nDims; d++ {
			if err := binary.Read(f, binary.LittleEndian, &ne[d]); err != nil {
				return nil, err
			}
		}
		var vtype uint32
		if err := binary.Read(f, binary.LittleEndian, &vtype); err != nil {
			return nil, err
		}
		var offset uint64
		if err := binary.Read(f, binary.LittleEndian, &offset); err != nil {
			return nil, err
		}

		shape := make([]int, nDims)
		for d := uint32(0); d < nDims; d++ {
			shape[d] = int(ne[d])
		}
		tensors[name] = ggufTensorInfo{
			Name:   name,
			Shape:  shape,
			Type:   vtype,
			Offset: offset,
		}
		// Compute tensor size for alignment
		elCount := int64(1)
		for _, s := range shape {
			elCount *= int64(s)
		}
		typeSize := ggmlTypeSize(vtype)
		blckSize := ggmlBlckSize(vtype)
		if blckSize == 0 {
			blckSize = 1
		}
		size := (elCount * int64(typeSize)) / int64(blckSize)
		dataOffset += pad(size, int64(alignment))
	}

	dataBase, _ := f.Seek(0, 1)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read gguf data: %w", err)
	}

	return &GGUFLoader{
		path:     path,
		meta:     meta,
		tensors:  tensors,
		data:     data,
		dataBase: dataBase,
	}, nil
}

func readGGUFString(f *os.File) (string, error) {
	var n uint64
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return "", err
	}
	buf := make([]byte, n)
	if _, err := f.Read(buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readGGUFValue(f *os.File, vtype uint32, version uint64) (any, error) {
	switch vtype {
	case ggufTypeUint8:
		var v uint8
		err := binary.Read(f, binary.LittleEndian, &v)
		return uint32(v), err
	case ggufTypeInt8:
		var v int8
		err := binary.Read(f, binary.LittleEndian, &v)
		return int32(v), err
	case ggufTypeUint16:
		var v uint16
		err := binary.Read(f, binary.LittleEndian, &v)
		return uint32(v), err
	case ggufTypeInt16:
		var v int16
		err := binary.Read(f, binary.LittleEndian, &v)
		return int32(v), err
	case ggufTypeUint32:
		var v uint32
		err := binary.Read(f, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt32:
		var v int32
		err := binary.Read(f, binary.LittleEndian, &v)
		return v, err
	case ggufTypeFloat32:
		var v float32
		err := binary.Read(f, binary.LittleEndian, &v)
		return v, err
	case ggufTypeBool:
		var v bool
		err := binary.Read(f, binary.LittleEndian, &v)
		return v, err
	case ggufTypeString:
		return readGGUFString(f)
	case ggufTypeArray:
		var arrType uint32
		if err := binary.Read(f, binary.LittleEndian, &arrType); err != nil {
			return nil, err
		}
		var n uint64
		if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
			return nil, err
		}
		arr := make([]any, n)
		for i := uint64(0); i < n; i++ {
			val, err := readGGUFValue(f, arrType, version)
			if err != nil {
				return nil, err
			}
			arr[i] = val
		}
		return arr, nil
	case ggufTypeUint64:
		var v uint64
		err := binary.Read(f, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt64:
		var v int64
		err := binary.Read(f, binary.LittleEndian, &v)
		return v, err
	case ggufTypeFloat64:
		var v float64
		err := binary.Read(f, binary.LittleEndian, &v)
		return v, err
	default:
		return nil, fmt.Errorf("unsupported gguf value type %d", vtype)
	}
}

func ggmlTypeSize(t uint32) int {
	switch t {
	case ggmlTypeF32:
		return 4
	case ggmlTypeF16, ggmlTypeBF16:
		return 2
	default:
		return 0
	}
}

func ggmlBlckSize(t uint32) int {
	switch t {
	case ggmlTypeF32, ggmlTypeF16, ggmlTypeBF16:
		return 1
	default:
		return 0
	}
}

func pad(x, n int64) int64 {
	return (x + n - 1) & ^(n - 1)
}

// Metadata returns the parsed GGUF metadata.
func (g *GGUFLoader) Metadata() map[string]any {
	return g.meta
}

// TensorNames returns all tensor names.
func (g *GGUFLoader) TensorNames() []string {
	names := make([]string, 0, len(g.tensors))
	for n := range g.tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// ggufNameAlternatives returns possible GGUF names for a HuggingFace name (some GGUF variants use different names).
func ggufNameAlternatives(hfName string) []string {
	main := ggufNameForHF(hfName)
	switch hfName {
	case "model.norm.weight":
		return []string{main, "output_norm.weight", "output.norm.weight"}
	default:
		return []string{main}
	}
}

// LoadTensor loads a tensor by name. Accepts HuggingFace-style names; translates to GGUF names internally.
// Supports F32, F16, BF16 only.
func (g *GGUFLoader) LoadTensor(name string) (Tensor, error) {
	candidates := append([]string{name}, ggufNameAlternatives(name)...)
	var info ggufTensorInfo
	var ok bool
	for _, c := range candidates {
		if info, ok = g.tensors[c]; ok {
			break
		}
	}
	if !ok {
		return Tensor{}, fmt.Errorf("tensor %q not found in GGUF (tried: %v)", name, candidates)
	}
	if ggmlTypeSize(info.Type) == 0 {
		return Tensor{}, fmt.Errorf("tensor %q: unsupported type %d (only F32/F16/BF16)", name, info.Type)
	}

	elCount := 1
	for _, s := range info.Shape {
		elCount *= s
	}
	typeSize := ggmlTypeSize(info.Type)
	blck := ggmlBlckSize(info.Type)
	if blck == 0 {
		blck = 1
	}
	size := (elCount * typeSize) / blck

	start := int(g.dataBase) + int(info.Offset)
	if start+size > len(g.data) {
		return Tensor{}, fmt.Errorf("tensor %q: data out of bounds", name)
	}
	raw := g.data[start : start+size]

	var data []float32
	switch info.Type {
	case ggmlTypeF32:
		data = make([]float32, elCount)
		for i := 0; i < elCount; i++ {
			data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4 : i*4+4]))
		}
	case ggmlTypeF16:
		data = make([]float32, elCount)
		for i := 0; i < elCount; i++ {
			data[i] = float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2 : i*2+2]))
		}
	case ggmlTypeBF16:
		data = make([]float32, elCount)
		for i := 0; i < elCount; i++ {
			bits16 := binary.LittleEndian.Uint16(raw[i*2 : i*2+2])
			bits32 := uint32(bits16) << 16
			data[i] = math.Float32frombits(bits32)
		}
	default:
		return Tensor{}, fmt.Errorf("tensor %q: unsupported type %d", name, info.Type)
	}

	return Tensor{
		Name:  name,
		Shape: info.Shape,
		Data:  data,
	}, nil
}

// ggufNameForHF returns the GGUF tensor name for a HuggingFace-style name.
func ggufNameForHF(hfName string) string {
	switch hfName {
	case "model.embed_tokens.weight":
		return "token_embd.weight"
	case "lm_head.weight":
		return "output.weight"
	case "model.norm.weight":
		return "norm.weight"
	}
	if strings.HasPrefix(hfName, "model.layers.") {
		rest := strings.TrimPrefix(hfName, "model.layers.")
		parts := strings.SplitN(rest, ".", 2)
		if len(parts) == 2 {
			layerIdx := parts[0]
			tensor := parts[1]
			m := map[string]string{
				"input_layernorm.weight":       "attn_norm.weight",
				"self_attn.q_proj.weight":      "attn_q.weight",
				"self_attn.k_proj.weight":      "attn_k.weight",
				"self_attn.v_proj.weight":      "attn_v.weight",
				"self_attn.o_proj.weight":      "attn_output.weight",
				"post_attention_layernorm.weight": "ffn_norm.weight",
				"mlp.gate_proj.weight":         "ffn_gate.weight",
				"mlp.up_proj.weight":           "ffn_up.weight",
				"mlp.down_proj.weight":          "ffn_down.weight",
			}
			if gguf, ok := m[tensor]; ok {
				return "blk." + layerIdx + "." + gguf
			}
		}
	}
	return hfName
}

// GGUFConfig holds config parsed from GGUF metadata.
type GGUFConfig struct {
	HiddenSize            int
	IntermediateSize      int
	NumAttentionHeads     int
	NumHiddenLayers       int
	NumKeyValueHeads      int
	VocabSize             int
	MaxPositionEmbeddings int
	RopeTheta             float64
	RMSNormEps            float64
}

// ConfigFromGGUF builds a GGUFConfig from GGUF metadata.
func ConfigFromGGUF(meta map[string]any) (GGUFConfig, error) {
	cfg := GGUFConfig{}

	getU32 := func(keys ...string) (uint32, bool) {
		for _, k := range keys {
			if v, ok := meta[k]; ok {
				switch x := v.(type) {
				case uint32:
					return x, true
				case int:
					return uint32(x), true
				case uint64:
					return uint32(x), true
				case int64:
					return uint32(x), true
				}
			}
		}
		return 0, false
	}
	getF64 := func(keys ...string) (float64, bool) {
		for _, k := range keys {
			if v, ok := meta[k]; ok {
				switch x := v.(type) {
				case float64:
					return x, true
				case float32:
					return float64(x), true
				case uint32:
					return float64(x), true
				}
			}
		}
		return 0, false
	}

	if v, ok := getU32("llama.embedding_length", "embedding_length"); ok {
		cfg.HiddenSize = int(v)
	} else {
		return cfg, fmt.Errorf("missing embedding_length in GGUF metadata")
	}
	if v, ok := getU32("llama.block_count", "block_count"); ok {
		cfg.NumHiddenLayers = int(v)
	} else {
		return cfg, fmt.Errorf("missing block_count in GGUF metadata")
	}
	if v, ok := getU32("llama.attention.head_count", "attention.head_count"); ok {
		cfg.NumAttentionHeads = int(v)
	} else {
		return cfg, fmt.Errorf("missing attention.head_count in GGUF metadata")
	}
	if v, ok := getU32("llama.attention.head_count_kv", "attention.head_count_kv"); ok {
		cfg.NumKeyValueHeads = int(v)
	} else {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if v, ok := getU32("llama.feed_forward_length", "feed_forward_length"); ok {
		cfg.IntermediateSize = int(v)
	} else {
		return cfg, fmt.Errorf("missing feed_forward_length in GGUF metadata")
	}
	if v, ok := getU32("llama.vocab_size"); ok {
		cfg.VocabSize = int(v)
	} else if arr, ok := meta["tokenizer.ggml.tokens"].([]any); ok && len(arr) > 0 {
		cfg.VocabSize = len(arr)
	} else {
		return cfg, fmt.Errorf("missing vocab_size in GGUF metadata")
	}
	if v, ok := getU32("llama.context_length", "context_length"); ok {
		cfg.MaxPositionEmbeddings = int(v)
	} else {
		cfg.MaxPositionEmbeddings = 2048
	}
	if v, ok := getF64("llama.rope.freq_base", "rope.freq_base"); ok {
		cfg.RopeTheta = v
	} else {
		cfg.RopeTheta = 10000.0
	}
	cfg.RMSNormEps = 1e-5

	if err := cfg.Validate(); err != nil {
		return cfg, err
	}
	return cfg, nil
}

func (c GGUFConfig) Validate() error {
	if c.HiddenSize <= 0 {
		return fmt.Errorf("invalid hidden_size: %d", c.HiddenSize)
	}
	if c.NumAttentionHeads <= 0 || c.NumKeyValueHeads <= 0 {
		return fmt.Errorf("invalid head counts")
	}
	if c.NumHiddenLayers <= 0 {
		return fmt.Errorf("invalid num_hidden_layers: %d", c.NumHiddenLayers)
	}
	if c.VocabSize <= 0 {
		return fmt.Errorf("invalid vocab_size: %d", c.VocabSize)
	}
	return nil
}

// ToConfig converts GGUFConfig to config.Config.
func (c GGUFConfig) ToConfig() config.Config {
	return config.Config{
		HiddenSize:            c.HiddenSize,
		IntermediateSize:      c.IntermediateSize,
		NumAttentionHeads:     c.NumAttentionHeads,
		NumHiddenLayers:       c.NumHiddenLayers,
		NumKeyValueHeads:      c.NumKeyValueHeads,
		VocabSize:             c.VocabSize,
		MaxPositionEmbeddings: c.MaxPositionEmbeddings,
		RopeTheta:             c.RopeTheta,
		RMSNormEps:            c.RMSNormEps,
	}
}

// TokenizerPath returns the directory to look for tokenizer.json (same dir as GGUF file).
func (g *GGUFLoader) TokenizerPath() string {
	return filepath.Dir(g.path)
}
