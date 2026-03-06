package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type tokenizerFile struct {
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
	Model struct {
		Vocab  map[string]int `json:"vocab"`
		Merges []string       `json:"merges"`
	} `json:"model"`
}

// Tokenizer is a pragmatic tokenizer that can load HuggingFace tokenizer.json.
// It uses exact-token and whitespace-token matching for speed and simplicity.
type Tokenizer struct {
	tokenToID map[string]int32
	idToToken map[int32]string
	unknownID int32
}

func LoadFromModelDir(modelDir string) (*Tokenizer, error) {
	return Load(filepath.Join(modelDir, "tokenizer.json"))
}

// LoadFromGGUF builds a tokenizer from GGUF metadata (tokenizer.ggml.tokens array).
// Falls back to tokenizer.json at tokenizerPath if the GGUF has no embedded tokenizer.
func LoadFromGGUF(meta map[string]any, tokenizerPath string) (*Tokenizer, error) {
	tokensVal, ok := meta["tokenizer.ggml.tokens"]
	if !ok {
		return Load(tokenizerPath)
	}
	arr, ok := tokensVal.([]any)
	if !ok || len(arr) == 0 {
		return Load(tokenizerPath)
	}
	tokenToID := make(map[string]int32, len(arr))
	idToToken := make(map[int32]string, len(arr))
	for i, v := range arr {
		s, ok := v.(string)
		if !ok {
			continue
		}
		id := int32(i)
		tokenToID[s] = id
		idToToken[id] = s
	}
	t := &Tokenizer{
		tokenToID: tokenToID,
		idToToken: idToToken,
		unknownID: chooseUnknownID(tokenToID),
	}
	return t, nil
}

func Load(path string) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer %q: %w", path, err)
	}
	var tf tokenizerFile
	if err := json.Unmarshal(raw, &tf); err != nil {
		return nil, fmt.Errorf("unmarshal tokenizer %q: %w", path, err)
	}
	tokenToID := make(map[string]int32, len(tf.Model.Vocab)+len(tf.AddedTokens))
	idToToken := make(map[int32]string, len(tf.Model.Vocab)+len(tf.AddedTokens))
	for tok, id := range tf.Model.Vocab {
		tokenToID[tok] = int32(id)
		idToToken[int32(id)] = tok
	}
	for _, a := range tf.AddedTokens {
		tokenToID[a.Content] = int32(a.ID)
		idToToken[int32(a.ID)] = a.Content
	}

	t := &Tokenizer{
		tokenToID: tokenToID,
		idToToken: idToToken,
		unknownID: chooseUnknownID(tokenToID),
	}
	return t, nil
}

func chooseUnknownID(tokenToID map[string]int32) int32 {
	candidates := []string{"<unk>", "<|unk|>", "<|reserved_special_token_0|>"}
	for _, c := range candidates {
		if id, ok := tokenToID[c]; ok {
			return id
		}
	}
	return 0
}

func (t *Tokenizer) Encode(text string) []int32 {
	if text == "" {
		return nil
	}
	// Try exact full-string token first.
	if id, ok := t.tokenToID[text]; ok {
		return []int32{id}
	}
	parts := splitKeepWhitespace(text)
	ids := make([]int32, 0, len(parts))
	for _, p := range parts {
		if p == "" {
			continue
		}
		if id, ok := t.tokenToID[p]; ok {
			ids = append(ids, id)
			continue
		}
		// Greedy byte fallback for unknown tokens.
		ids = append(ids, t.greedyFallback(p)...)
	}
	return ids
}

func splitKeepWhitespace(s string) []string {
	var out []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	inSpace := false
	for _, r := range s {
		isSpace := r == ' ' || r == '\n' || r == '\t' || r == '\r'
		if cur.Len() == 0 {
			inSpace = isSpace
			cur.WriteRune(r)
			continue
		}
		if isSpace == inSpace {
			cur.WriteRune(r)
		} else {
			flush()
			inSpace = isSpace
			cur.WriteRune(r)
		}
	}
	flush()
	return out
}

func (t *Tokenizer) greedyFallback(s string) []int32 {
	ids := make([]int32, 0, len(s))
	runes := []rune(s)
	for i := 0; i < len(runes); {
		found := false
		for j := len(runes); j > i; j-- {
			chunk := string(runes[i:j])
			if id, ok := t.tokenToID[chunk]; ok {
				ids = append(ids, id)
				i = j
				found = true
				break
			}
		}
		if found {
			continue
		}
		ids = append(ids, t.unknownID)
		i++
	}
	return ids
}

func (t *Tokenizer) Decode(ids []int32) string {
	var b strings.Builder
	for _, id := range ids {
		if tok, ok := t.idToToken[id]; ok {
			b.WriteString(tok)
		}
	}
	return b.String()
}

func (t *Tokenizer) VocabSize() int {
	return len(t.tokenToID)
}

func (t *Tokenizer) SortedVocab() []string {
	keys := make([]string, 0, len(t.tokenToID))
	for k := range t.tokenToID {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
