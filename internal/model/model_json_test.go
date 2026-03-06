package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGenerateJSON_ObjectShape(t *testing.T) {
	modelPath := os.Getenv("GO_LLAMA3_MODEL_PATH")
	if modelPath == "" {
		t.Skip("set GO_LLAMA3_MODEL_PATH to run integration inference tests")
	}
	if st, err := os.Stat(modelPath); err != nil {
		t.Skip("model path not accessible")
	} else if st.IsDir() {
		if _, err := os.Stat(filepath.Join(modelPath, "config.json")); err != nil {
			t.Skip("model dir missing config.json")
		}
	} else if !strings.HasSuffix(strings.ToLower(modelPath), ".gguf") {
		t.Skip("model path must be directory or .gguf file")
	}

	m, err := Load(modelPath)
	if err != nil {
		t.Fatalf("LoadFromDir: %v", err)
	}

	prompt := `Return only valid JSON object with keys "name" and "age". Example: {"name":"Alice","age":30}.`
	out, err := m.Generate(prompt, GenerateOptions{
		MaxNewTokens: 64,
		Temperature:  0.1,
		TopK:         20,
		TopP:         0.9,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	jsonText, ok := extractJSONObject(out)
	if !ok {
		t.Fatalf("no JSON object found in output: %q", out)
	}

	var parsed map[string]any
	if err := json.Unmarshal([]byte(jsonText), &parsed); err != nil {
		t.Fatalf("invalid JSON %q: %v", jsonText, err)
	}
	if _, ok := parsed["name"]; !ok {
		t.Fatalf("missing key 'name' in %v", parsed)
	}
	if _, ok := parsed["age"]; !ok {
		t.Fatalf("missing key 'age' in %v", parsed)
	}
}

func TestGenerateJSON_ArrayShape(t *testing.T) {
	modelPath := os.Getenv("GO_LLAMA3_MODEL_PATH")
	if modelPath == "" {
		t.Skip("set GO_LLAMA3_MODEL_PATH to run integration inference tests")
	}

	m, err := Load(modelPath)
	if err != nil {
		t.Fatalf("LoadFromDir: %v", err)
	}

	prompt := `Return only a JSON array with 2 color names, like ["red","blue"].`
	out, err := m.Generate(prompt, GenerateOptions{
		MaxNewTokens: 48,
		Temperature:  0.1,
		TopK:         20,
		TopP:         0.9,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	jsonText, ok := extractJSONArray(out)
	if !ok {
		t.Fatalf("no JSON array found in output: %q", out)
	}
	var arr []string
	if err := json.Unmarshal([]byte(jsonText), &arr); err != nil {
		t.Fatalf("invalid JSON array %q: %v", jsonText, err)
	}
	if len(arr) != 2 {
		t.Fatalf("expected 2 entries, got %d (%v)", len(arr), arr)
	}
}

func extractJSONObject(s string) (string, bool) {
	s = stripCodeFence(strings.TrimSpace(s))
	start := strings.IndexByte(s, '{')
	end := strings.LastIndexByte(s, '}')
	if start == -1 || end == -1 || end <= start {
		return "", false
	}
	return s[start : end+1], true
}

func extractJSONArray(s string) (string, bool) {
	s = stripCodeFence(strings.TrimSpace(s))
	start := strings.IndexByte(s, '[')
	end := strings.LastIndexByte(s, ']')
	if start == -1 || end == -1 || end <= start {
		return "", false
	}
	return s[start : end+1], true
}

func stripCodeFence(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "```json")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "```")
	return strings.TrimSpace(s)
}
