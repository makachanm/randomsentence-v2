package core

import (
	"strings"
)

const ENDTOKEN = "[<EOT>]"

type Tokenizer struct {
	Tokens     map[string]int
	UnigramMap map[int]int
	Count      int
}

func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		Tokens:     make(map[string]int),
		UnigramMap: make(map[int]int),
		Count:      0,
	}
}

func (t *Tokenizer) AddToken(token string, nexttoken string) {
	if _, exists := t.Tokens[token]; exists {
		return
	}
	t.Tokens[token] = t.Count
	t.Count++

	next, nexsit := t.GetTokenIndex(nexttoken) // ensure next token is also added
	if !nexsit {
		t.Tokens[nexttoken] = t.Count
		t.UnigramMap[t.Count-1] = next
		t.Count++
	} else {
		t.UnigramMap[t.Count-1] = next
	}
}

func (t *Tokenizer) GetTokenIndex(token string) (int, bool) {
	idx, exists := t.Tokens[token]
	return idx, exists
}

func (t *Tokenizer) GetToken(idx int) string {
	for k, v := range t.Tokens {
		if v == idx {
			return k
		}
	}
	return ""
}

func (t *Tokenizer) AddtoModel(text string) {
	words := strings.Split(text, " ")
	words = append(words, ENDTOKEN)

	for i := 0; i < len(words)-2; i++ {
		t.AddToken(words[i], words[i+1])
	}
}
