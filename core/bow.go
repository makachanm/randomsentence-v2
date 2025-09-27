package core

import (
	"runtime"
	"strings"
)

const ENDTOKEN = "[<EOT>]"

type Tokenizer struct {
	Tokens      map[string]int
	UnigramMap  map[int]int
	Count       int
	UnigramFreq map[int]map[int]int // Persisted for reward mechanism
}

func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		Tokens:      make(map[string]int),
		UnigramMap:  make(map[int]int),
		UnigramFreq: make(map[int]map[int]int),
		Count:       0,
	}
}

// AddToken populates the UnigramFreq map to track frequencies.
func (t *Tokenizer) AddToken(token string, nexttoken string) {
	// Ensure the first token exists in the dictionary.
	tokIdx, exists := t.GetTokenIndex(token)
	if !exists {
		t.Tokens[token] = t.Count
		tokIdx = t.Count
		t.Count++
	}

	// Ensure the next token exists in the dictionary.
	nextIdx, exists := t.GetTokenIndex(nexttoken)
	if !exists {
		t.Tokens[nexttoken] = t.Count
		nextIdx = t.Count
		t.Count++
	}

	// Update counts for the next token using the new structure.
	if t.UnigramFreq[tokIdx] == nil {
		t.UnigramFreq[tokIdx] = make(map[int]int)
	}
	t.UnigramFreq[tokIdx][nextIdx]++
}

// BuildUnigramMap iterates through the counts and selects the most frequent next token for each token.
func (t *Tokenizer) BuildUnigramMap() {
	for tokID, freqMap := range t.UnigramFreq {
		maxFreq := 0
		bestNextID := -1
		for nextID, freq := range freqMap {
			if freq > maxFreq {
				maxFreq = freq
				bestNextID = nextID
			}
		}
		if bestNextID != -1 {
			t.UnigramMap[tokID] = bestNextID
		}
	}

	// We no longer clear the frequency map to use it for rewards.
	runtime.GC()
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

	for i := 0; i < len(words)-1; i++ {
		t.AddToken(words[i], words[i+1])
	}
}