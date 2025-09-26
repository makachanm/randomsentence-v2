package core

import (
	"runtime"
	"strings"
)

const ENDTOKEN = "[<EOT>]"

type Tokenizer struct {
	Tokens     map[string]int
	UnigramMap map[int]int
	Count      int

	// Unexported field for memory optimization. It will be used for counting
	// and then discarded. It won't be saved in the model file.
	nextTokenCounts map[[2]int]int
}

func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		Tokens:          make(map[string]int),
		UnigramMap:      make(map[int]int),
		nextTokenCounts: make(map[[2]int]int), // Use the new efficient structure
		Count:           0,
	}
}

// AddToken populates the nextTokenCounts map to track frequencies.
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
	pair := [2]int{tokIdx, nextIdx}
	t.nextTokenCounts[pair]++
}

// BuildUnigramMap iterates through the counts and selects the most frequent next token for each token.
func (t *Tokenizer) BuildUnigramMap() {
	// This map stores the count of the best next token found so far.
	maxCounts := make(map[int]int)

	for pair, count := range t.nextTokenCounts {
		tokID := pair[0]
		nextID := pair[1]

		// Check if the current pair's count is greater than the max count
		// seen so far for this tokID.
		if count > maxCounts[tokID] {
			maxCounts[tokID] = count
			t.UnigramMap[tokID] = nextID
		}
	}

	// Free up memory after building the final map.
	t.nextTokenCounts = nil
	maxCounts = nil
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
