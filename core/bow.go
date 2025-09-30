package core

import (
	"runtime"
	"sort"
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

// ReorderTokensByRelation re-indexes all tokens to group related tokens closer together in the index space.
// It calculates a "centrality" score for each token based on the frequency-weighted average index of the tokens that follow it.
// It then sorts tokens based on this score and re-assigns indices to all tokens, ensuring no collisions.
// Finally, it rebuilds the UnigramFreq and UnigramMap to reflect the new indices.
func (t *Tokenizer) ReorderTokensByRelation() {
	if len(t.Tokens) < 2 {
		return // Nothing to reorder
	}

	// Create a reverse mapping for efficient index-to-token lookups.
	idxToToken := make(map[int]string, len(t.Tokens))
	for token, idx := range t.Tokens {
		idxToToken[idx] = token
	}

	// Use a map to calculate a "centrality" score for each token.
	// The score is the weighted-average index of tokens that follow it.
	tokenScores := make(map[int]int)

	// To ensure deterministic behavior, iterate over sorted keys.
	sortedTokIDs := make([]int, 0, len(t.UnigramFreq))
	for tokID := range t.UnigramFreq {
		sortedTokIDs = append(sortedTokIDs, tokID)
	}
	sort.Ints(sortedTokIDs)

	for _, tokID := range sortedTokIDs {
		subsequentTokens := t.UnigramFreq[tokID]
		if len(subsequentTokens) == 0 {
			continue
		}

		// Calculate the weighted average index of subsequent tokens.
		weightedIndexSum := 0
		totalFrequency := 0
		for relatedIdx, freq := range subsequentTokens {
			weightedIndexSum += relatedIdx * freq
			totalFrequency += freq
		}

		if totalFrequency > 0 {
			averageIndex := weightedIndexSum / totalFrequency
			tokenScores[tokID] = averageIndex
		}
	}

	// Create a list of all tokens to be re-indexed.
	type tokenInfo struct {
		id    int
		token string
		score int
	}
	allTokens := make([]tokenInfo, 0, len(t.Tokens))
	for token, id := range t.Tokens {
		score, exists := tokenScores[id]
		if !exists {
			// Assign a default high score to tokens with no subsequent relations
			// to place them after the more "central" tokens.
			score = int(^uint(0) >> 1) // Max int
		}
		allTokens = append(allTokens, tokenInfo{id: id, token: token, score: score})
	}

	// Sort tokens by their score, then alphabetically for stability.
	sort.Slice(allTokens, func(i, j int) bool {
		if allTokens[i].score != allTokens[j].score {
			return allTokens[i].score < allTokens[j].score
		}
		return allTokens[i].token < allTokens[j].token
	})

	// Build the new token map and a map from old index to new index.
	newTokens := make(map[string]int, len(t.Tokens))
	oldToNewIndexMap := make(map[int]int, len(t.Tokens))
	for i, info := range allTokens {
		newTokens[info.token] = i
		oldToNewIndexMap[info.id] = i
	}

	// Create new UnigramFreq and UnigramMap using the new indices.
	newUnigramFreq := make(map[int]map[int]int)
	for oldTokID, oldFreqMap := range t.UnigramFreq {
		newTokID, ok := oldToNewIndexMap[oldTokID]
		if !ok {
			continue
		}

		newFreqMap := make(map[int]int)
		for oldNextID, freq := range oldFreqMap {
			if newNextID, ok := oldToNewIndexMap[oldNextID]; ok {
				newFreqMap[newNextID] = freq
			}
		}
		if len(newFreqMap) > 0 {
			newUnigramFreq[newTokID] = newFreqMap
		}
	}

	newUnigramMap := make(map[int]int)
	for oldTokID, oldNextID := range t.UnigramMap {
		newTokID, okT := oldToNewIndexMap[oldTokID]
		newNextID, okN := oldToNewIndexMap[oldNextID]
		if okT && okN {
			newUnigramMap[newTokID] = newNextID
		}
	}

	// Atomically replace the old data structures.
	t.Tokens = newTokens
	t.UnigramFreq = newUnigramFreq
	t.UnigramMap = newUnigramMap
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
