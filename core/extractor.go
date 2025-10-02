package core

import (
	"regexp"
	"sort"
	"strings"
)

// Keyword represents a token and its importance score.
type Keyword struct {
	Token string
	Score float32
}

// byScore implements sort.Interface for []Keyword based on the Score field.
type byScore []Keyword

func (a byScore) Len() int           { return len(a) }
func (a byScore) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byScore) Less(i, j int) bool { return a[i].Score < a[j].Score }

// Extractor finds important keywords in a text.
type Extractor struct {
	tokenizer *Tokenizer
}

// NewExtractor creates a new Extractor.
func NewExtractor(model *LinearModel) *Extractor {
	return &Extractor{
		tokenizer: model.Tokenizer,
	}
}

// Extract returns a sorted list of keywords from the input text.
// topK specifies the number of keywords to return. If topK <= 0, all keywords are returned.
func (e *Extractor) Extract(rawInput string, topK int) []Keyword {
	input := filterString(rawInput)
	tokens := strings.Fields(input)

	if len(tokens) == 0 {
		return []Keyword{}
	}

	// Calculate term frequencies in the current text.
	tf := make(map[string]int)
	for _, token := range tokens {
		tf[token]++
	}

	var candidates []Keyword
	for token, freq := range tf {
		// Filter short words
		if len([]rune(token)) < 2 {
			continue
		}

		tokIdx, exists := e.tokenizer.GetTokenIndex(token)

		// Base score from word length
		score := float32(len([]rune(token)))

		// Add specificity bonus
		if exists {
			if nextTokens, ok := e.tokenizer.UnigramFreq[tokIdx]; ok && len(nextTokens) > 0 {
				specificity := 1.0 / float32(len(nextTokens))
				score += specificity * 5.0 // Weight for specificity
			}
		}

		// Penalize by frequency in the current text
		finalScore := score / float32(freq)

		candidates = append(candidates, Keyword{Token: token, Score: finalScore})
	}

	sort.Sort(sort.Reverse(byScore(candidates)))

	if topK > 0 && len(candidates) > topK {
		return candidates[:topK]
	}

	return candidates
}

// filterString cleans the input string by removing HTML tags, links, mentions, and punctuation.
func filterString(input string) string {
	// Remove HTML tags
	re := regexp.MustCompile(`<[^>]*>`)
	input = re.ReplaceAllString(input, "")

	// Remove URLs
	removeLink := regexp.MustCompile(`https?://\S+`)
	input = removeLink.ReplaceAllString(input, "")

	// Remove mentions (e.g., @user)
	removeMention := regexp.MustCompile(`\B@\w+`)
	input = removeMention.ReplaceAllString(input, "")

	// Remove common punctuation that might interfere with tokenization.
	removePunctuation := regexp.MustCompile(`[.,!?;:'"()[\]{}]`)
	input = removePunctuation.ReplaceAllString(input, "")

	// Normalize to lowercase and trim space
	return strings.TrimSpace(strings.ToLower(input))
}