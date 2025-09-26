package core

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	"golang.org/x/net/html"
)

// Structs to parse the outbox.json format
type Outbox struct {
	OrderedItems []Item `json:"orderedItems"`
}

// Item uses json.RawMessage to handle cases where 'object' can be a JSON object or a string (URL).
type Item struct {
	Object json.RawMessage `json:"object"`
}

// Object struct is for unmarshaling the actual object when it's not a string.
type Object struct {
	Content string `json:"content"`
}

func stripHTML(htmlString string) string {
	var textContent strings.Builder
	t := html.NewTokenizer(strings.NewReader(htmlString))

	for {
		tt := t.Next()
		if tt == html.ErrorToken {
			break // End of document
		}
		if tt == html.TextToken {
			textContent.WriteString(t.Token().Data)
		}
	}

	return strings.TrimSpace(textContent.String())
}

// This function is modified to correctly parse the outbox.json format.
func TestMakeData(t *testing.T) {
	bd, fe := os.ReadFile("./outbox.json")
	if fe != nil {
		t.Fatalf("Error reading outbox.json: %v", fe)
	}

	var outbox Outbox
	xe := json.Unmarshal(bd, &outbox)
	if xe != nil {
		t.Fatalf("Error unmarshaling outbox.json: %v", xe)
	}

	fmt.Println("Inserting Sentences from outbox.json...")
	var sentens []string
	for _, item := range outbox.OrderedItems {
		// Check if the RawMessage is a JSON object (starts with '{')
		if len(item.Object) > 0 && item.Object[0] == '{' {
			var obj Object
			err := json.Unmarshal(item.Object, &obj)
			if err == nil {
				cleanedText := stripHTML(obj.Content)
				if cleanedText != "" {
					sentens = append(sentens, cleanedText)
				}
			}
		}
		// If item.Object is a string (URL), it's ignored.
	}

	// Shuffle the collected sentences to get a random sample.
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(sentens), func(i, j int) {
		sentens[i], sentens[j] = sentens[j], sentens[i]
	})

	// Limit to 8000 sentences if more are available.
	if len(sentens) > 8000 {
		sentens = sentens[:8000]
	}

	if len(sentens) == 0 {
		t.Fatal("No sentences extracted from outbox.json")
	}

	CreateAndTrainModel(sentens, 0.1, 15, "model.bin")
	t.Logf("Model created successfully from %d sentences.", len(sentens))
}

func TestLongSentense(t *testing.T) {
	model, err := LoadModel("model.bin", 0.01)
	if err != nil {
		t.Error(err)
		t.Fail()
	}

	pick := func(length int, dict Tokenizer) string {
		rndn := rand.Intn(length)
		for key := range dict.Tokens {
			if rndn == 0 {
				return key
			}
			rndn--
		}
		panic("unreachable!")
	}

	selects := pick(model.Tokenizer.Count, *model.Tokenizer)

	data := model.Predict(model.Tokenizer.Tokens[selects])
	fmt.Printf("Input: %s, Predict: %s\n", selects, model.Tokenizer.GetToken(data))

	for {
		data = model.Predict(data)
		if model.Tokenizer.GetToken(data) == ENDTOKEN {
			break
		}
		fmt.Printf("%s ", model.Tokenizer.GetToken(data))
	}
}
