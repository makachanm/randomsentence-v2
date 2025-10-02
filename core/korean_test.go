package core

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"regexp"
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

func removeURLs(text string) string {
	re := regexp.MustCompile(`https?://[\w\d\.\-/?=#&%@]+`)
	return re.ReplaceAllString(text, "")
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
			if name, _ := t.TagName(); string(name) == "a" {
				t.Next()
				continue // Skip links
			} else {
				textContent.WriteString(t.Token().Data)
			}
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
	fmt.Printf("Model created successfully from %d sentences.", len(sentens))
}

func TestMakeDatasetData(t *testing.T) {
	bd, fe := os.ReadFile("./steam.txt")
	if fe != nil {
		t.Fatalf("Error, %v", fe)
	}

	raw := string(bd)
	datas := strings.Split(raw, "\n")

	rand.Shuffle(len(datas), func(i, j int) {
		datas[i], datas[j] = datas[j], datas[i]
	})

	// Limit to 8000 sentences if more are available.
	if len(datas) > 5000 {
		datas = datas[:5000]
	}

	CreateAndTrainModel(datas, 0.1, 5, "model_x.bin")
	fmt.Printf("Model created successfully from %d sentences.", len(datas))
}

func TestMakeTotalData(t *testing.T) {
	var sentens []string

	fmt.Println("Inserting Sentences from outbox.json, steam.txt, and kommongen_train.json...")

	// --- Read from outbox.json ---
	bd, fe := os.ReadFile("./outbox.json")
	if fe != nil {
		t.Logf("Could not read outbox.json: %v", fe)
	} else {
		var outbox Outbox
		if err := json.Unmarshal(bd, &outbox); err != nil {
			t.Fatalf("Error unmarshaling outbox.json: %v", err)
		}
		for _, item := range outbox.OrderedItems {
			if len(item.Object) > 0 && item.Object[0] == '{' {
				var obj Object
				if err := json.Unmarshal(item.Object, &obj); err == nil {
					cleanedText := stripHTML(obj.Content)
					cleanedText = removeURLs(cleanedText)
					if cleanedText != "" {
						sentens = append(sentens, cleanedText)
					}
				}
			}
		}
	}

	// --- Read from steam.txt ---
	bx, fe := os.ReadFile("./steam.txt")
	if fe != nil {
		t.Logf("Could not read steam.txt: %v", fe)
	} else {
		raw := string(bx)
		rdatas := strings.Split(raw, "\n")
		for i, line := range rdatas {
			rdatas[i] = removeURLs(line)
		}
		rand.Shuffle(len(rdatas), func(i, j int) {
			rdatas[i], rdatas[j] = rdatas[j], rdatas[i]
		})
		sentens = append(sentens, rdatas[:7000]...)
	}

	// --- Read from kommongen_train.json ---
	kd, ke := os.ReadFile("./kommongen_train.json")
	if ke != nil {
		t.Logf("Could not read kommongen_train.json: %v", ke)
	} else {
		lines := strings.Split(string(kd), "\n")
		rand.Shuffle(len(lines), func(i, j int) {
			lines[i], lines[j] = lines[j], lines[i]
		})
		for i, line := range lines {
			if i > 7000 {
				break
			}

			if strings.TrimSpace(line) == "" {
				continue
			}
			var item KommonGenItem
			if err := json.Unmarshal([]byte(line), &item); err != nil {
				t.Logf("Skipping malformed line in kommongen_train.json: %v", err)
				continue
			}
			if item.Scene != "" {
				cleanedScene := removeURLs(item.Scene)
				if cleanedScene != "" {
					sentens = append(sentens, cleanedScene)
				}
			}
		}
	}

	if len(sentens) == 0 {
		t.Fatal("No sentences extracted from any data source.")
	}

	// Optional: Limit the total number of sentences.
	//if len(sentens) > 10000 {
	//	sentens = sentens[:10000]
	//}

	CreateAndTrainModel(sentens, 0.1, 5, "model.bin")
}

func TestLongSentense(t *testing.T) {
	model, err := LoadModel("model.bin", 0.1)
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
	fmt.Println(selects)
	data := model.Predict(model.Tokenizer.Tokens[selects], make([]int, 0))
	fmt.Println(model.Tokenizer.GetToken(data))

	fmt.Printf("Input: %s, Predict: %s\n", selects, model.Tokenizer.GetToken(data))

	var intmake []int
	for {
		data = model.Predict(data, intmake)
		if model.Tokenizer.GetToken(data) == ENDTOKEN {
			break
		}
		intmake = append(intmake, data)
		fmt.Printf("%s ", model.Tokenizer.GetToken(data))
	}

}

// KommonGenItem defines the structure for an item in the kommongen_train.json file,
// based on the user-provided example.
type KommonGenItem struct {
	ConceptSet string `json:"concept-set"`
	Scene      string `json:"scene"`
}

func TestMakeKommongenData(t *testing.T) {
	// The file 'core/kommongen_train.json' is ignored by .gitignore.
	// Please ensure the file exists at that path.
	bd, err := os.ReadFile("./kommongen_train.json")
	if err != nil {
		t.Fatalf("Error reading core/kommongen_train.json: %v. Make sure the file exists.", err)
	}

	// The file is in JSON Lines format (one JSON object per line).
	// We need to read and parse it line by line.
	lines := strings.Split(string(bd), "\n")

	fmt.Println("Inserting Sentences from kommongen_train.json...")
	var sentens []string
	for i, line := range lines {
		// Skip empty lines
		if strings.TrimSpace(line) == "" {
			continue
		}

		if i > 1000 {
			break
		}

		var item KommonGenItem
		err := json.Unmarshal([]byte(line), &item)
		if err != nil {
			t.Logf("Skipping malformed line in kommongen_train.json: %v", err)
			continue
		}

		// We use the "scene" as the sentence for training.
		if item.Scene != "" {
			sentens = append(sentens, item.Scene)
		}
	}

	if len(sentens) == 0 {
		t.Fatal("No sentences extracted from kommongen_train.json")
	}

	// You can now use the 'sentens' slice to train your model, similar to other test functions.
	fmt.Printf("Extracted %d sentences from kommongen_train.json.\n", len(sentens))

	// Example of training with the extracted data:
	rand.Shuffle(len(sentens), func(i, j int) {
		sentens[i], sentens[j] = sentens[j], sentens[i]
	})
	if len(sentens) > 5000 {
		sentens = sentens[:5000]
	}
	CreateAndTrainModel(sentens, 0.1, 5, "kommongen_model.bin")
	fmt.Printf("Model created successfully from %d sentences.", len(sentens))
}
