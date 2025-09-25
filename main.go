package main

import (
	"bytes"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"randomsentensbot/core"
	"strings"
)

func main() {
	key := os.Getenv("MSTDN_KEY")
	server := os.Getenv("MSTDN_SERVER")

	if key == "" || server == "" {
		log.Fatal("MSTDN_KEY and MSTDN_SERVER must be set")
	}

	model, err := core.LoadModel("model.bin", 0.01)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	selects := pick(model.Tokenizer.Count, *model.Tokenizer)
	data := model.Predict(model.Tokenizer.Tokens[selects])

	var content strings.Builder

	content.WriteString(selects)
	content.WriteString(" ")
	content.WriteString(model.Tokenizer.GetToken(data))
	content.WriteString(" ")

	for i := 0; i < 8; i++ {
		data = model.Predict(data)
		if model.Tokenizer.GetToken(data) == core.ENDTOKEN {
			break
		}
		content.WriteString(model.Tokenizer.GetToken(data))
		content.WriteString(" ")
	}

	fmt.Printf("Generated content: %s\n", content.String())

	// Post to Mastodon
	apiURL := fmt.Sprintf("%s/api/v1/statuses", server)
	formData := url.Values{}
	formData.Set("status", content.String())

	req, err := http.NewRequest("POST", apiURL, bytes.NewBufferString(formData.Encode()))
	if err != nil {
		log.Fatalf("Error creating request: %v", err)
	}

	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Fatalf("Error sending request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Fatalf("Failed to post status: %s", resp.Status)
	}

	fmt.Println("Successfully posted to Mastodon!")
}

func pick(length int, dict core.Tokenizer) string {
	rndn := rand.Intn(length)
	for key := range dict.Tokens {
		if rndn == 0 {
			return key
		}
		rndn--
	}
	panic("unreachable!")
}
