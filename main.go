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
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())

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

	for {
		generatedIndices := []int{}
		var content strings.Builder

		// Pick an initial token.
		initialToken := pick(model.Tokenizer.Count, *model.Tokenizer)
		initialIndex := model.Tokenizer.Tokens[initialToken]
		generatedIndices = append(generatedIndices, initialIndex)
		content.WriteString(initialToken)
		content.WriteString(" ")

		// Continue generating the sentence from the initial token.
		currentIndex := initialIndex
		for i := 0; i < 50; i++ { // Generate up to 50 more tokens
			predictedIndex := model.Predict(currentIndex, generatedIndices)
			predictedToken := model.Tokenizer.GetToken(predictedIndex)

			if predictedToken == core.ENDTOKEN {
				break // Stop at end token
			}

			generatedIndices = append(generatedIndices, predictedIndex)
			content.WriteString(predictedToken)
			content.WriteString(" ")

			currentIndex = predictedIndex
		}

		//content.WriteString("#GenereatedByBot")

		fmt.Printf("Generated content: %s\n", content.String())

		// Post to Mastodon
		apiURL := fmt.Sprintf("%s/api/v1/statuses", server)
		formData := url.Values{}
		formData.Set("status", content.String())

		req, err := http.NewRequest("POST", apiURL, bytes.NewBufferString(formData.Encode()))
		if err != nil {
			log.Printf("Error creating request: %v", err)
			continue
		}

		req.Header.Set("Authorization", "Bearer "+key)
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

		client := &http.Client{}
		resp, err := client.Do(req)
		if err != nil {
			log.Printf("Error sending request: %v", err)
			continue
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			log.Printf("Failed to post status: %s", resp.Status)
		} else {
			fmt.Println("Successfully posted to Mastodon!")
		}

		fmt.Println("Waiting for 20 minutes...")
		time.Sleep(20 * time.Minute)
	}
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
