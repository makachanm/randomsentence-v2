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

		// Pick an initial bigram token.
		initialBigram := pick(model.Tokenizer.Count, *model.Tokenizer)
		for !strings.Contains(initialBigram, " ") { // Ensure it's a bigram
			initialBigram = pick(model.Tokenizer.Count, *model.Tokenizer)
		}

		initialIndex := model.Tokenizer.Tokens[initialBigram]
		generatedIndices = append(generatedIndices, initialIndex)
		content.WriteString(initialBigram)
		content.WriteString(" ")

		// Continue generating the sentence from the initial bigram.
		currentToken := initialBigram
		for i := 0; i < 30; i++ { // Generate up to 30 more tokens
			currentIndex, exists := model.Tokenizer.GetTokenIndex(currentToken)
			if !exists {
				break // Stop if the current token is not in the dictionary
			}

			predictedIndex := model.Predict(currentIndex, generatedIndices)
			predictedToken := model.Tokenizer.GetToken(predictedIndex)

			if predictedToken == core.ENDTOKEN || !strings.Contains(predictedToken, " ") {
				break // Stop at end token or if not a valid bigram
			}

			generatedIndices = append(generatedIndices, predictedIndex)
			// Add only the second word of the predicted bigram to the content.
			parts := strings.Split(predictedToken, " ")
			if len(parts) > 1 {
				content.WriteString(parts[1])
				content.WriteString(" ")
			}

			currentToken = predictedToken
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
