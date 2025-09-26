package core

import (
	"encoding/gob"
	"fmt"
	"os"
)

// CreateAndTrainModel creates a tokenizer and a model from texts, trains the model,
// and saves it to a file using gob binary format in a memory-efficient way.
func CreateAndTrainModel(texts []string, learningRate float32, epochs int, savePath string) (*LinearModel, error) {
	// 1. Create and build tokenizer
	tokenizer := NewTokenizer()
	for _, text := range texts {
		tokenizer.AddtoModel(text)
	}

	// 2. Build the final UnigramMap from frequency counts
	fmt.Println("Building unigram map...")
	tokenizer.BuildUnigramMap()

	// 3. Create and train model
	model := NewLinearModel(tokenizer, learningRate)
	fmt.Println("Training model...")
	model.Train(epochs)

	// 4. Save to a binary file using gob, streaming the weights
	file, err := os.Create(savePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)

	// Encode Tokenizer
	if err := encoder.Encode(model.Tokenizer); err != nil {
		return nil, err
	}

	// Encode Weights row by row
	fmt.Println("Saving tokens...")
	vocabSize := model.Tokenizer.Count
	if err := encoder.Encode(vocabSize); err != nil {
		return nil, err
	}
	fmt.Println("Saving weights...")
	for i := 0; i < vocabSize; i++ {
		if i%1000 == 0 {
			fmt.Printf("%d / %d\n", i, vocabSize)
		}
		if err := encoder.Encode(model.Weights[i]); err != nil {
			return nil, err
		}
	}

	return model, nil
}

// LoadModel loads a model and tokenizer from a gob binary file,
// reading the weights in a streaming fashion.
func LoadModel(loadPath string, learningRate float32) (*LinearModel, error) {
	// 1. Open binary file
	file, err := os.Open(loadPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)

	// 2. Decode Tokenizer
	var tokenizer Tokenizer
	if err := decoder.Decode(&tokenizer); err != nil {
		return nil, err
	}

	// 3. Decode Weights row by row
	var vocabSize int
	if err := decoder.Decode(&vocabSize); err != nil {
		return nil, err
	}

	weights := make([][]float32, vocabSize)
	for i := 0; i < vocabSize; i++ {
		if err := decoder.Decode(&weights[i]); err != nil {
			return nil, err
		}
	}

	// 4. Create and populate model
	model := &LinearModel{
		Weights:      weights,
		Tokenizer:    &tokenizer,
		LearningRate: learningRate,
	}

	return model, nil
}
