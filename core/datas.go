package core

import (
	"encoding/gob"
	"fmt"
	"math/rand"
	"os"

	"github.com/x448/float16"
)

// CreateAndTrainModel creates a tokenizer and a model from texts, trains the model,
// and saves it to a file using gob binary format in a memory-efficient way.
func CreateAndTrainModel(texts []string, learningRate float32, epochs int, savePath string) (*LinearModel, error) {
	// Determine the split point for training data
	rand.Shuffle(len(texts), func(i, j int) {
		texts[i], texts[j] = texts[j], texts[i]
	})

	fmt.Printf("Using %d for training...\n", len(texts)/2)
	texts = texts[:len(texts)/2]

	// 1. Create and build tokenizer using only training data
	tokenizer := NewTokenizer()
	for _, text := range texts {
		tokenizer.AddtoModel(text)
	}

	// 2. Build the final UnigramMap from frequency counts
	fmt.Println("Building unigram map...")
	tokenizer.BuildUnigramMap()

	// 3. Create and train model with float32
	model := NewLinearModel(tokenizer, learningRate)
	fmt.Println("Training model...")
	model.Train(epochs, 32) // Using a batch size of 32

	// 4. Save to a binary file using gob, converting weights to float16 for storage
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

	// Encode Weights row by row after converting to float16
	fmt.Println("Saving tokens...")
	vocabSize := model.Tokenizer.Count
	if err := encoder.Encode(vocabSize); err != nil {
		return nil, err
	}

	fmt.Println("Saving weights...")
	weightsF16 := make([][]float16.Float16, vocabSize)
	for i := 0; i < vocabSize; i++ {
		weightsF16[i] = make([]float16.Float16, vocabSize)
		for j := 0; j < vocabSize; j++ {
			weightsF16[i][j] = float16.Fromfloat32(model.Weights[i][j])
		}
		if i%1000 == 0 {
			fmt.Printf("%d / %d\n", i, vocabSize)
		}
		if err := encoder.Encode(weightsF16[i]); err != nil {
			return nil, err
		}
	}

	fmt.Printf("Model created successfully from a total of %d sentences.", len(texts))

	return model, nil
}

// LoadModel loads a model and tokenizer from a gob binary file,
// converting float16 weights to float32 for use in the model.
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

	// 3. Decode Weights row by row (as float16)
	var vocabSize int
	if err := decoder.Decode(&vocabSize); err != nil {
		return nil, err
	}

	weightsF32 := make([][]float32, vocabSize)
	weightsF16Row := make([]float16.Float16, vocabSize)
	for i := 0; i < vocabSize; i++ {
		if err := decoder.Decode(&weightsF16Row); err != nil {
			return nil, err
		}
		weightsF32[i] = make([]float32, vocabSize)
		for j := 0; j < vocabSize; j++ {
			weightsF32[i][j] = weightsF16Row[j].Float32()
		}
	}

	// 4. Create and populate model with float32 weights
	model := &LinearModel{
		Weights:      weightsF32,
		Tokenizer:    &tokenizer,
		LearningRate: learningRate,
	}

	return model, nil
}
