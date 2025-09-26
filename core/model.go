package core

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"
)

// LinearModel represents a simple linear model for predicting the next token.
type LinearModel struct {
	Weights      [][]float32
	LearningRate float32
	Tokenizer    *Tokenizer
}

// NewLinearModel creates and initializes a new LinearModel.
func NewLinearModel(tokenizer *Tokenizer, learningRate float32) *LinearModel {
	rand.Seed(time.Now().UnixNano())
	vocabSize := tokenizer.Count
	weights := make([][]float32, vocabSize)
	for i := range weights {
		weights[i] = make([]float32, vocabSize)
		for j := range weights[i] {
			weights[i][j] = rand.Float32() - 0.5 // Initialize with small random values
		}
	}

	return &LinearModel{
		Weights:      weights,
		LearningRate: learningRate,
		Tokenizer:    tokenizer,
	}
}

// softmax applies the softmax function to a slice of float32.
func softmax(input []float32) []float32 {
	if len(input) == 0 {
		return []float32{}
	}
	maxVal := input[0]
	for _, v := range input {
		if v > maxVal {
			maxVal = v
		}
	}

	sum := float32(0.0)
	output := make([]float32, len(input))
	for i, v := range input {
		output[i] = float32(math.Exp(float64(v - maxVal)))
		sum += output[i]
	}

	for i := range output {
		output[i] /= sum
	}
	return output
}

// Predict predicts the next token index given the current token index.
func (m *LinearModel) Predict(currentTokenIndex int) int {
	if currentTokenIndex < 0 || currentTokenIndex >= len(m.Weights) {
		// Return a random token if the input is out of bounds
		return rand.Intn(m.Tokenizer.Count)
	}

	scores := m.Weights[currentTokenIndex]
	probabilities := softmax(scores)

	maxProb := float32(-1.0)
	predictedIndex := 0
	for i, p := range probabilities {
		if p > maxProb {
			maxProb = p
			predictedIndex = i
		}
	}
	return predictedIndex
}

// Train trains the model using full-batch gradient descent based on the Tokenizer's UnigramMap.
func (m *LinearModel) Train(epochs int) {

	vocabSize := m.Tokenizer.Count
	if vocabSize == 0 {
		return // Cannot train on an empty vocabulary
	}

	for epoch := 0; epoch < epochs; epoch++ {
		runtime.GC()
		fmt.Println("Epoch:", epoch)
		gradients := make([][]float32, vocabSize)
		for i := range gradients {
			gradients[i] = make([]float32, vocabSize)
		}

		// Full-batch: calculate gradients for all data points from the UnigramMap
		for currentTokenIndex, nextTokenIndex := range m.Tokenizer.UnigramMap {

			scores := m.Weights[currentTokenIndex]
			probabilities := softmax(scores)

			// Calculate gradient for the scores (y_pred - y_true)
			dScores := make([]float32, vocabSize)
			copy(dScores, probabilities)
			dScores[nextTokenIndex] -= 1

			// Add to gradients for the current input token's weights
			for j := 0; j < vocabSize; j++ {
				gradients[currentTokenIndex][j] += dScores[j]
			}
		}

		// Update weights using the accumulated gradients
		for i := 0; i < vocabSize; i++ {
			for j := 0; j < vocabSize; j++ {
				m.Weights[i][j] -= m.LearningRate * gradients[i][j]
			}
		}
	}
}
