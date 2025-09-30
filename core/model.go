package core

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"

	"github.com/kshard/float8"
)

// LinearModel represents a simple linear model for predicting the next token.
type LinearModel struct {
	Weights      [][]float8.Float8
	LearningRate float8.Float8
	Tokenizer    *Tokenizer
}

// NewLinearModel creates and initializes a new LinearModel.
func NewLinearModel(tokenizer *Tokenizer, learningRate float32) *LinearModel {
	rand.Seed(time.Now().UnixNano())
	vocabSize := tokenizer.Count
	weights := make([][]float8.Float8, vocabSize)
	for i := range weights {
		weights[i] = make([]float8.Float8, vocabSize)
		for j := range weights[i] {
			weights[i][j] = float8.ToFloat8(rand.Float32() - 0.5) // Initialize with small random values
		}
	}

	return &LinearModel{
		Weights:      weights,
		LearningRate: float8.ToFloat8(learningRate),
		Tokenizer:    tokenizer,
	}
}

// softmax applies the softmax function to a slice of float8.Float8.
func softmax(input []float8.Float8) []float8.Float8 {
	if len(input) == 0 {
		return []float8.Float8{}
	}
	maxVal := input[0]
	for _, v := range input {
		if float8.ToFloat32(v) > float8.ToFloat32(maxVal) {
			maxVal = v
		}
	}

	sum := float8.ToFloat8(0.0)
	output := make([]float8.Float8, len(input))
	for i, v := range input {
		f64val := float64(float8.ToFloat32(v) - float8.ToFloat32(maxVal))
		output[i] = float8.ToFloat8(float32(math.Exp(f64val)))
		sum = float8.Add(sum, output[i])
	}

	for i := range output {
		output[i] = float8.Div(output[i], sum)
	}
	return output
}

// Predict predicts the next token index given the current token index,
// applying rewards for co-occurrence and penalties for repetition.
func (m *LinearModel) Predict(currentTokenIndex int, generatedTokens []int) int {
	if currentTokenIndex < 0 || currentTokenIndex >= len(m.Weights) {
		return rand.Intn(m.Tokenizer.Count) // Out of bounds safety
	}

	scores := m.Weights[currentTokenIndex]
	probabilities := softmax(scores)

	// 1. Apply co-occurrence reward
	if freqMap, ok := m.Tokenizer.UnigramFreq[currentTokenIndex]; ok {
		totalFreq := 0
		for _, freq := range freqMap {
			totalFreq += freq
		}

		if totalFreq > 0 {
			for nextIdx, freq := range freqMap {
				reward := float8.Div(float8.ToFloat8(float32(totalFreq)), float8.ToFloat8(float32(freq)))
				//one_plus_reward := float8.Add(float8.ToFloat8(1.0), reward)
				probabilities[nextIdx] = float8.Mul(probabilities[nextIdx], reward)
			}
		}
	}

	// 2. Apply repetition penalty
	repetitionPenalty := float8.ToFloat8(1.5)
	for _, tokenIndex := range generatedTokens {
		if tokenIndex >= 0 && tokenIndex < len(probabilities) {
			probabilities[tokenIndex] = float8.Div(probabilities[tokenIndex], repetitionPenalty)
		}
	}

	// 3. Re-normalize probabilities
	var sum float8.Float8 = float8.ToFloat8(0.0)
	for _, p := range probabilities {
		sum = float8.Add(sum, p)
	}
	if float8.ToFloat32(sum) > 0 {
		for i := range probabilities {
			probabilities[i] = float8.Div(probabilities[i], sum)
		}
	}

	// 4. Select the token with the highest probability
	maxProb := float8.ToFloat8(-1.0)
	predictedIndex := 0
	for i, p := range probabilities {
		if float8.ToFloat32(p) > float8.ToFloat32(maxProb) {
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
		fmt.Print("Epoch:", epoch, " ")
		gradients := make([][]float8.Float8, vocabSize)
		for i := range gradients {
			gradients[i] = make([]float8.Float8, vocabSize)
		}

		// Full-batch: calculate gradients for all data points from the UnigramMap
		for currentTokenIndex, nextTokenIndex := range m.Tokenizer.UnigramMap {

			scores := m.Weights[currentTokenIndex]
			probabilities := softmax(scores)

			// Calculate gradient for the scores (y_pred - y_true)
			dScores := make([]float8.Float8, vocabSize)
			copy(dScores, probabilities)
			dScores[nextTokenIndex] = float8.Sub(dScores[nextTokenIndex], float8.ToFloat8(1.0))

			// Add to gradients for the current input token's weights
			for j := 0; j < vocabSize; j++ {
				gradients[currentTokenIndex][j] = float8.Add(gradients[currentTokenIndex][j], dScores[j])
			}
		}

		// Update weights using the accumulated gradients
		for i := 0; i < vocabSize; i++ {
			for j := 0; j < vocabSize; j++ {
				update := float8.Mul(m.LearningRate, gradients[i][j])
				m.Weights[i][j] = float8.Sub(m.Weights[i][j], update)
			}
		}
		fmt.Println("Done.")
	}
}
