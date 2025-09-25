package core

import (
	"encoding/gob"
	"os"
)

// ModelData is a container for saving and loading the model and tokenizer.
type ModelData struct {
	Tokenizer *Tokenizer
	Weights   [][]float64
}

// CreateAndTrainModel creates a tokenizer and a model from texts, trains the model,
// and saves it to a file using gob binary format.
func CreateAndTrainModel(texts []string, learningRate float64, epochs int, savePath string) (*LinearModel, error) {
	// 1. Create and build tokenizer
	tokenizer := NewTokenizer()
	for _, text := range texts {
		tokenizer.AddtoModel(text)
	}

	// 2. Create and train model
	model := NewLinearModel(tokenizer, learningRate)
	model.Train(epochs)

	// 3. Prepare data for saving
	dataToSave := ModelData{
		Tokenizer: tokenizer,
		Weights:   model.Weights,
	}

	// 4. Save to a binary file using gob
	file, err := os.Create(savePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(dataToSave)
	if err != nil {
		return nil, err
	}

	return model, nil
}

// LoadModel loads a model and tokenizer from a gob binary file.
func LoadModel(loadPath string, learningRate float64) (*LinearModel, error) {
	// 1. Open binary file
	file, err := os.Open(loadPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// 2. Decode gob data
	var loadedData ModelData
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&loadedData)
	if err != nil {
		return nil, err
	}

	// 3. Create and populate model
	model := &LinearModel{
		Weights:      loadedData.Weights,
		Tokenizer:    loadedData.Tokenizer,
		LearningRate: learningRate, // learning rate can be set on load
	}

	return model, nil
}