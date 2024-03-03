# NeuraTagger

## Overview

This project implements a PyTorch-powered Intent Classification model using a Long Short-Term Memory (LSTM) neural network architecture. The model is trained to predict the intent of user queries, distinguishing between different classes. The training process involves the use of pre-trained word embeddings (GloVe), and the model is evaluated on validation and test datasets.

## Prerequisites

Make sure you have the following dependencies installed:

- Python 3.7 or later
- PyTorch
- NumPy
- NLTK
- Pandas
- Scikit-learn

Install the required libraries using:

```bash
pip install torch numpy nltk pandas scikit-learn
```

## Project Structure
- **intent_classifier_model**: Module that contains the IntentModelArchitecture, IntentModelTrainer, and the IntentModelDataset classes
- **IntentModelArchitecture.py**: Contains the implementation of the neural network structure
- **IntentModelTrainer.py**: Trains and validates data within DataLoaders (training, validation, and testing sets) and predicts single text inputs
- **IntentModelDataset.py**: Handles PyTorch DataLoaders to iterate through each dataframe for training/validation
- **DataProcessor.py**: Handles data cleaning, preprocessing, and vocabulary creation
- **main.py**: Entry point for program execution with option to train and save model, or load a pre-trained model (best_intent_model.pth)
- **best_intent_model.pth**: Pre-trained model checkpoint
- **vocab.json**: Vocabulary file

## Dataset
The dataset used for training, validation, and testing is loaded from the data_full.json file. It contains labeled examples for different intents.

## Instructions
1. Ensure _model_is_saved_ boolean within main.py is set to False to begin training the model
2. Set _model_is_saved_ to True to use the trained model and use IntentModelTrainer.predict() to predict the intent of text input or evaluate the trained model

