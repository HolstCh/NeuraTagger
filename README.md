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

- **IntentModelArchitecture.py**: Contains the implementation of the neural network.
- **DataProcessor.py**: Handles data cleaning, preprocessing, and vocabulary creation.
- **IntentModelTrainer.py**: Trains, validates, and predicts text inputs
- **best_intent_model.pth**: Pre-trained model checkpoint.
- **vocab.json**: Vocabulary file.

## Dataset
The dataset used for training, validation, and testing is loaded from the data_full.json file. It contains labeled examples for different intents.

