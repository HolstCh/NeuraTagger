import torch
import torch.nn as nn
import json
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import GloVe
from sklearn.metrics import accuracy_score


class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings=None):
        super(IntentClassifier, self).__init__()

        # embedding layer is first layer since need to covert text to numerical format, reduce dimensionality of input data, and find semantics
        if pretrained_embeddings is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(pretrained_embeddings)
        else:
            self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer is a hidden layer that allows model to learn long-term dependencies in sequential data and mitigates vanishing gradient problem
        self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Fully connected layer
        self.linear_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding_layer(text)
        lstm_out, _ = self.lstm_layer(embedded)
        lstm_out = lstm_out[:, -1, :]
        output = self.linear_layer(lstm_out)
        return output


class IntentClassificationModel:
    def __init__(self, model, criterion, optimizer, device='cpu'):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy

    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            inputs = text.to(self.device)
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
        return preds.item()


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    # method to return the text and labels at the given index
    def __getitem__(self, idx):
        text = torch.tensor(self.dataframe.iloc[idx, :-1].values)
        label = torch.tensor(self.dataframe.iloc[idx, -1])
        return text, label


# load json file to be parsed into different dataframes
with open('data_full.json', 'r') as fp:
    data_full = json.load(fp)

# combine all training, validation, and test datasets (using both oos and is prompts)
for key in data_full.keys():
    if key == "oos_val":
        oos_val = pd.DataFrame(data_full[key], columns=['text', 'intent'])
    elif key == "val":
        is_val = pd.DataFrame(data_full[key], columns=['text', 'intent'])
    elif key == "train":
        is_train = pd.DataFrame(data_full[key], columns=['text', 'intent'])
    elif key == "oos_test":
        oos_test = pd.DataFrame(data_full[key], columns=['text', 'intent'])
    elif key == "test":
        is_test = pd.DataFrame(data_full[key], columns=['text', 'intent'])
    elif key == "oos_train":
        oos_train = pd.DataFrame(data_full[key], columns=['text', 'intent'])

# concatenate dataframes for each training, validation, and testing dataset
train = pd.concat([is_train, oos_train])
validate = pd.concat([is_val, oos_val])
test = pd.concat([is_test, oos_test])

# instantiate CustomDataset objects to be loaded
train_dataset = CustomDataset(train)
validate_dataset = CustomDataset(validate)
test_dataset = CustomDataset(test)

# instantiate DataLoader objects to shuffle data to reduce bias, speed training process with batch size, etc.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Example Usage:
# Initialize your dataset and DataLoader, and load pre-trained embeddings (e.g., GloVe)
# Define hyperparameters
# Create an instance of TextClassifier
# Create an instance of TextClassificationModel with the TextClassifier, criterion, optimizer, and device
# Train and evaluate the model