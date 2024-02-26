import json
import string
import nltk
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import GloVe
# nltk.download('punkt')
from sklearn.metrics import accuracy_score


class IntentModelArchitecture(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings=None):
        super(IntentModelArchitecture, self).__init__()

        # embedding layer is first layer since need to covert text to numerical format, reduce dimensionality of input data, and find semantics
        if pretrained_embeddings is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(pretrained_embeddings)
        else:
            self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer is a hidden layer that allows model to learn long-term dependencies in sequential data and mitigates vanishing gradient problem
        self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Linear layer
        self.linear_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # convert input text to dense word embeddings
        embedded = self.embedding_layer(text)
        # feed word embeddings into LSTM and return output for each timestep and ignored final hidden/cell states
        lstm_out, _ = self.lstm_layer(embedded)
        # take all elements from 1D, take last element in 2D, and take all elements from 3D
        lstm_out = lstm_out[:, -1, :]
        # final linear layer that acts as classifier
        output = self.linear_layer(lstm_out)
        return output


class IntentModelTrainer:
    def __init__(self, model, criterion, optimizer, device='cpu'):
        self.model = model.to(device)  # RNN model to be trained, a RNNArchitecture object for intent classification
        self.criterion = criterion  # loss function to measure model's performance during training
        self.optimizer = optimizer  # optimization algorithm that updates gradients
        self.device = device  # device that trains the model

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


class IntentModelDataset(Dataset):
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
train_df = pd.concat([is_train, oos_train])
validate_df = pd.concat([is_val, oos_val])
test_df = pd.concat([is_test, oos_test])

# preprocess training data
vocab = set()
train_df['text'] = train_df['text'].apply(lambda x: x.lower())  # apply lowercase to all text inputs
# train_df['text'] = train_df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))  # remove punctuation
train_df['text'] = train_df['text'].apply(lambda x: nltk.tokenize.word_tokenize(x))  # tokenize text
# stop_words = set(nltk.corpus.stopwords.words('english')) # get set of stopwords
# train_df['text'] = train_df['text'].apply(lambda x: [word for word in x if word not in stop_words]) # remove stopwords
# stemmer = nltk.stem.PorterStemmer()
# train_df['text'] = train_df['text'].apply(lambda x: [stemmer.stem(word) for word in x])
# lematizer = nltk.stem.WordNetLemmatizer()
# train_df['text'] = train_df['text'].apply(lambda x: [lematizer.lemmatize(word) for word in x])

# add each token to the set
for tokens in train_df['text']:
    vocab.update(tokens)

# instantiate CustomDataset objects to be loaded
train_dataset = IntentModelDataset(train_df)
validate_dataset = IntentModelDataset(validate_df)
test_dataset = IntentModelDataset(test_df)

# instantiate DataLoader objects to shuffle data to reduce bias, speed training process with batch size, etc.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# define hyperparameters for IntentModelArchitecture object and instantiate pre-trained word embedding
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 500
output_dim = 151  # 150 in scope intent class and 1 oos intent class
glove = GloVe(name='6B', dim=100)

# create instance of IntentModelArchitecture
model = IntentModelArchitecture(vocab_size, embedding_dim, hidden_dim, output_dim, glove.vectors)

# define the loss function and optimization algo
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# instantiate IntentModelTrainer to start training the model

# Example Usage:
# Initialize your dataset and DataLoader, and load pre-trained embeddings (e.g., GloVe)
# Define hyperparameters
# Create an instance of TextClassifier
# Create an instance of TextClassificationModel with the TextClassifier, criterion, optimizer, and device
# Train and evaluate the model
