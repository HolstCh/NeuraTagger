import json
import string
import nltk
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

model_saved = False


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

        # dropout layer with p=0.5
        self.dropout = nn.Dropout(p=0.1)

        # linear layer
        self.linear_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # convert input text to dense word embeddings
        embedded = self.embedding_layer(text.long())
        # feed word embeddings into LSTM and return output for each timestep and ignored final hidden/cell states
        lstm_out, _ = self.lstm_layer(embedded)
        # take all elements from 1D, take last element in 2D, and take all elements from 3D
        lstm_out = lstm_out[:, -1, :]
        # apply dropout after LSTM layer
        lstm_out = self.dropout(lstm_out)
        # final linear layer that acts as classifier
        output = self.linear_layer(lstm_out)
        return output


class IntentModelTrainer:
    def __init__(self, model, criterion, optimizer, device='cpu'):
        # RNN model to be trained, a RNNArchitecture object for intent classification
        self.model = model.to(device)
        # loss function to measure model's performance during training
        self.criterion = criterion
        # optimization algorithm that updates gradients
        self.optimizer = optimizer
        # device that trains the model
        self.device = device

    def train(self, train_data_loader):
        print(len(train_data_loader))
        # set model to training mode (layers behave differently compared to validate mode)
        self.model.train()
        # keep track of total loss and correct predictions
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        # iterate through training data in batches (batches of features/labels)
        for texts, intents in train_data_loader:
            # move features/labels to CPU device
            texts, intents = texts.to(self.device), intents.to(self.device)
            # zero gradients before backward pass to refrain from accumulation each backward pass
            self.optimizer.zero_grad()
            # feed features into model to get labels
            outputs = self.model(texts)
            # compute the loss using the loss function (cross entropy)
            loss = self.criterion(outputs, intents.long())
            # performs backpropagation to compute gradients with respect to model's parameters
            loss.backward()
            # update step which updates parameters using gradients computed in backward pass
            self.optimizer.step()
            # adds loss value of the current pass to total loss value (converted to Python number)
            total_loss += loss.item()

            # calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == intents).sum().item()
            total_samples += intents.size(0)

        # calculate accuracy
        accuracy = correct_predictions / total_samples
        # return average loss and accuracy per batch
        return total_loss / len(train_data_loader), accuracy

    def validate(self, val_loader):
        # set the model to evaluation mode
        self.model.eval()
        # lists for preds and labels
        all_preds, all_labels = [], []
        # disable gradient computations during validation
        with torch.no_grad():
            # loop over batches of texts and intents from the validation loader
            for texts, intents in val_loader:
                # move features/labels to CPU device
                texts, intents = texts.to(self.device), intents.to(self.device)
                # feed features as forward pass in model to get predictions
                outputs = self.model(texts)
                # extract predicted classes using index of max value of dim=1
                _, preds = torch.max(outputs, 1)
                # extend lists with current batches preds and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(intents.cpu().numpy())
        # compute accuracy of model
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
    def __init__(self, dataframe, vocabulary):
        self.dataframe = dataframe
        self.vocabulary = list(vocabulary)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.dataframe['intent'])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i):
        text = self.dataframe.iloc[i]['text']
        intent = self.dataframe.iloc[i]['intent']
        # convert each word in the text to its corresponding index in vocab
        text_indices = [self.vocabulary.index(word) for word in text if word in self.vocabulary]
        # convert intent into numeric format
        intent = self.label_encoder.transform([intent])[0]

        return torch.tensor(text_indices), torch.tensor(intent)


# combine samples into a batch for same dimensions for training tensor
def padding_fn(batch):
    # pad sequences to have the same length
    text_indices, intents = zip(*batch)
    padded_text_indices = pad_sequence(text_indices, batch_first=True)

    return padded_text_indices, torch.tensor(intents)


class DataProcessor:
    def __init__(self, df):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.df = df
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def clean_text(self):
        self.df['text'] = self.df['text'].apply(lambda x: x.lower())  # apply lowercase to all text inputs
        self.df['text'] = self.df['text'].apply(
            lambda x: x.translate(str.maketrans('', '', string.punctuation)))  # remove punctuation

    def tokenize_text(self):
        self.df['text'] = self.df['text'].apply(lambda x: nltk.tokenize.word_tokenize(x)) # tokenize words in input text

    def remove_stopwords(self):
        self.df['text'] = self.df['text'].apply(lambda x: [word for word in x if word not in self.stop_words]) # remove stopwords

    def lemmatize_text(self):
        self.df['text'] = self.df['text'].apply(lambda x: [self.lemmatizer.lemmatize(word) for word in x])


# load json file to be parsed into different dataframes
with open('data_full.json', 'r') as fp:
    data_full = json.load(fp)

# combine all training, validation, and test datasets (using both oos and is prompts)
for key in data_full.keys():
    if key == "oos_val":
        oos_val = pd.DataFrame(data_full[key], columns=['text', 'intent'])
        oos_val = oos_val.sample(n=20)
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

# preprocess training, validation, and testing data
vocab = set()
processed_train = DataProcessor(train_df)
processed_validate = DataProcessor(validate_df)
processed_test = DataProcessor(test_df)

# clean text of each dataset
processed_train.clean_text()
processed_validate.clean_text()
processed_test.clean_text()

# tokenize text of each dataset
processed_train.tokenize_text()
processed_validate.tokenize_text()
processed_test.tokenize_text()

# remove stopwords from each dataset
processed_train.remove_stopwords()
processed_validate.remove_stopwords()
processed_test.remove_stopwords()

# lemmatize text for each dataset
processed_train.lemmatize_text()
processed_validate.lemmatize_text()
processed_test.lemmatize_text()

# add each token to the set
for tokens in processed_train.df['text']:
    vocab.update(tokens)

print(vocab)
# instantiate CustomDataset objects to be loaded
train_dataset = IntentModelDataset(processed_train.df, vocab)
validate_dataset = IntentModelDataset(processed_validate.df, vocab)
test_dataset = IntentModelDataset(processed_test.df, vocab)

# instantiate DataLoader objects to shuffle data to reduce bias, speed training process with batch size, padding, etc.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=padding_fn)
validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=True, collate_fn=padding_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=padding_fn)

# define hyperparameters for IntentModelArchitecture object and instantiate pre-trained word embedding
vocab_size = len(vocab)
embedding_dim = 300
hidden_dim = 500
output_dim = 151  # 150 in scope intent class and 1 oos intent class
glove = GloVe(name='6B', dim=embedding_dim)

# create instance of IntentModelArchitecture
model = IntentModelArchitecture(vocab_size, embedding_dim, hidden_dim, output_dim, glove.vectors)

# define the loss function and optimization algo
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# instantiate IntentModelTrainer to start training the model (7/10 epoch seems best so far with avg loss per batch at 0.049522,
# next run had 8/10 epoch with avg loss per batch at 0.063428 ... also tuned learning rate to 0.005 from 0.001)
if not model_saved:
    trainer = IntentModelTrainer(model, criterion, optimizer)
    epochs = 7
    for epoch in range(epochs):
        avg_loss, accuracy = trainer.train(train_loader)
        print(
            f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}, Training Set Accuracy: {accuracy}')  # need to improve avg_loss from 4.8148555311106020

    val_accuracy = trainer.validate(validate_loader)
    print(f'Validation Set Accuracy: {val_accuracy}')
    # save the trainer object to be loaded later
    torch.save(trainer, 'intent_model_trainer.pth')

if model_saved:
    # Load the saved trainer object
    trainer = torch.load('intent_model_trainer.pth')
    val_accuracy = trainer.validate(validate_loader)
    print(f'Validation Set Accuracy: {val_accuracy}')

# Example Usage:
# Initialize your dataset and DataLoader, and load pre-trained embeddings (e.g., GloVe)
# Define hyperparameters
# Create an instance of TextClassifier
# Create an instance of TextClassificationModel with the TextClassifier, criterion, optimizer, and device
# Train and evaluate the model
