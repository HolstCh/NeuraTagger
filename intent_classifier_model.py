import string
import nltk
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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
        # apply softmax activation function to convert raw scores to probabilities
        output_probs = F.softmax(output, dim=1)
        return output_probs


class IntentModelTrainer:
    def __init__(self, model, criterion, optimizer, label_encoder=None, vocab=None, device='cpu'):
        # RNN model to be trained, a RNNArchitecture object for intent classification
        self.model = model.to(device)
        # loss function to measure model's performance during training
        self.criterion = criterion
        # optimization algorithm that updates gradients
        self.optimizer = optimizer
        # device that trains the model
        self.device = device
        # used to decode labels
        self.label_encoder = label_encoder
        # vocab of indices is used for glove library (needs numerical indexing to find semantic relationships)
        self.vocab = vocab

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

        # compute accuracy, precision, recall, f1_score, and confusion matrix of model
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")
        cm = confusion_matrix(all_labels, all_preds)
        return accuracy, precision, recall, f1, cm

    def predict(self, text):
        self.model.eval()

        # clean, tokenize, remove stopwords, and lemmatize the input text
        processed_text = text.lower()
        processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))
        processed_text = nltk.tokenize.word_tokenize(processed_text)
        processed_text = [word for word in processed_text if word not in set(nltk.corpus.stopwords.words('english'))]
        processed_text = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in processed_text]

        # convert processed text into numerical format using the vocabulary
        text_indices = [self.vocab.index(word) for word in processed_text if word in self.vocab]
        text_tensor = torch.tensor(text_indices).to(self.device).unsqueeze(0)  # add batch dimension

        if not text_indices:
            # handle the case where no words from processed_text are in the vocabulary
            print("None of the words in processed_text are in the vocabulary.")
            # return a default prediction
            return "oos"

        # make prediction
        with torch.no_grad():
            outputs = self.model(text_tensor)
            _, preds = torch.max(outputs, 1)

        # decode numerical prediction to intent label
        predicted_intent = self.label_encoder.inverse_transform([preds.item()])[0]

        return predicted_intent


# IntentModelDataset implements a dunder method (__getitem__) for PyTorch DataLoader to iterate through each dataframe
# using indices similar to how an array operates; such functionality is used while training and evaluating
class IntentModelDataset(Dataset):
    def __init__(self, dataframe, vocabulary, label_encoder):
        self.dataframe = dataframe
        self.vocabulary = vocabulary
        self.label_encoder = label_encoder  # used to solely encode labels in each dataset

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i):
        text = self.dataframe.iloc[i]['text']
        intent = self.dataframe.iloc[i]['intent']

        # convert intent into numeric format using label_encoder
        intent = self.label_encoder.transform([intent])[0]

        # convert each word in the text to its corresponding index in vocab
        text_indices = [self.vocabulary.index(word) for word in text if word in self.vocabulary]

        return torch.tensor(text_indices), torch.tensor(intent)


# combine samples into a batch for same dimensions for training tensor
def padding_fn(batch):
    # pad sequences to have the same length
    text_indices, intents = zip(*batch)
    padded_text_indices = pad_sequence(text_indices, batch_first=True)

    return padded_text_indices, torch.tensor(intents)
