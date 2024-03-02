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

model_is_saved = True


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
        # compute accuracy of model
        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy

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


class DataProcessor:
    def __init__(self, df):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.df = df
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.label_encoder = LabelEncoder()  # used to encode labels
        self.vocab = set()

    def clean_text(self):
        self.df['text'] = self.df['text'].apply(lambda x: x.lower())  # apply lowercase to all text inputs
        self.df['text'] = self.df['text'].apply(
            lambda x: x.translate(str.maketrans('', '', string.punctuation)))  # remove punctuation

    def tokenize_text(self):
        self.df['text'] = self.df['text'].apply(
            lambda x: nltk.tokenize.word_tokenize(x))  # tokenize words in input text

    def remove_stopwords(self):
        self.df['text'] = self.df['text'].apply(
            lambda x: [word for word in x if word not in self.stop_words])  # remove stopwords

    def lemmatize_text(self):
        self.df['text'] = self.df['text'].apply(lambda x: [self.lemmatizer.lemmatize(word) for word in x])

    def encode_labels(self):
        self.label_encoder.fit(self.df['intent'])  # fit the label encoder on intent labels

    def process_data(self):
        self.clean_text()
        self.tokenize_text()
        self.remove_stopwords()
        self.lemmatize_text()
        self.encode_labels()

    def build_vocab(self):
        for tokens in self.df['text']:
            self.vocab.update(tokens)
        self.vocab = sorted(list(self.vocab))
        with open('vocab.json', 'w') as vocab_file:
            json.dump(self.vocab, vocab_file)

    def save_vocab(self, filepath):
        with open(filepath, 'w') as vocab_file:
            json.dump(self.vocab, vocab_file)

    def load_vocab(self, filepath):
        with open(filepath, 'r') as vocab_file:
            loaded_vocab = json.load(vocab_file)
        self.vocab = loaded_vocab


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
        oos_test = oos_test.sample(n=30)
    elif key == "test":
        is_test = pd.DataFrame(data_full[key], columns=['text', 'intent'])
    elif key == "oos_train":
        oos_train = pd.DataFrame(data_full[key], columns=['text', 'intent'])

# concatenate dataframes for each training, validation, and testing dataset
train_df = pd.concat([is_train, oos_train])
validate_df = pd.concat([is_val, oos_val])
test_df = pd.concat([is_test, oos_test])

# preprocess training, validation, and testing data
processed_train = DataProcessor(train_df)
processed_validate = DataProcessor(validate_df)
processed_test = DataProcessor(test_df)

# clean, tokenize, remove stopwords, and lemmatize text of each dataset. Also, encode labels of each dataset
processed_train.process_data()
processed_validate.process_data()
processed_test.process_data()

if not model_is_saved:
    # add each token to the vocabulary
    processed_train.build_vocab()
    # save vocabulary during training
    processed_train.save_vocab('vocab.json')

if model_is_saved:
    # load vocabulary before loading the model
    processed_train.load_vocab("vocab.json")

# instantiate CustomDataset objects to be loaded
label_encoder = LabelEncoder
train_dataset = IntentModelDataset(processed_train.df, processed_train.vocab, processed_train.label_encoder)
validate_dataset = IntentModelDataset(processed_validate.df, processed_train.vocab, processed_validate.label_encoder)
test_dataset = IntentModelDataset(processed_test.df, processed_train.vocab, processed_test.label_encoder)

# instantiate DataLoader objects to shuffle data to reduce bias, speed training process with batch size, padding, etc.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=padding_fn)
validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=True, collate_fn=padding_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=padding_fn)

# define hyperparameters for IntentModelArchitecture object and instantiate pre-trained word embedding
embedding_dim = 300
hidden_dim = 500
output_dim = 151  # 150 in scope intent class and 1 oos intent class
glove = GloVe(name='6B', dim=embedding_dim)
vocab_size = glove.vectors.shape[0]

# create instance of IntentModelArchitecture
model = IntentModelArchitecture(vocab_size, embedding_dim, hidden_dim, output_dim, glove.vectors)

# define the loss function and optimization algo
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# instantiate IntentModelTrainer to start training the model (7/10 epoch seems best so far with avg loss per batch at 0.049522,
# next run had 8/10 epoch with avg loss per batch at 0.063428 ... also tuned learning rate to 0.005 from 0.001)
if not model_is_saved:
    trainer = IntentModelTrainer(model, criterion, optimizer, label_encoder=processed_train.label_encoder,
                                 vocab=processed_train.vocab)
    epochs = 100  # set the maximum number of epochs
    early_stopping_patience = 10  # number of epochs to wait for improvement
    best_val_accuracy = 0.0
    epochs_since_last_improvement = 0

    for epoch in range(epochs):
        avg_loss, accuracy = trainer.train(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}, Training Set Accuracy: {accuracy}')

        val_accuracy = trainer.validate(validate_loader)
        print(f'Validation Set Accuracy: {val_accuracy}')

        # check for improvement in validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_since_last_improvement = 0
            # save the model state when there is an improvement
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'best_intent_model.pth')
        else:
            epochs_since_last_improvement += 1

        # early stopping condition
        if epochs_since_last_improvement >= early_stopping_patience:
            print(f'No improvement in validation accuracy for {early_stopping_patience} epochs. Stopping early.')
            break

    print("Training complete.")

if model_is_saved:
    loaded_checkpoint = torch.load('best_intent_model.pth')
    # instantiate the saved model
    loaded_model = IntentModelArchitecture(vocab_size, embedding_dim, hidden_dim, output_dim, glove.vectors)
    # load the state dictionary into the new model
    loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=0.005)
    loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    loaded_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    # instantiate model trainer object with loaded model state
    trainer = IntentModelTrainer(loaded_model, criterion, loaded_optimizer, label_encoder=processed_train.label_encoder,
                                 vocab=processed_train.vocab)
    # evaluate accuracy of loaded model on validation dataset
    val_accuracy = trainer.validate(validate_loader)
    print(f'Validation Set Accuracy: {val_accuracy}')
    print(trainer.predict("when was Kobe in the NBA?"))
    test_accuracy = trainer.validate(test_loader)
    print(f'Test Set Accuracy: {test_accuracy}')
