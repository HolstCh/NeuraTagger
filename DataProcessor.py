import json
import nltk
import string
from sklearn.preprocessing import LabelEncoder


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
