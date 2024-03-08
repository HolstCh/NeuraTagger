import json
import nltk
import string
from sklearn.preprocessing import LabelEncoder


class DataProcessor:
    """
    DataProcessor is used to process data for Natural Language Processing (NLP) tasks. Such tasks are performed on Pandas
    dataframes in-place. This class is currently being used to process training, testing, and validation data in Main.py
    """
    def __init__(self, df):
        """
        Constructs all the necessary attributes for the DataProcessor object.

        Args:
            df (Pandas DataFrame): data to be be processed for NLP tasks
        """
        # download the necessary NLTK resources
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        # initialize corresponding train, validate, or test dataframe (df variable name is convention when using Pandas)
        self.df = df
        # set of English stop words from NLTK
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        # lemmatizer for word lemmatization process
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        # encodes categorical labels
        self.label_encoder = LabelEncoder()
        # vocabulary set to store unique words
        self.vocab = set()

    def clean_text(self):
        """
        Converts all text inputs to lowercase and removes punctuation from each word of text inputs.

        No Args.

        Returns:
            None. The function operates on the DataFrame in-place.
        """
        # convert all text inputs lowercase to lowercase
        self.df['text'] = self.df['text'].apply(lambda x: x.lower())
        # remove punctuation from each word of text inputs
        self.df['text'] = self.df['text'].apply(
            lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    def tokenize_text(self):
        """
        Tokenizes words within each input text.

        No Args.

        Returns:
            None. The function operates on the DataFrame in-place.
        """
        self.df['text'] = self.df['text'].apply(
            lambda x: nltk.tokenize.word_tokenize(x))

    def remove_stopwords(self):
        """
        Removes stopwords from the text.

        No Args.

        Returns:
            None. The function operates on the DataFrame in-place.
        """
        self.df['text'] = self.df['text'].apply(
            lambda x: [word for word in x if word not in self.stop_words])

    def lemmatize_text(self):
        """
        Applies lemmatization to each word within the text.

        No Args.

        Returns:
            None. The function operates on the DataFrame in-place.
        """
        self.df['text'] = self.df['text'].apply(lambda x: [self.lemmatizer.lemmatize(word) for word in x])

    def encode_labels(self):
        """
        Fits the label encoder on intent labels (converts words to indices for later use with transform())

        No Args.

        Returns:
           None. The function operates on the DataFrame in-place.
        """
        self.label_encoder.fit(self.df['intent'])

    def process_data(self):
        """
        Applies all the above preprocessing steps on the data in order from top to bottom.

        No Args.

        Returns:
           None. The function operates on the DataFrame in-place.
        """
        self.clean_text()
        self.tokenize_text()
        self.remove_stopwords()
        self.lemmatize_text()
        self.encode_labels()

    def build_vocab(self):
        """
        Builds a vocabulary from the text and saves it to a JSON file.

        No Args.

        Returns:
           None. The function operates on the DataFrame in-place.
        """
        # update set with unique tokens
        for tokens in self.df['text']:
            self.vocab.update(tokens)
        # convert to list and sort the unique tokens
        self.vocab = sorted(list(self.vocab))
        # serialize vocab and write to JSON file
        with open('vocab.json', 'w') as vocab_file:
            json.dump(self.vocab, vocab_file)

    def save_vocab(self, filepath):
        """
        Saves the vocabulary to a JSON file at the specified filepath.

        Args:
            filepath (str): The path where the vocabulary will be saved.

        Returns:
            None. The function operates on the DataFrame in-place.
        """
        # serialize vocab and write to JSON file
        with open(filepath, 'w') as vocab_file:
            json.dump(self.vocab, vocab_file)

    def load_vocab(self, filepath):
        """
        Loads the vocabulary from a JSON file at the specified filepath.

        Args:
           filepath (str): The path from where the vocabulary will be loaded.

        Returns:
           None. The function operates on the DataFrame in-place.
        """
        # read JSON file, deserialize JSON file, and assign to vocab for use within program
        with open(filepath, 'r') as vocab_file:
            self.vocab = json.load(vocab_file)
