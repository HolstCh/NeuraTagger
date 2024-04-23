"""
Filename: DataProcessor.py
Author: Chad Holst
Date: March 10, 2024

Description:
    The DataProcessor class is designed for preprocessing text data in Natural Language Processing (NLP) tasks. It includes
    methods for cleaning and transforming text data, such as converting to lowercase, removing punctuation and stopwords,
    tokenizing, and label encoding. The processed data is intended for use in supervised training Machine Learning (ML) models.

    The input DataFrame is expected to have two columns, each containing string values: one for input text and another for
    output labels.

    This class works as a parent class as it follows SOLID principles. Each child class may be extended for different
    processing pipelines; creating new process_data() functions, adding additional input data cols, and adding different processing
    functions (i.e., stemming). Such children/pipelines can be evaluated to find the best NLP pipeline for the ML model.

Directions:
    1. Create an instance of the DataProcessor class with a Pandas DataFrame, input text column name, and output label column name
    (additional parameters can be added for handling more columns in the future)
    2. Call the default process_data() method to apply text preprocessing steps.
    (additional processing methods and order of processing methods can also be added to experiment with NLP/ML results)
    3. Use the processed data for training or evaluating NLP models.
"""

import json
import nltk
import string
from sklearn.preprocessing import LabelEncoder


class DataProcessor:
    def __init__(self, df, text_col, label_col):
        """
        Constructs all the necessary attributes for the DataProcessor object.

        Args:
            df (Pandas DataFrame): df to be processed for NLP tasks containing two columns with str values.
            text_col (str): The name of the column containing the input text data.
            label_col (str): The name of the column containing the output label data.
        """
        # download the necessary NLTK resources
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        # initialize corresponding train, validate, or test dataframe (df variable name is convention when using Pandas)
        self.df = df
        # initialize input text column name for Pandas DataFrame
        self.text_col = text_col
        # initialize output column name for Pandas DataFrame
        self.label_col = label_col
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
        Converts each text input (str) in text column to lowercase and removes punctuation from each word of text input.

        No Args.

        Returns:
            None. The function operates on the DataFrame in-place.
        """
        # convert all text inputs to lowercase
        self.df[self.text_col] = self.df[self.text_col].apply(lambda x: x.lower())
        # remove punctuation from each word of text inputs
        self.df[self.text_col] = self.df[self.text_col].apply(
            lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    def tokenize_text(self):
        """
        Tokenizes words within each input text (str) in text column.

        No Args.

        Returns:
            None. The function operates on the DataFrame in-place.
        """
        self.df[self.text_col] = self.df[self.text_col].apply(
            lambda x: nltk.tokenize.word_tokenize(x))

    def remove_stopwords(self):
        """
        Removes stopwords from each input text (List[str]) in text column.

        No Args.

        Returns:
            None. The function operates on the DataFrame in-place.
        """
        self.df[self.text_col] = self.df[self.text_col].apply(
            lambda x: [word for word in x if word not in self.stop_words])

    def lemmatize_text(self):
        """
        Applies lemmatization to each word within each input text (List[str]) in text column.

        No Args.

        Returns:
            None. The function operates on the DataFrame in-place.
        """
        self.df[self.text_col] = self.df[self.text_col].apply(lambda x: [self.lemmatizer.lemmatize(word) for word in x])

    def encode_labels(self):
        """
        Fits the label encoder on each label in label column (str-index conversion for use with transform() or inverse_transform())

        No Args.

        Returns:
           None. The function operates on the DataFrame in-place.
        """
        self.label_encoder.fit(self.df[self.label_col])

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
           None. The function operates on the DataFrame and vocabulary set in-place.
        """
        # update set with unique tokens
        for tokens in self.df[self.text_col]:
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
            None. The function operates on the vocab set in-place.
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
           None. The function operates on the vocab set in-place.
        """
        # read JSON file, deserialize JSON file, and assign to vocab for use within program
        with open(filepath, 'r') as vocab_file:
            self.vocab = json.load(vocab_file)
