from collections import Counter
from nltk.util import ngrams
import numpy as np
import json

clean_files_dir = ""

def load_author_names():
    authors = []
    with open("author_names.txt", "r", encoding="utf8") as f:
        for line in f:            
            author = line.split("\t")[0]           
            name = author[:-12]
            authors.append(name)
    del authors[-3:]
    return authors


class EncodedData:
    def __init__(self, authors):
        self.authors = authors
        self.corpus = None

    def load_corpus(self, tokenized=False, verbose=False):
        if not self.corpus:
            if tokenized == True:
                filename = "author_corpus_tok.json"
            else:
                filename = "author_corpus.json"

            with open(filename, "r", encoding="utf8") as file:
                novel_data = json.load(file)
                self.corpus = novel_data
                if verbose:
                    print(f"Loaded {filename}")
        else:
            print("The corpus has already been loaded.")

    def create_segments(self, author_texts, n_segments, feature_type):
        """Divide the texts of a given author into segments of equal size. 
        Input is a dictionary with text names as keys and (tokenized) texts (in lists) as values."""
        list_of_texts = author_texts.values()
        if n_segments == None:
            segments = list(list_of_texts)
            return segments

        if feature_type in ["freq_250", "freq_140", "freq_120", "freq_22"]:
            all_text = [token for text in list_of_texts for token in text] # Merge all texts

            if len(all_text) % n_segments == 0:
                segments = np.array_split(all_text, n_segments) # A list of arrays
            else:
                segments = np.array_split(all_text, n_segments+1)
                segments = segments[:-1] # Remove last array with trailing tokens

            segments = [segment.tolist() for segment in segments]
            
        else:
            all_text = ""
            for text in list_of_texts:
                text += " "
                all_text += text
            segment_size = len(all_text) // n_segments 
            segments = [all_text[i:i+segment_size] for i in range(0, len(all_text), segment_size)]
            if len(segments) > n_segments:
                segments = segments[:-1] # Remove last segment with trailing tokens
        
        return segments 

    def token_freq_encode_segment(self, segment, feature_words):
        """Create vector of top n words for a segment, normalized by total number of tokens in segment.
        segment: a list of tokenized text. feature_words: a list of tokens to use as features."""
        total_freq = len(segment)
        x = np.zeros(len(feature_words))
        unique_word_counts = (Counter(segment))

        for pos, word in enumerate(feature_words):
            x[pos] = unique_word_counts[word] / total_freq

        return x

    def char_ngram_encode_segment(self, segment, n, profile_length, feature_ngrams):
        """Create a vector of length 'profile_length', consisting of character ngrams of size 'n', for a segment.
        segment: a string of text."""
        x = np.zeros(profile_length)
        characters = [c for c in segment]
        ngram_tuples = list(ngrams(characters, n))
        total_ngrams = len(ngram_tuples)
        ngram_counts = Counter(ngram_tuples)
        joined_ngram_counts = {"".join(key): value for key, value in ngram_counts.items()} 
             
        for pos, ngram in enumerate(feature_ngrams[:profile_length]):
            try:
                x[pos] = joined_ngram_counts[ngram] / total_ngrams
            except KeyError:
                x[pos] = 0
        return x

def load_features(feature_type, ngram=None):
    """Load a file with feature words into a list. Available feature types are 'freq_250', 
    freq_140', 'freq_120', 'freq_22', 'ngrams'"""
    features = []
    if feature_type == "freq_250":
        filename = "top_250_tokens.txt"
    elif feature_type == "freq_120":
        filename = "top_120_tokens.txt"
    elif feature_type == "freq_140":
        filename = "top_140_tokens.txt"
    elif feature_type == "freq_22":
        filename = "top_22_tokens.txt"
        
    elif feature_type == "ngrams":
        if ngram == 4:
            filename = "top_4_grams.txt"
        elif ngram == 5:
            filename = "top_5_grams.txt"
        elif ngram == 6:
            filename = "top_6_grams.txt"
        elif ngram == 7:
            filename = "top_7_grams.txt"
        else:
            print("There is no file with the specified ngram size.")
            return
    else:
        print(f"{feature_type} is not a valid feature type.")
        return

    with open(filename, "r", encoding="utf8") as file:
        for line in file:
            if ngram == None:
                features.append(line.strip())
            else: 
                line = line.replace("\n", "")
                features.append(line)

    return features


def load_and_encode_data(feature_type, n_segments=9, profile_size=500, ngram_size=None, verbose=True):
    """feature_type: 'freq_250', 'freq_140', freq_120', 'freq_22', 'ngrams'.
    ngram_size: None if encoding is not ngrams, else within range of 4-7."""
    features = load_features(feature_type, ngram_size)
    if not features:
        return None

    all_authors = load_author_names()
    if verbose:
        print("Loading cleaned novels...")
    data = EncodedData(all_authors)

    if feature_type in ["freq_250", "freq_140", "freq_120", "freq_22"]:
        data.load_corpus(tokenized=True)
    else:
        data.load_corpus()
    if verbose:
        print("Encoding features...")

    authors_segments = {}

    for author in data.corpus:
        segments = data.create_segments(data.corpus[author], n_segments, feature_type)

        encoded_segments = []

        if feature_type == "ngrams":
            for segment in segments:
                encoded = data.char_ngram_encode_segment(segment, ngram_size, profile_size, features)
                encoded_segments.append(encoded)

            authors_segments[author] = encoded_segments

        elif feature_type in ["freq_250", "freq_140", "freq_120", "freq_22"]:
            for segment in segments:
                encoded = data.token_freq_encode_segment(segment, features)
                encoded_segments.append(encoded)

            authors_segments[author] = encoded_segments

    X_raw = []
    authors = []
    genders = []

    for author in authors_segments:
        items = author.split("@@")
        for segment in authors_segments[author]:
            X_raw.append(segment)
            authors.append(items[0])
            genders.append(items[1])

    X = np.vstack(X_raw)
    if verbose:
        print(f"Loaded {len(authors)} segments encoded with {len(X[0])} features.")
    if n_segments == None:
        author_text_dict = {}
        for author_name, text_d in data.corpus.items():
            author_text_dict[author_name[:-3]] = list(text_d.keys())
        
        text_names = list(author_text_dict.values())
        text_names = [name for name_list in text_names for name in name_list]

        return X, authors, genders, text_names
    else:
        return X, authors, genders