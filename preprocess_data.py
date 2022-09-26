import os
import re
from bs4 import BeautifulSoup as bs
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import json

#TODO: Maybe show list of word use for each author? 
#TODO: Show a graph of the word use for a given author. Like the example in the notes document. 
#TODO: See which author is the most different from others. See if this is stringbergs translated texts. 

raw_dir = "C:/Users/marti/Documents/UU/Research and development/Project/data/Litteraturbanken/files"
clean_files_dir = "C:/Users/marti/Documents/UU/Research and development/Project/data/Litteraturbanken/cleaned_texts"
clean_files_dir_1 = "C:/Users/marti/Documents/UU/Research and development/Project/data/Litteraturbanken/cleaned_texts_1"

def get_files(directory):
    """Get a list of file paths for a given directory"""
    file_paths = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        if os.path.isfile(f) and f.endswith("xml") or f.endswith("txt"):
            file_paths.append(f)
    return file_paths


def get_author_names(directory, out_file):
    """Creates a file listing all authors (and their years of living) in the raw data."""
    novel_files = get_files(directory)
    names = []

    for file_name in novel_files:
        with open(file_name, "r", encoding="utf8") as file:
            content = file.readlines()
            content = "".join(content)
            bs_content = bs(content, "lxml")
            author = str(bs_content.find("author").string)
            if author not in names:
                names.append(author)

    # Save to file
    with open(out_file, "w", encoding = "utf8") as f:
        for name in names:
            f.write(f"{name}\n")


def create_info(corpus_obj, outfile):
    """Create a tab separated text file listing all authors, their number of works and number of
    tokens. Input is a corpus class object. An existing tokenized json corpus is required"""
    author_info = {}
    for author in corpus_obj.corpus:
        n_texts = len(corpus_obj.corpus[author])
        total_tokens = 0
        gender = author[-1]

        for text in corpus_obj.corpus[author]:
            token_counts = len(corpus_obj.corpus[author][text])
            total_tokens += token_counts

        author_info[author[:-3]] = [n_texts, total_tokens, gender]

    # Save to file
    with open(outfile, "w", encoding = "utf8") as f:
        f.write("{:<30}{:<10}{:<10}{:<10}\n".format("Author", "Works", "Tokens", "Gender"))
        for name in author_info:
            f.write("{:<30}{:<10}{:<10}{:<10}\n".format(name, author_info[name][0], author_info[name][1], author_info[name][2]))
        f.write("\n")
        f.write(f"Number of authors: {len(author_info)}\n")
        f.write(f"Number of works: {sum([v[0] for v in author_info.values()])}")


def create_clean_files(directory):
    """Preprocess all xml file novels and saves them to new text files."""
    novel_files = get_files(directory)
    print(len(novel_files))
    for i, file in enumerate(novel_files):
        print(i)
        print(f"Processing {file}")
        novel = Novel(file)
        novel.prepare_data()


def load_author_names(filename):
    """Get a list of author names and genders from an existing text file."""
    authors = []
    with open(filename, "r", encoding="utf8") as f:
        for line in f:        
            author = line.split("\t")[0]
            gender = line.split("\t")[1]         
            name = author[:-12]
            name_gender = name + "@@" + gender.strip()
            authors.append(name_gender)
    #del authors[-3:]
    return authors


class Novel:
    """Class for preprocessing and saving a novel from an xml file."""
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf8") as file:
            content = file.readlines()
            content = "".join(content)
            bs_content = bs(content, "lxml")

            self.raw_text = (bs_content.find("body"))
            self.novel_text = self.clean_text(self.raw_text)
            self.author = str(bs_content.find("author").string)
            self.title = str(bs_content.find("title").string)

    def clean_text(self, raw_text):
        """Removes unwanted characters."""
        cleaned = str(raw_text).replace("<lb></lb>", "")
        cleaned = str(cleaned).replace('»', '').replace("«", "")
        cleaned = re.sub(r'<.*>', '', cleaned)
        cleaned = cleaned.replace("\n", " ")
        cleaned = re.sub('\s\s+', ' ', cleaned)
        cleaned = cleaned.replace(u'\u00AD', "@")
        cleaned = re.sub(r'@\W', "", cleaned)        
        cleaned = re.sub('\s\s­+', '', cleaned)
        return cleaned

    def tokenize_and_linebreak(self, text):
        cleaned = text.replace(":", " :").replace(";", " ;").replace(",", " ,").replace("(", "( ").replace(")", " )")
        cleaned = re.sub(r"'(?=\S)", r"' ", cleaned)
        cleaned = re.sub(r"(?<=\S)'", r" '", cleaned) 
        cleaned = re.sub(r"”(?=\S)", r"” ", cleaned)
        cleaned = re.sub(r"(?<=\S)”", r" ”", cleaned)
        cleaned = re.sub(r"\? (?=[a-z])", " ? ", cleaned)
        cleaned = re.sub(r'(\w)(\.+)', r'\1 \2\n', cleaned)
        cleaned = re.sub(r'(\w)(!+)', r'\1 \2\n', cleaned)
        cleaned = re.sub(r"\? (?![a-z])", " ?\n", cleaned)
        cleaned = cleaned.replace("\n ", "\n")
               
        return cleaned

    def write_to_file(self, text, name, filepath):
        complete_name = os.path.join(filepath, name)
        with open(complete_name, "w", encoding="utf8") as file:
            file.write(text)

    def prepare_data(self):
        """Creates a filename and saves the novel to a file."""
        author_name = self.author[:-12].split()
        author_name = "_".join(author_name)
        title_name = self.title.split()
        title_name = "_".join(title_name)
        title_name = re.sub("[.,:?*»«]", "", title_name)
        title_name = title_name[:100]

        # Write the cleaned text of the novel to a file, all on one line without tokenization. 
        clean_file = author_name + "_" + title_name + "_clean_1.txt"
        self.write_to_file(self.novel_text, clean_file, clean_files_dir_1)


class Corpus:
    """Class for creating and managing a corpus with authors and their works."""
    def __init__(self, authors):
        self.authors = authors
        self.corpus = None
        #self.author_genders = None

    def create_authors_corpus(self, data_dir, tokenize=False):
        """Creates a nested dictionary with author name, names of each text for the author, and the texts in themselves. 
        If tokenize = True, it has the format {'Author 1': {'name of text 1' : ['token1', token2'], 'name of text 2': ['token1', token2']}, 'Author 2': ...}
        If tokenize = False, it has the format {'Author 1': {'name of text 1' : 'words in string 1', 'name of text 2': 'words in string 2'}, 'Author 2': ...}
        """
        corpus = {}
        files = get_files(data_dir)

        for path in files:
            file_name = path.split("_")
            file_name = " ".join(file_name)
            file_name = re.sub("C:.*\\\\", "", file_name)

            with open(path, "r", encoding="utf8") as f:
                text = f.readlines()[0]
                text = text.lower()

            if tokenize == True:
                text = word_tokenize(text, language="swedish")

            for i, author_name in enumerate(self.authors):
                if author_name[:-3] in file_name:
                    text_name = re.sub(author_name[:-3], "", file_name)
                    text_name = re.sub("clean", "", text_name)
                    text_name = re.sub(" .txt", "", text_name)
                    text_name = text_name.strip()

                    if author_name in corpus:
                        corpus[self.authors[i]][text_name] = text
                    else:
                        corpus[self.authors[i]] = {text_name : text}
        self.corpus = corpus

        if tokenize == True:
            filename = "author_corpus_tok.json"
        else:
            filename = "author_corpus.json"

        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.corpus, file)

        return self.corpus

    def load_corpus(self, tokenized=False):
        if tokenized == True:
            filename = "author_corpus_tok.json"
        else:
            filename = "author_corpus.json"

        with open(filename, "r", encoding="utf8") as file:
            data = json.load(file)
            self.corpus = data
            print(f"Loaded {filename}")


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

    with open(filename, "r", encoding="utf8") as file:
        for line in file:
            if ngram == None:
                features.append(line.strip())
            else: 
                line = line.replace("\n", "")
                features.append(line)

    return features


def check_features(feature_type, corpus, ngram=None, tokenize=True):
    """Counts how many of the features in the specified feature set that do not occur in all the texts"""
    not_in_all = []
    features = load_features(feature_type, ngram)

    for author in corpus:
        list_of_texts = corpus[author].values()

        for text in list_of_texts:
            if tokenize == False:
                    characters = [c for c in text]
                    ngram_tuples = list(ngrams(characters, ngram))
                    joined_ngrams = ["".join(t) for t in ngram_tuples] 
                    novel = joined_ngrams
            else:
                novel = text

            for feature in features:
                feature = feature.strip("\n")
                if feature not in novel:
                    not_in_all.append(feature)

    result = Counter(not_in_all)
    return result


def get_top_n_words(files_dir, n):
    """Get the top n most frequent words in all files from a directory, and write the words to a file."""
    all_counts = Counter()
    files = get_files(files_dir)

    for file in files:
        with open(file, "r", encoding="utf8") as f:
            text = f.readlines()[0]
            text = word_tokenize(text.lower(), language="swedish")
        token_counts = Counter(text)
        all_counts += token_counts
    top_n = all_counts.most_common(350)
    top_n = [tup[0] for tup in top_n]

    filename = "top_" + str(n) + "_tokens.txt"
    with open(filename, "w", encoding="utf8") as file:
        for token in top_n:
            file.write(token)
            file.write("\n")
    print(f"Created {filename}")

def get_top_ngrams(files_dir, n, length):
    """Get the {length} most frequent ngrams in all files from a directory, and write the words to a file.
    n: ngram size"""
    all_counts = Counter()
    files = get_files(files_dir)
    for file in files:
        with open(file, "r", encoding="utf8") as f:
            text = f.readlines()[0]
            text = text.lower()

        characters = [c for c in text]
        ngram_tuples = ngrams(characters, n)
        ngram_counts = Counter(ngram_tuples)
        all_counts += ngram_counts

    top_ngrams = all_counts.most_common(length)
    top_ngrams = [tup[0] for tup in top_ngrams]
    ngram_list = ["".join(t) for t in top_ngrams]

    filename = "top_" + str(n) + "_grams" + ".txt"
    with open(filename, "w", encoding="utf8") as file:
        for token in ngram_list:
            file.write(token)
            file.write("\n")
    print(f"Created {filename}")


def create_feature_files():
    """Create all the feature files to use as style representations"""
    get_top_ngrams(clean_files_dir, 4, 2500)
    get_top_ngrams(clean_files_dir, 5, 2500)
    get_top_ngrams(clean_files_dir, 6, 2500)
    get_top_ngrams(clean_files_dir, 7, 2500)
    get_top_n_words(clean_files_dir, 22)
    get_top_n_words(clean_files_dir, 120)
    get_top_n_words(clean_files_dir, 140)
    get_top_n_words(clean_files_dir, 250)


def remove_uncommon_features(feature_file, corpus, limit, feature_type, ngram, n, tokenize):
    """Removes features that do not occur in a high number of texts. This number is set using 'limit'.
       n: the intended number of features in the file."""
    uncommon = check_features(feature_type, corpus, ngram, tokenize)
    uncommon = dict(uncommon)
    remove = []
    for token in uncommon.keys():
        if uncommon[token] > limit:
            remove.append(token)

    filtered_features = []
    features = []
    with open(feature_file, "r", encoding="utf8") as file:
        for line in file:
            if ngram == None:
                features.append(line.strip())
            else: 
                line = line.replace("\n", "")
                features.append(line)

    for feature in features:
        if feature not in remove:
            filtered_features.append(feature)

    filtered_features = filtered_features[:n]

    with open(feature_file, "w", encoding="utf8") as f:
        for token in filtered_features:
            f.write(token + "\n")
    print(f"Filtered {feature_file}")


def filter_features(corpus_obj):
    """Filter out uncommon features from all feature sets."""
    corpus_obj.load_corpus(tokenized=True)
    corpus = corpus_obj.corpus
    remove_uncommon_features("top_22_tokens.txt", corpus, 124, "freq_22", ngram=None, n=22, tokenize=True)
    remove_uncommon_features("top_120_tokens.txt", corpus, 124, "freq_120", ngram=None, n=120, tokenize=True)
    remove_uncommon_features("top_140_tokens.txt", corpus, 124, "freq_140", ngram=None, n=140, tokenize=True)
    remove_uncommon_features("top_250_tokens.txt", corpus, 124, "freq_250", ngram=None, n=250, tokenize=True)

    corpus_obj.load_corpus(tokenized=False)
    corpus = corpus_obj.corpus
    remove_uncommon_features("top_5_grams.txt", corpus, 124, "ngrams", ngram=5, n=2000, tokenize=False)
    remove_uncommon_features("top_6_grams.txt", corpus, 124, "ngrams", ngram=6, n=2000, tokenize=False)
    remove_uncommon_features("top_7_grams.txt", corpus, 124, "ngrams", ngram=7, n=2000, tokenize=False)
    remove_uncommon_features("top_4_grams.txt", corpus, 124, "ngrams", ngram=4, n=2000, tokenize=False)


def count_tokens_in_texts(files_dir):
    """Counts the number of tokens in each text and prints the longest and shortest counts."""
    counts = []
    files = get_files(files_dir)
    for file in files:
        with open(file, "r", encoding="utf8") as f:
            text = f.readlines()[0]
        text = word_tokenize(text.lower(), language="swedish")
        count = len(text)
        counts.append(count)
    print(counts)
    print(f"shortest: {min(counts)}")
    print(f"longest: {max(counts)}")


def remove_token(token, directory):
    """Ad-hoc function. Used to remove single token types in all cleaned text
       found upon manual inspection after initial cleaning and file creation."""
    files = get_files(directory)
    for file in files:
        with open(file, "r", encoding="utf8") as f:
            content = f.readlines()
        if token in content[0]:
            new_text = content[0].replace(token, "")
            with open(file, "w", encoding="utf8") as f:
                f.write(new_text)
            print("removed token")


#This is the real main, if starting over
def main():
    print("Creating clean files...")
    create_clean_files(raw_dir) # Create cleaned files from raw xml data
    print("File creation done.")
    get_author_names(raw_dir, "author_names.txt") # Create file with author names. Genders need to be added manually into 
                                                # this file before use in load_author_names()
    authors = load_author_names("author_names.txt") # Get author names from saved file
    novel_corpus = Corpus(authors) # Create corpus object with authors

    print("Creating corpora...")
    novel_corpus.create_authors_corpus(clean_files_dir_1, tokenize=True) # Create json file with tokenized novels
    novel_corpus.create_authors_corpus(clean_files_dir_1, tokenize=False) # Create json file with novels as strings
    print("Corpus creation done.")

    novel_corpus.load_corpus(tokenized=True)
    create_info(novel_corpus, "info_1.txt") # Create information on existing authors in the corpus and save to file
    print("Info file created.")

    print("Creating feature files...")
    create_feature_files() # Create lists of features saved in textfiles
    print("Filtering features...")
    filter_features(novel_corpus) # Filter out uncommon features and create new files

    print("The novel data is ready to use in classification/similarity calculations.")


if __name__ == "__main__":
    main()
    #get_author_names(raw_dir, "author_names.txt")
   # authors = load_author_names("author_names.txt")
   # for name in authors:
     #   print(name)
    #print(authors)
    #a = Corpus(authors)
    #a.load_corpus(tokenized=True)
    #create_info(a)
    #create_info(raw_dir, a)
    #a.create_authors_corpus(clean_files_dir, tokenize=True)
    #a.create_authors_corpus(clean_files_dir, tokenize=False)

    #print(len(a.authors_texts["Martin Koch@@M"]['Ellen En liten historia']))
    #print(a.authors_token_counts())
    #print(check_features("ngrams", a.authors_texts, ngram=7, tokenize=False))
    #print(check_features("ngrams", a.authors_texts, ngram=4, tokenize=False))
    #create_feature_files()
    #a.load_corpus(tokenized=True)
   #create_feature_files()
    #remove_uncommon_features("top_350_tokens.txt", a.authors_texts, 124, "freq_250", None, tokenize=True)
   # remove_uncommon_features("freq_short.txt", a.authors_texts, 124, "freq_short", None, tokenize=True)
  #  remove_uncommon_features("stopwords.txt", a.authors_texts, 124, "stopwords", None, tokenize=True)
   ## remove_uncommon_features("top_5_grams_2500.txt", a.authors_texts, 124, "ngrams", 5, tokenize=False)
   # remove_uncommon_features("top_6_grams_2500.txt", a.authors_texts, 124, "ngrams", 6, tokenize=False)
   # remove_uncommon_features("top_7_grams_2500.txt", a.authors_texts, 124, "ngrams", 7, tokenize=False)
    #print(check_features("ngrams", a.authors_texts, ngram=5, tokenize=False))
    #remove_uncommon_features("top_4_grams_2500.txt", a.authors_texts, 124, "ngrams", 4, tokenize=False)
   # remove_uncommon_features("top_5_grams_2500.txt", a.authors_texts, 124, "ngrams", 5, tokenize=False)
   # remove_uncommon_features("top_6_grams_2500.txt", a.authors_texts, 124, "ngrams", 6, tokenize=False)
   # remove_uncommon_features("top_7_grams_2500.txt", a.authors_texts, 124, "ngrams", 7, tokenize=False)
   # print(check_features("ngrams", a.authors_texts, ngram=4, tokenize=False))
   # print(check_features("ngrams", a.authors_texts, ngram=5, tokenize=False))
   # print(check_features("freq_short", a.authors_texts, None, tokenize=True))
    #print(check_features("stopwords", a.authors_texts, None, tokenize=True))


    #def get_genders(self):
    #"""Possibly useful in classification tasks. Gender info has been added manually in 
    #info file."""
    #with open("author_info_2.txt", "r", encoding="utf8") as f:
    #    genders = []
    #    for line in f:
    #        try:
    #            items = line.split("\t")
    #            genders.append(items[-1].strip())
    #        except IndexError:
    #            continue

    #    author_genders = []
    #    for i, author in enumerate(self.authors):
    #        author_genders.append(author+"@@"+genders[i])
    #    self.author_genders = author_genders

