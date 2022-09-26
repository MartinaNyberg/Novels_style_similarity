# An exploration of stylistic similarity in Swedish novels

This project examines the writing style of Swedish authors from the perspective of authorship attribution and similarity calculations. Using data consisting of 249 Swedish novels written by 78 authors, several variants of feature sets are employed in an SVM and a Naive Bayes classifier for the task of identifying the authors based on stylistic properties. The novels (which are not included in this repo), written during the 19th and 20th centuries, were collected from the Swedish Literature Bank (litteraturbanken.se). 

For a more detailed presentation of the project, please see the (unpublished) paper "Authorship Attribution for Swedish Novels" in this repository. 

## Feature sets
The writing style in each novel is represented as either the relative frequency counts of the k most common tokens, or the k most common character n-grams with size ranging from 4 to 7. The features were extracted by computing the token/n-gram counts of all tokens/n-grams across all texts. The k most frequent tokens/n-grams were then chosen as features.

All the preprocessing of the original novel xml files are implemented in the file ``preprocess_data.py``, including the creation of feature sets. 

## Encoding
For the classification tasks, the collected texts of each author are split into nine segments of equal size, removing any trailing tokens. This results in 702 segments. Each segment is then encoded as a vector containing the token/n-gram frequency in the segment for each token/n-gram in the predefined feature set, normalized by the total number of tokens/n-grams for that segment. 

The division into segments and vector encoding is implemented in the file ``load_encode.py``.

## Classification (Authorship Attribution)
The different variants of encoded segments were used as input in classification tasks with a Support Vector Machine classifier (SVM) and a Multinomial Naive Bayes classifier, implemented through the scikit-klearn library. The file ``classification.py`` contains several functions for running classifications across all feature sets or for a single variant, including grid searching for the best hyper parameters, tracking misclassifications and displaying the most contributing features per author. 

## Author/novel similarity
On the basis of the results in the authorship attribution task, one feature set is selected to represent the writing style in pairwise comparisons of stylistic similarity. The Jensen-Shannon divergence is used to measure similarity, which quantifies the difference between two probability distributions. The relative frequency vectors of each novel/author are converted into probability distributions and used as input to the similarity measure.

This implementation is found in the file ``calculate_similarity.py``, calculating e.g. which authors/novels are the most similar to a given author/novel in the collection, based on the chosen represenation of style. 

Output from some example runs are given below. 

```
>> get_similar_novels("Doktor Glas Roman", 4)

The 4 novel(s) most similar to Doktor Glas Roman:
Hjalmar Söderberg             Den allvarsamma leken
Hjalmar Söderberg             Hjärtats oro
Elin Wägner                   Norrtullsligan Elisabeths krönika       
Karin Boye                    Kallocain Roman från 2000-talet
```

```
>> get_similar_authors("Selma Lagerlöf", 3)

The 3 author(s) most similar to Selma Lagerlöf:
Sophie Elkan
Elin Wägner
Jenny Maria Ödmann
```
```
>> find_most_different_novel("En herrgårdssägen")

The novel most different to En herrgårdssägen:
August Strindberg: Samlade Verk Nationalupplaga 22 Han och hon en själs utvecklingshistoria (1875-76)
```
