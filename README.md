# An exploration of stylistic similarity in Swedish novels

This project examines the writing style of Swedish authors from the perspective of authorship attribution and similarity calculations. Using data consisting of Swedish novels written by 72 authors, several variants of feature sets are employed in an SVM and a Naive Bayes classifier for the task of identifying the authors based on stylistic properties.

The writing style in each novel is represented as either the relative frequency counts of the k most common tokens, or the k most common character n-grams with size ranging from 4 to 7. The features were extracted by computing the token/n-gram counts of all tokens/n-grams across all texts. The k most frequent tokens/n-grams were then chosen as features.

For the classification tasks, the collected texts of each author were split into nine segments of equal size, removing any trailing tokens. This resulted in 702 segments. Each segment was then encoded as a vector containing the token/n-gram frequency in the segment for each token/n-gram in the predefined feature set, normalized by the total number of tokens/n-grams for that segment.
