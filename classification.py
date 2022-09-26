from load_encode import *
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate

svc_hyperparameters = [{"kernel": ["linear"], "C": [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000], "random_state": [None]}]
nb_hyperparameters = [{"alpha": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1]}]

frequency_features = ["freq_250", "freq_140", "freq_120", "freq_22"]
ngrams = [4, 5, 6, 7]
profiles = [22, 120, 250, 500, 1000, 1500, 2000]

combinations = []
for ngram in ngrams:
    for size in profiles:
        combinations.append((ngram, size))


def classification(X, Y, classifier, kernel=None, C=None, random_state=None):
    if classifier == "svc":
        clf = SVC(C=C, kernel=kernel, random_state=random_state)
        X = StandardScaler().fit_transform(X)
       
    elif classifier == "nb":
        clf = MultinomialNB(alpha=0.0001)

    cv_results = cross_validate(clf, X, Y, cv=9, return_estimator=True)
    print("\n")
    print("%0.2f accuracy with a standard deviation of %0.2f \n" % (cv_results['test_score'].mean(), cv_results['test_score'].std()))
    predictions = cross_val_predict(clf, X, Y, cv=9)
    print_errors(predictions, Y)
    return cv_results


def print_errors(predictions, Y):
    for i, pred in enumerate(predictions):
        if pred != Y[i]:
            print(f"predicted: {pred}, true: {Y[i]}, {i}")
    print("\n")
    

def print_top20(feature_names, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    print("Most important features for each author:")
    for i, class_label in enumerate(class_labels):
        top20 = np.argsort(clf[i])[-20:]

        print(f"{class_label}: {' '.join(feature_names[j] for j in top20)}")
        print(f"Indices in feature set: {top20}\n")


def get_important_features(cv_results, feature_names):
    class_labels=cv_results['estimator'][0].classes_
    all_runs = dict()
    for n in range(len(cv_results['estimator'])):
        i = 0
        for author_coef in cv_results['estimator'][n].coef_:
            if str(i) in all_runs:
                all_runs[str(i)].append(author_coef)
            else:
                all_runs[str(i)] = [author_coef]
            i += 1
    means = []
    for coefs in all_runs.values():
        all_coefs = np.vstack(coefs)
        means.append(all_coefs.mean(axis=0))

    print_top20(feature_names, means, class_labels)

def grid_search(X, Y, classifier):
    if classifier == "svc":
        svc = SVC()
        X = StandardScaler().fit_transform(X)
        clf = GridSearchCV(svc, svc_hyperparameters, cv = 9, scoring='accuracy')
        clf.fit(X, Y)
        print(clf.best_params_)
        print(clf.best_score_)

    elif classifier == "nb":
        nb = MultinomialNB()
        clf = GridSearchCV(nb, nb_hyperparameters, cv = 9, scoring='accuracy')
        clf.fit(X, Y)
        print(clf.best_params_)
        print(clf.best_score_)


def grid_search_all_nb(combinations):
    print("Word frequency features.")
    X, Y = load_and_encode_data("freq_250", n_segments=9)
    grid_search(X, Y, "nb")
    print("\n")
    count = 1
    for t in combinations:
        n_gram = t[0]
        profile_n = t[1]
        print(f"Run {count}")
        print(f"ngram: {n_gram}, profile: {profile_n}")
        X, Y = load_and_encode_data("ngrams", n_segments=9, profile_size=profile_n, ngram_size=n_gram)
        grid_search(X, Y, "nb")
        print("\n")
        count += 1

def grid_search_all_svm(combinations):
    print("Word frequency features.")
    X, Y, _ = load_and_encode_data("freq_140")
    grid_search(X, Y, "svc")
    print("\n")
    count = 1
    for t in combinations:
        n_gram = t[0]
        profile_n = t[1]
        print(f"Run {count}")
        print(f"ngram: {n_gram}, profile: {profile_n}")
        X, Y, _ = load_and_encode_data("ngrams", n_segments=9, profile_size=profile_n, ngram_size=n_gram)
        grid_search(X, Y, "svc")
        print("\n")
        count += 1


def run_all_svm(combinations, token_features):
    count = 1
    for feature in token_features:
        print(feature, "\n")
        X, Y, _ = load_and_encode_data(feature, n_segments=9)
        classification(X, Y, "svc", kernel="linear", C=0.01, random_state=65)
        print("\n")
        count += 1
    for t in combinations:
        n_gram = t[0]
        profile_n = t[1]
        print(f"Run {count}")
        print(f"ngram: {n_gram}, profile: {profile_n}")
        X, Y, _ = load_and_encode_data("ngrams", n_segments=9, profile_size=profile_n, ngram_size=n_gram)
        classification(X, Y, "svc", kernel="linear", C=0.01, random_state=65) 
        print("\n")
        count += 1


def run_all_multinomialNB(combinations, token_features):
    print("------- Multinomial NaiveBayes ------")
    count = 1
    for feature in token_features:
        print(feature, "\n")
        X, Y, _ = load_and_encode_data(feature, n_segments=9)
        classification(X, Y, "nb")
        print("\n")
        count += 1
    for t in combinations:
        n_gram = t[0]
        profile_n = t[1]
        print(f"Run {count}")
        print(f"ngram: {n_gram}, profile: {profile_n}")
        X, Y, _ = load_and_encode_data("ngrams", n_segments=9, profile_size=profile_n, ngram_size=n_gram)
        classification(X, Y, "nb")
        print("\n")
        count += 1

if __name__ == "__main__": # Example run
    X, Y, _ = load_and_encode_data("freq_120", n_segments=9, profile_size=120, ngram_size=None)
    feature_names = load_features("freq_120", ngram=None)
    result = classification(X, Y, "svc", kernel="linear", C=0.01, random_state=65) # Perform classification
    get_important_features(result, feature_names) # Print the most important features per author



