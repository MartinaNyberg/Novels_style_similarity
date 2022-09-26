from load_encode import *
import numpy as np
from scipy.spatial import distance

def get_prob_dist(X):
    prob_distribution = []
    for segment in X:
        probabilities = []
        sum_of_all = np.sum(segment)
        new = np.zeros(len(segment))
        for i, value in enumerate(segment):
            probabilities.append(value / sum_of_all)
        prob_distribution.append(probabilities)

    pd = np.vstack(prob_distribution)
    return pd


def calculate_novel_similarity(pd, x, n, Y, novels=None):
    sim_list = []
    for i, seg in enumerate(pd):
        divergence = distance.jensenshannon(x, pd[i]) ** 2
        similarity = 1 - divergence
        sim_list.append((Y[i], novels[i], similarity))
    sim_list.sort(key=lambda x:x[2], reverse=True)
    return sim_list[:n]


def calculate_segment_similarity(pd, x, n, Y):
    sim_list = []
    for i, seg in enumerate(pd):
        divergence = distance.jensenshannon(x, pd[i]) ** 2
        similarity = 1 - divergence
        sim_list.append((Y[i], similarity))
    sim_list.sort(key=lambda x:x[1], reverse=True)
    return sim_list[:n]


def scale(sim_list, novels=None):
    scaled_values = []
    if novels == None:
        sim_values = [t[1] for t in sim_list]
        for i, m in enumerate(sim_list):
            scaled = ((m[1] - min(sim_values)) / (max(sim_values) - min(sim_values)))
            scaled_values.append((m[0], scaled))
    else:
        sim_values = [t[2] for t in sim_list]
        for i, m in enumerate(sim_list):
            scaled = ((m[2] - min(sim_values)) / (max(sim_values) - min(sim_values)))
            scaled_values.append((m[0], m[1], scaled))
    return scaled_values


def print_novel_names():
    _, Y, _, text_names = load_and_encode_data("freq_120", n_segments=None, verbose=False)
    text_names = [name[:-7] for name in text_names]
    for i, author in enumerate(Y):
        print("{:<5}{:<30}{:<20}".format(i, author, text_names[i]))


def print_authors():
    _, Y, _ = load_and_encode_data("freq_120", n_segments=1, verbose=False)
    for i, author in enumerate(Y):
        print(f"{i}  {author}")


def get_similar_segments(segment_index, Y, n, pd):
    sim = calculate_segment_similarity(pd, pd[segment_index], n, Y)
    scaled = scale(sim, novels=None)
    return scaled


def find_most_different_novel(novel_name):
    X, Y, _, text_names = load_and_encode_data("freq_120", n_segments=None, verbose=False)
    text_names = [name[:-7] for name in text_names]
    pd = get_prob_dist(X)
    if novel_name in text_names:
        segment_index = text_names.index(novel_name)
        sim = calculate_novel_similarity(pd, pd[segment_index], len(text_names), Y, novels=text_names)
        scaled = scale(sim, novels=text_names)
        print(f"\nThe novel most different to {novel_name}:")
        print(f"{scaled[-1][0]}: {scaled[-1][1]}")
    print("\n")

def get_similar_novels(novel_name, n):
    X, Y, _, text_names = load_and_encode_data("freq_120", n_segments=None, verbose=False)
    text_names = [name[:-7] for name in text_names]
    pd = get_prob_dist(X)
    if novel_name in text_names:
        segment_index = text_names.index(novel_name)
        sim = calculate_novel_similarity(pd, pd[segment_index], n+1, Y, novels=text_names)
        scaled = scale(sim, novels=text_names)
    else:
        print(f"ERROR: The name {novel_name} is not a novel in the collection.")
        return None
    print(f"\nThe {n} novel(s) most similar to {novel_name}:")
    for t in scaled[1:]:
            print("{:<30}{:<40}".format(t[0], t[1]))
    print("\n")

def get_similar_authors(author_name, n):
    X, Y, _ = load_and_encode_data("freq_120", n_segments=1, verbose=False)
    pd = get_prob_dist(X)
    for author_i in range(0, 78):
        result = get_similar_segments(author_i, Y, n+1, pd)
        if result[0][0] == author_name:
            print(f"\nThe {n} author(s) most similar to {author_name}:")
            for t in result[1:]:
                print(t[0])
            print("\n")
            return
    print("ERROR: f{author_name} was not found in the collection.")


if __name__ == "__main__": # Example run

    #Displays all authors and novels in the collection.
    print_authors()
    print_novel_names()

    #Prints the 4 novels most similar to 'Doktor Glas Roman' in descending order.  
    get_similar_novels("Doktor Glas Roman", 4)

    #Prints the 3 authors most similar to 'Selma Lagerlöf' in descending order.  
    get_similar_authors("Selma Lagerlöf", 3)

    #Prints the novel most different from 'En herrgårdssägen'.  
    find_most_different_novel("En herrgårdssägen")


