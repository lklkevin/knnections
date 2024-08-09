import random
import gensim.downloader as api
import numpy as np
from k_means_constrained import KMeansConstrained
import scraper
from pca import apply_pca


def init_vectors(vect_name: str):
    data = scraper.get_data()
    vectorizer = api.load(vect_name)

    features = []
    labels = []
    correct = []

    for entry in data:
        x = entry[1]
        y = entry[0]
        if x == "" or y == "":
            continue

        xk, y = vectorize(x, y, vectorizer)
        if xk is not None and len(x) == 16:
            features.append(xk)
            correct.append(np.array(x))
            labels.append(y)

    return features, correct


def vectorize(x: list, y: list, vectorizer):
    # y is suppose to be labels but they cant be vectorized bc theyre mostly longer than 1 word
    word_vectors = []
    label_vectors = []
    for word in x:
        word = word.lower()
        if word in vectorizer:
            word_vectors.append(vectorizer[word])
        else:
            return None, None

    # for word in y:
    #     word = word.lower()z
    #     if word in vectorizer:
    #         label_vectors.append(vectorizer[word])
    #     else:
    #         print(word)
    #         return None, None

    return np.array(word_vectors), np.array(label_vectors)


def random_guess():
    data = scraper.get_data()
    total_groups = 0
    correct_groups = 0

    for entry in data:
        correct = entry[1]
        if len(correct) != 16:
            continue

        g1 = correct[0:4]
        g2 = correct[4:8]
        g3 = correct[8:12]
        g4 = correct[12:16]

        correct_groups_list = [g1, g2, g3, g4]

        shuffled_correct = correct[:]
        random.shuffle(shuffled_correct)

        guessed_groups = [shuffled_correct[i:i + 4] for i in range(0, 16, 4)]

        for guessed_group in guessed_groups:
            if guessed_group in correct_groups_list:
                correct_groups += 1

        total_groups += 4

    return correct_groups / total_groups


def kmeans(word_vectors, correct):
    """
    returns # of groups predicted correctly using k-means clustering algorithm

    Args:
        word_vectors (np array): 16 x D, contains 16 vectorized words
        correct (np array): 16 x 1, contains string of correct group of ith word

    Returns:
        int: # of correct groups
    """
    km = KMeansConstrained(n_clusters=4, size_min=4, size_max=4, random_state=0)
    fitted = km.fit(word_vectors)
    g1 = correct[0:4]
    g2 = correct[4:8]
    g3 = correct[8:12]
    g4 = correct[12:16]

    clusters = km.predict(word_vectors)
    grouped_words = {}
    for word, cluster in zip(correct, clusters):
        if cluster not in grouped_words:
            grouped_words[cluster] = []
        grouped_words[cluster].append(word)

    right = 0
    for cluster, group in grouped_words.items():
        if any(np.array_equal(group, g) for g in [g1, g2, g3, g4]):
            right += 1

    return right
