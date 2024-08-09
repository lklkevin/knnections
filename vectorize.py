import random
import gensim.downloader as api
import numpy as np
from k_means_constrained import KMeansConstrained
import scraper


def init_vectors(vect_name: str):
    data = scraper.get_data()
    vectorizer = api.load(vect_name)

    features = []
    correct = []

    for entry in data:
        x = entry[1]
        y = entry[0]
        if x == "" or y == "":
            continue

        xk = vectorize(x, vectorizer)
        if xk is not None and len(x) == 16:
            features.append(xk)
            correct.append(np.array(x))

    return np.stack(features), np.stack(correct)


def vectorize(x: list, vectorizer):
    # y is suppose to be labels but they cant be vectorized bc theyre mostly longer than 1 word
    word_vectors = []
    for word in x:
        word = word.lower()
        if word in vectorizer:
            word_vectors.append(vectorizer[word])
        else:
            return None

    return np.array(word_vectors)


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
        word_vectors (np array): 16 x D, contains 16 vectorized words in the correct order
        correct (np array): 16 x 1, the 16 words as strings in the same order as the vectors
    Returns:
        int: # of correct groups
    """
    km = KMeansConstrained(n_clusters=4, size_min=4, size_max=4, random_state=0)
    fitted = km.fit(word_vectors)
    g1 = correct[0:4]
    g2 = correct[4:8]
    g3 = correct[8:12]
    g4 = correct[12:16]
    km = KMeansConstrained(n_clusters=4, size_min=4, size_max=4, random_state=0).fit(word_vectors)

    clusters = km.predict(word_vectors)
    print(clusters)
    grouped_words = {}
    for i in range(16):
        if clusters[i] not in grouped_words:
            grouped_words[clusters[i]] = []
        grouped_words[clusters[i]].append(correct[i])

    right = 0
    for cluster, group in grouped_words.items():
        if any(np.array_equal(group, g) for g in [g1, g2, g3, g4]):
            right += 1

    return right


if __name__ == "__main__":
    w2v, w2v_actual = init_vectors('word2vec-google-news-300')
    glv, glv_actual = init_vectors('glove-wiki-gigaword-300')
    fst, fst_actual = init_vectors('fasttext-wiki-news-subwords-300')

    print(w2v.shape)
    print(glv.shape)
    print(fst.shape)

    np.save('w2v.npy', w2v)
    np.save('w2v_actual.npy', w2v_actual)
    np.save('glv.npy', glv)
    np.save('glv_actual.npy', glv_actual)
    np.save('fst.npy', fst)
    np.save('fst_actual.npy', fst_actual)

# print(random_guess())
# # a, b = init_vectors('fasttext-wiki-news-subwords-300')
# data = np.load("data.npy") # N x 16 x D
# clusters = np.load("ordered.npy") # N x 16
# # a = apply_pca(a)
# ALL_FEATURES = len(data[0][0])
# for FEATURE_COUNT in range(1, min(ALL_FEATURES, 16) + 1):
#     counter = 0
#     for i in range(len(data)):
#         day_vectors = data[i]
#         counter += kmeans(apply_pca(day_vectors, FEATURE_COUNT), clusters[i])
#     print(f"{FEATURE_COUNT} pca dim: {counter / (len(data) * 4)}")
#
# print("------------------------------------------------------------------")
# data2 = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

# for FEATURE_COUNT in range(1, ALL_FEATURES):
#     counter = 0
#     temp = apply_pca(data2, FEATURE_COUNT)
#     temp = temp.reshape(data.shape[0], data.shape[1], temp.shape[1])
#     for i in range(len(data)):
#         day_vectors = temp[i]
#         counter += kmeans(day_vectors, clusters[i])
#     print(f"{FEATURE_COUNT} pca dim: {counter / (len(data) * 4)}")
