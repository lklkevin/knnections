import random
import gensim.downloader as api
import numpy as np
from k_means_constrained import KMeansConstrained
import scraper
from sklearn.model_selection import train_test_split


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


def load_vect(x1, x2, x3=None, x4=None):
    x1 = np.load(x1)
    x2 = np.load(x2)
    if not x3 or not x4:
        return train_test_split(np.concatenate((x1, x2), axis=1), test_size=0.3, random_state=0)
    
    x3 = np.load(x3)
    x4 = np.load(x4)

    return train_test_split(x1, x2, x3, x4, test_size=0.3, random_state=0)


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
