import gensim.downloader as api
import numpy as np
import scraper
from sklearn.model_selection import train_test_split


def init_vectors(vect_name: str):
    # Data is a list of tuples, each tuple is one day, with the categories as the 0th
    # element and words as the 1st element
    data = scraper.get_data()
    vectorizer = api.load(vect_name)

    features = []
    ft_strs = []

    for entry in data:
        x = entry[1]
        y = entry[0]

        # Remove invalid days
        if x == "" or y == "" or len(x) != 16:
            continue
        
        # xk is 16 by D
        xk = vectorize(x, vectorizer)

        # Ensure the words can be vectorized
        if xk is not None:
            features.append(xk)
            ft_strs.append(np.array(x))

    # Stack the list into a matrix
    return np.stack(features), np.stack(ft_strs)


def vectorize(x: list, vectorizer):
    word_vectors = []
    for word in x:
        word = word.lower()
        if word in vectorizer:
            word_vectors.append(vectorizer[word])
        else:
            # As soon as a word is unable to be vectorized, return None
            return None

    return np.array(word_vectors)


def load_vect(x1, x2, x3=None, x4=None, merge=False):
    x1 = np.load(x1)
    x2 = np.load(x2)
    if (not x3 or not x4) and merge:
        tr, tt = train_test_split(np.concatenate((x1, x2), axis=1), test_size=0.3, random_state=0)
        return tr, tt
    elif not x3 or not x4:
        return x1, x2
    
    x3 = np.load(x3)
    x4 = np.load(x4)

    return train_test_split(x1, x2, x3, x4, test_size=0.3, random_state=0)


if __name__ == "__main__":
    w2v, w2v_actual = init_vectors('word2vec-google-news-300')
    glv, glv_actual = init_vectors('glove-wiki-gigaword-300')
    fst, fst_actual = init_vectors('fasttext-wiki-news-subwords-300')

    np.save('w2v.npy', w2v)
    np.save('w2v_actual.npy', w2v_actual)
    np.save('glv.npy', glv)
    np.save('glv_actual.npy', glv_actual)
    np.save('fst.npy', fst)
    np.save('fst_actual.npy', fst_actual)
