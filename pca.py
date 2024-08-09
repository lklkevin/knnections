import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from vectorize import load_vect as load


def apply_pca(word_vectors, features = 2):
    """applies PCA to given word vectors and # of projected features

    Args:
        word_vectors (20 x D): word vectors to PCA based on
        features (int, optional): Number of projected features. Defaults to 2.

    Returns:
        tuple[20 x features, float]: returns all word vectors projected on PCA matrix and the variance of the original data accounted for by each PCA component
    """
    pca = PCA(n_components=features)
    result = pca.fit_transform(word_vectors)
    return result, pca.explained_variance_ratio_


def pca_gridsearch(word_vectors):
    ratios = []
    for i in range(2, 16):
        pca = PCA(n_components=i).fit(word_vectors)
        ratios.append(np.sum(pca.explained_variance_ratio_))

    return np.array(ratios)


if __name__ == "__main__":
    data, _ = load('./data/bert_ft_vect.npy', './data/bert_lb_vect.npy')
    final = np.zeros(14)
    for i in range(data.shape[0]):
        final = np.add(final, pca_gridsearch(data[i]))
    
    final /= data.shape[0]
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(2, 16), final, marker='o', linestyle='-')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Cumulative Explained Variance Ratio from the x Components')
    plt.title('Cumulative Explained Variance Ratio vs Number of Components in Training Data')
    plt.grid(True)
    plt.show()
