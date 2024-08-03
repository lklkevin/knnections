import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def apply_pca(word_vectors, NUM_FEATURES = 2):
    """applies PCA to given word vectors and # of projected features

    Args:
        word_vectors (20 x D): word vectors to PCA based on
        NUM_FEATURES (int, optional): Number of projected features. Defaults to 2.

    Returns:
        tuple[20 x NUM_FEATURES, float]: returns all word vectors projected on PCA matrix and the variance for the PCA matrix
    """
    pca = PCA(n_components=NUM_FEATURES)
    result = pca.fit_transform(word_vectors)
    return result, pca.explained_variance_ratio_