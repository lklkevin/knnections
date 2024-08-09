from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


def kmeans(word_vectors, words, constrained=True):

    # Perform constrained variant of KMeans algorithm
    if constrained:
        km = KMeansConstrained(n_clusters=4, size_min=4, size_max=4, random_state=0).fit(word_vectors)
    else:
        km = KMeans(4, random_state=0, n_init='auto').fit(word_vectors)

    # Clusters is a row of 16 ints, indicating which cluster the i-th word is in
    clusters = km.predict(word_vectors)

    # Groups of actual words using clusters
    groups = [[],[],[],[]]
    for i in range(words.shape[0]):
        groups[clusters[i]].append(words[i])

    return groups


def hca(word_vectors, words):
    # Same as kmeans but we're using a hierarchical clustering algorithm
    hc = AgglomerativeClustering(n_clusters=4).fit(word_vectors)

    clusters = hc.labels_
    groups = [[],[],[],[]]
    for i in range(words.shape[0]):
        groups[clusters[i]].append(words[i])

    return groups
