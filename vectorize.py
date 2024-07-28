import gensim.downloader as api
import numpy as np
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
import matplotlib.pyplot as plt

words = ["CLAY",
         "PAPYRUS",
         "PARCHMENT",
         "WAX",
         "ANCHOR",
         "HOST",
         "MODERATE",
         "PRESENT",
         "FACULTY",
         "FLAIR",
         "INSTINCT",
         "TALENT",
         "BURRITO",
         "GIFT",
         "MUMMY",
         "SPRAIN"
         ]

vectorizer = api.load('glove-twitter-25')
word_vectors = []

for word in words:
    word = word.lower()
    if word in vectorizer:
        word_vectors.append(vectorizer[word])

word_vectors = np.array(word_vectors)
km = KMeansConstrained(n_clusters=4, size_min=4, size_max=4, random_state=0)
fitted = km.fit(word_vectors)

clusters = km.predict(word_vectors)

# Group words by their cluster
grouped_words = {}
for word, cluster in zip(words, clusters):
    if cluster not in grouped_words:
        grouped_words[cluster] = []
    grouped_words[cluster].append(word)

# Print the grouped words
for cluster, group in grouped_words.items():
    print(f"Cluster {cluster}: {group}")
