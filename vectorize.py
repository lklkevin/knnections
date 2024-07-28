import gensim.downloader as api
import numpy as np

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

vectorizer = api.load('fasttext-wiki-news-subwords-300')
print('fuck')
word_vectors = []

for word in word_vectors:
    if word in vectorizer:
        word_vectors.append(vectorizer[word])

print(np.array(word_vectors))
