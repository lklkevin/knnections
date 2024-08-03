from vectorize import *
from pca import *
# from distance_opt import *
from sklearn.model_selection import train_test_split

WORDS_PER_DAY = 16
CATEGORY_COUNT = 4

def gridsearch_psa(train_data, train_labels, train_label_strings):
    ALL_FEATURES = len(train_data[0][0])
    for FEATURE_COUNT in range(1, min(ALL_FEATURES, WORDS_PER_DAY + CATEGORY_COUNT) + 1):
        counter = 0
        for i in range(len(train_data)):
            day_vectors = np.vstack([train_data[i], train_labels[i]])
            day_vectors, variance = apply_pca(day_vectors, FEATURE_COUNT)
            inputs, labels = day_vectors[:WORDS_PER_DAY], day_vectors[WORDS_PER_DAY:]
            counter += kmeans(inputs, train_label_strings)
        print(f"{FEATURE_COUNT} pca dim: {counter / (len(data) * 4)}")

if __name__ == '__main__':
    print(f"Accuracy by randomly guessing: {random_guess()}\n")
    
    RUN_SETTINGS = {
        'gridsearch_pca': True,
    }
    
    data = np.load("features.npy") # N x 16 x D
    clusters = np.load("labels.npy") # N x 16
    label_strings = np.load("full_ordered.npy") # N x 16
    
    X_train, X_off, y_train, y_off, label_str_train, label_str_off = train_test_split(data, clusters, label_strings, test_size=0.3)
    X_valid, X_test, y_valid, y_test, label_str_valid, label_str_test = train_test_split(X_off, y_off, label_str_off, test_size=0.5)
    
    if (RUN_SETTINGS['gridsearch_pca']):
        gridsearch_psa(X_train, y_train, label_str_train)
        
    
    
    
        