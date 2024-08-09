from vectorize import *
from pca import *
from distance_opt import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import autoencoder
import cluster

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
        print(f"{FEATURE_COUNT} pca dim: {counter / (len(train_data) * 4)}")


def optimize_day(day_vectors):
    inputs, labels = day_vectors[:WORDS_PER_DAY], np.repeat(day_vectors[WORDS_PER_DAY:], 4, axis=0)
    model = train_and_get_model(inputs, labels, inputs.shape[1], False)
    return model(inputs).detach().numpy()


def reduce_dim_day(x, y, reduced_dim, default=True):
    day_vectors = np.vstack([x, y])
    if default:
        day_vectors, _ = apply_pca(day_vectors, reduced_dim)
    else:
        day_vectors, _ = autoencoder.optimize(day_vectors.shape[1], day_vectors, reduced_dim)

    return day_vectors


def check(groupings, correct_order):
    g1 = list(correct_order[0:4])
    g2 = list(correct_order[4:8])
    g3 = list(correct_order[8:12])
    g4 = list(correct_order[12:16])
    combined = [g1, g2, g3, g4]
    correct = {i:0 for i in range(4)}

    for group in groupings:
        if group in combined:
            correct[combined.index(group)] += 1

    return correct


if __name__ == '__main__':
    # print(f"Accuracy by randomly guessing: {random_guess()}\n")
    
    RUN_SETTINGS = {
        'gridsearch_pca': False,
        'pca_k': 7,
        'enable_epoch_print': False,
    }
    
    feature_vects = np.load('bert_ft_vect.npy') # N x 16 x D
    feature_names = np.load('bert_ft.npy') # N x 16
    label_vects = np.load('bert_lb_vect.npy') # N x 4 x D
    label_names = np.load('bert_lb.npy') # N x 4

    X_tr, X_tt, X_tr_str, X_tt_str, y_tr, y_tt, y_tr_str, y_tt_str = train_test_split(
        feature_vects, feature_names, label_vects, label_names, test_size=0.3, random_state=0
    )
    
    if (RUN_SETTINGS['gridsearch_pca']):
        gridsearch_psa(X_tr, y_tr, y_tr_str)
    
    correct = {i:0 for i in range(4)}

    for i in range(len(X_tr)):
        transformed_inputs = optimize_day(reduce_dim_day(X_tr[i], y_tr[i], 7, True))
        transformed_inputs = preprocessing.normalize(transformed_inputs)
        guesses = cluster.kmeans(transformed_inputs, X_tr_str[i])
        right = check(guesses, X_tr_str[i])

        correct = {i:correct[i] + right[i] for i in range(4)}

        print(right)
        