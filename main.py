"""
Runner file for knnections.
"""
from vectorize import *
from pca import *
from distance_opt import *
from sklearn import preprocessing
import autoencoder
import cluster
import visualize_vectors

WORDS_PER_DAY = 16
CATEGORY_COUNT = 4


def optimize_day(day_vectors):
    # Runs the distance optimizing NN on a given day of vectors, returns the embedding
    inputs, labels = day_vectors[:WORDS_PER_DAY], np.repeat(day_vectors[WORDS_PER_DAY:], 4, axis=0)
    model = train_and_get_model(inputs, labels, inputs.shape[1])
    return model(inputs).detach().numpy()


def reduce_dim_day(x, y, reduced_dim, default=True):
    # Performs dimension reduction on a single day of features and labels, default uses PCA
    day_vectors = np.vstack([x, y])
    if default:
        day_vectors, _ = apply_pca(day_vectors, reduced_dim)

    else:
        day_vectors, _ = autoencoder.optimize(day_vectors.shape[1], day_vectors, reduced_dim)

    return day_vectors


def check(groupings, correct_order):
    g1 = set(correct_order[0:4])
    g2 = set(correct_order[4:8])
    g3 = set(correct_order[8:12])
    g4 = set(correct_order[12:16])
    combined = [g1, g2, g3, g4]
    correct = {i:0 for i in range(4)}

    for group in groupings:
        if set(group) in combined:
            correct[combined.index(set(group))] += 1

    return correct


def baseline_model(x_vect, x_str):
    # Assume x is correctly ordered
    correct = {i:0 for i in range(4)}

    for i in range(len(x_vect)):

        shuffled_index = np.arange(16)
        np.random.shuffle(shuffled_index)
        feature_vect = preprocessing.normalize(x_vect[i][shuffled_index])
        feature_strs = x_str[i][shuffled_index]

        guesses = cluster.kmeans(feature_vect, feature_strs)
        right = check(guesses, x_str[i])
        correct = {i:correct[i] + right[i] for i in range(4)}

    correct = {i:correct[i] / len(x_vect) for i in range(4)}
    return correct


def random_guess():
    x_str = np.load('./data/fst_actual.npy')
    correct = {i:0 for i in range(4)}

    for i in range(len(x_str)):

        shuffled_index = np.arange(16)
        np.random.shuffle(shuffled_index)
        guesses = x_str[i][shuffled_index]

        right = check([guesses[0:4], guesses[4:8], guesses[8:12], guesses[12:16]], x_str[i])
        correct = {i:correct[i] + right[i] for i in range(4)}

    correct = {i:correct[i] / len(x_str) for i in range(4)}
    return correct


if __name__ == '__main__':
    # tests = [('./data/bert_ft_vect.npy', './data/bert_ft.npy'), ('./data/fst.npy', './data/fst_actual.npy'), ('./data/w2v.npy', './data/w2v_actual.npy'), ('./data/glv.npy', './data/glv_actual.npy')]
    # for test in tests:
    #     x, x_str = load_vect(*test)

    #     fst_result = baseline_model(x, x_str)
    #     avg = sum(fst_result.values()) / 4
    #     print(fst_result)
    #     print(avg)

    # The data above should be something like:
    # vectorizer_performance = {
    #     'FastText': [0.3611859838274933, 0.2884097035040431, 0.1967654986522911, 0.1320754716981132, 0.24460916442048516],
    #     'Word2Vec': [0.3, 0.2257142857142857, 0.14285714285714285, 0.06857142857142857, 0.1842857142857143],
    #     'GloVe': [0.2700534759358289, 0.19518716577540107, 0.13636363636363635, 0.08823529411764706, 0.17245989304812834],
    #     'BERT': [0.12826603325415678, 0.07125890736342043, 0.0498812351543943, 0.019002375296912115, 0.0671021377672209]
    # }

    # groups = ['FastText', 'Word2Vec', 'GloVe', 'BERT']
    categories = ['Yellow', 'Green', 'Blue', 'Purple', 'Total']
    # fig, ax = plt.subplots(figsize=(12, 6))

    # for i, group in enumerate(groups):
    #     ax.bar(np.arange(len(categories)) + i * 0.2, vectorizer_performance[group], 0.2, label=group, color=plt.cm.viridis(i / len(groups)))

    # ax.set_xlabel('Difficulties')
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Comparison of Vectorizer Accuracies Across Connections Difficulties')
    # ax.set_xticks(np.arange(len(categories)) + 0.2 * 1.5)
    # ax.set_xticklabels(categories)
    # ax.legend()
    # plt.tight_layout()
    # plt.show()


    clustering_performance = {
        'Normalized KMeans Constrained': [0.7637795275590551, 0.7322834645669292, 0.6929133858267716, 0.7165354330708661, 0.7263779527559056],
        'Unnormalized KMeans Constrained': [0.84251968503937, 0.8188976377952756, 0.8503937007874016, 0.8582677165354331, 0.8425196850393701],
        'Unnormalized KMeans': [0.8031496062992126, 0.7559055118110236, 0.7716535433070866, 0.7559055118110236, 0.7716535433070866],
        'Unnormalized HCA (Agglomerative)': [0.8031496062992126, 0.7952755905511811, 0.7637795275590551, 0.7322834645669292, 0.7736220472440946]
    }
    groups = ['Normalized KMeans Constrained', 'Unnormalized KMeans Constrained', 'Unnormalized KMeans', 'Unnormalized HCA (Agglomerative)']
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, group in enumerate(groups):
        ax.bar(np.arange(len(categories)) + i * 0.2, clustering_performance[group], 0.2, label=group, color=plt.cm.viridis(i / len(groups)))

    ax.set_xlabel('Difficulties')
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparison of Post-Transform Clustering Algorithm Accuracies Across Connections Difficulties')
    ax.set_xticks(np.arange(len(categories)) + 0.2 * 1.5)
    ax.set_xticklabels(categories)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # print(random_guess())
    
    # The 4 loaded .npy files have respective sizes:
    # N x 16 x D
    # N x 16
    # N x 4 x D
    # N x 4

    X_tr, X_tt, X_tr_str, X_tt_str, y_tr, y_tt, y_tr_str, y_tt_str = load_vect(
        './data/bert_ft_vect.npy', './data/bert_ft.npy', './data/bert_lb_vect.npy', './data/bert_lb.npy'
    )

    # visualize_vectors.show_indiv(optimize_day(reduce_dim_day(X_tr[83], y_tr[83], 8, True)), X_tr_str[83])
    
    correct = {i:0 for i in range(4)}

    for i in range(len(X_tt)):

        transformed_inputs = optimize_day(reduce_dim_day(X_tt[i], y_tt[i], 8, True))

        # To normalize, uncomment below line
        # transformed_inputs = preprocessing.normalize(transformed_inputs)

        # Right now, the inputs are correctly ordered, we will shuffle the inputs
        shuffled_index = np.arange(16)
        np.random.shuffle(shuffled_index)
        transformed_inputs = transformed_inputs[shuffled_index]
        feature_strs = X_tt_str[i][shuffled_index]

        # Perform clustering to get guesses and check the guesses
        # You can switch the clustering algs here
        guesses = cluster.hca(transformed_inputs, feature_strs)
        right = check(guesses, X_tt_str[i])

        correct = {i:correct[i] + right[i] for i in range(4)}

        print(i)
        print(correct)

    correct = {i:correct[i] / len(X_tt) for i in range(4)}
    print(correct)
    print(sum(correct.values()) / 4)
        