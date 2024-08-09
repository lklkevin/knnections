import numpy as np
import matplotlib.pyplot as plt
import pca


def show(input_vects, input_words, index, labels_vects=None, labels=None):
    vects = np.load(input_vects)
    words = np.load(input_words)[index]

    results, _ = pca.apply_pca(vects[index], 2)
    colors = [i // 4 for i in range(16)]

    plt.figure(figsize=(10, 8))

    if labels_vects and labels:
        lb_vects = np.load(labels_vects)[index]
        lbs = np.load(labels)[index]
        lb_results, _ = pca.apply_pca(lb_vects, 2)
        cols = [0, 1, 2, 3]
        plt.scatter(lb_results[:, 0], lb_results[:, 1], marker=',', c=cols, cmap='viridis_r')

        for i, word in enumerate(lbs):
            plt.annotate(word, (lb_results[i, 0], lb_results[i, 1]), fontsize=8)

    plt.scatter(results[:, 0], results[:, 1], c=colors, cmap='viridis_r')

    for i, word in enumerate(words):
        plt.annotate(word, (results[i, 0], results[i, 1]))

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Plot of Words in ' + input_vects)
    plt.show()


if __name__ == "__main__":
    show('glv.npy', 'glv_actual.npy', 0)
    show('w2v.npy', 'w2v_actual.npy', 0)
    show('fst.npy', 'fst_actual.npy', 0)
    show('bert_ft_vect.npy', 'bert_ft.npy', 1, 'bert_lb_vect.npy', 'bert_lb.npy')
    show('gpt_ft_vect.npy', 'gpt_ft.npy', 1, 'gpt_lb_vect.npy', 'gpt_lb.npy')
    show('bert_reduced_ft.npy', 'bert_ft.npy', 1, 'bert_reduced_lb.npy', 'bert_lb.npy')
    show('gpt_reduced_ft.npy', 'gpt_ft.npy', 1, 'gpt_reduced_lb.npy', 'gpt_lb.npy')