import numpy as np
import matplotlib.pyplot as plt

def visualize_weights(som, filename=None):
    """
    將SOM的權重視覺化
    :param som: 已經訓練好的Self-Organizing Map物件
    :param filename: 圖片保存的檔案名稱（如果想要保存的話）
    """
    weights = som.W
    n_rows, n_cols = som.output

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axs[i, j]
            weight_vector = weights[:, i * n_cols + j]
            ax.imshow(weight_vector.reshape((som.dataset.X.shape[1], 1)), cmap='viridis', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
