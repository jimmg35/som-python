from src.model import SOM, draw
import numpy as np
import pandas as pd

if __name__ == '__main__':

    #數據集：每三個是一組分別是西瓜的編號，密度，含糖量
    data = pd.read_csv(r"./data/rgbn.csv")

    a = data.split(',')
    dataset = np.mat([[float(a[i]), float(a[i+1])] for i in range(1, len(a)-1, 3)])
    dataset_old = dataset.copy()

    som = SOM(dataset, (5, 5), 1, 30)
    som.train()
    res = som.train_result()
    classify = {}
    for i, win in enumerate(res):
        if not classify.get(win[0]):
            classify.setdefault(win[0], [i])
        else:
            classify[win[0]].append(i)
    C = []#未歸一化的數據分類結果
    D = []#歸一化的數據分類結果
    for i in classify.values():
        C.append(dataset_old[i].tolist())
        D.append(dataset[i].tolist())
    draw(C)
    draw(D)
    