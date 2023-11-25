from src.model import SOM, draw
from src.dataset import Dataset
import numpy as np

if __name__ == '__main__':


    dataset = Dataset(r'./data/sample.csv')
    dataset_original = dataset.X.copy()

    som = SOM(dataset, (5, 5), 1, 30)
    som.train()
    res = som.train_result()
    classify = {}
    for i, win in enumerate(res):
        if not classify.get(win):
            classify.setdefault(win, [i])
        else:
            classify[win].append(i)
    
    C = []#未歸一化的數據分類結果
    D = []#歸一化的數據分類結果
    for i in classify.values():
        C.append(dataset_original[i].tolist())
        D.append(dataset.X[i].tolist())
    draw(C)
    draw(D)
    