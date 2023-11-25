from src.model import SOM, draw
import numpy as np

if __name__ == '__main__':

    #數據集：每三個是一組分別是西瓜的編號，密度，含糖量
    data = """
    1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
    6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
    11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
    16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
    21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
    26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459
    
    """

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
    
    print(classify)
    C = []#未歸一化的數據分類結果
    D = []#歸一化的數據分類結果
    for i in classify.values():
        C.append(dataset_old[i].tolist())
        D.append(dataset[i].tolist())
    draw(C)
    draw(D)
    