import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import json
import os
from itertools import combinations
from .graph import save_dataframe_as_tiff

np.random.seed(666)

class SOM(object):
    def __init__(self, dataset, output, iteration, batch_size, checkpoint_step, output_dir, output_dir_name):
        """
        :param X:  形狀是N*D， 輸入樣本有N個,每個D維
        :param output: (n,m)一個元組，爲輸出層的形狀是一個n*m的二維矩陣
        :param iteration:迭代次數
        :param batch_size:每次迭代時的樣本數量
        初始化一個權值矩陣，形狀爲D*(n*m)，即有n*m權值向量，每個D維
        """
        self.dataset = dataset
        self.output = output
        self.iteration = iteration
        self.batch_size = batch_size
        self.W = np.random.rand(dataset.X.shape[1], output[0] * output[1])
        self.checkpoint_step = checkpoint_step
        self.output_dir = output_dir
        self.output_dir_name = output_dir_name

    def GetN(self, t):
        """
        :param t:時間t, 這裏用迭代次數來表示時間
        :return: 返回一個整數，表示拓撲距離，時間越大，拓撲鄰域越小
        """
        a = min(self.output)
        return int( a - float(a) * t / self.iteration )

    def Geteta(self, t, n):
        """
        :param t: 時間t, 這裏用迭代次數來表示時間
        :param n: 拓撲距離
        :return: 返回學習率，
        """
        return np.power(np.e, -n)/(t+2)

    def updata_W(self, X, t, winner):
        N = self.GetN(t)
        for x, i in enumerate(winner):
            to_update = self.getneighbor(i, N)
            for j in range(N+1):
                e = self.Geteta(t, j)
                for w in to_update[j]:
                    self.W[:, w] = np.add(self.W[:,w], e*(X[x,:] - self.W[:,w]))

    def getneighbor(self, index, N):
        """
        :param index:獲勝神經元的下標
        :param N: 鄰域半徑
        :return ans: 返回一個集合列表，分別是不同鄰域半徑內需要更新的神經元座標
        """
        a, b = self.output
        length = a*b
        def distence(index1, index2):
            i1_a, i1_b = index1 // a, index1 % b
            i2_a, i2_b = index2 // a, index2 % b
            return np.abs(i1_a - i2_a), np.abs(i1_b - i2_b)

        ans = [set() for i in range(N+1)]
        for i in range(length):
            dist_a, dist_b = distence(i, index)
            if dist_a <= N and dist_b <= N: ans[max(dist_a, dist_b)].add(i)
        return ans

    def train(self):
        """
        train_Y:訓練樣本與形狀爲batch_size*(n*m)
        winner:一個一維向量，batch_size個獲勝神經元的下標
        :return:返回值是調整後的W
        """
        for epoch in tqdm(range(self.iteration), desc="Training", unit="epoch"):
            for step in tqdm(range(self.dataset.X.shape[0] // self.batch_size), desc="Updating", unit="step"):
                start_idx = step * self.batch_size
                end_idx = (step + 1) * self.batch_size
                train_X = self.dataset.X[start_idx:end_idx]
                normal_W(self.W)
                normal_X(train_X)
                train_Y = train_X.dot(self.W)
                winner = np.argmax(train_Y, axis=1).tolist()
                self.updata_W(train_X, epoch, winner)
            
            if epoch != 0 and epoch % self.checkpoint_step == 0:
        
                self.export_training_result(
                    os.path.join(self.output_dir, self.output_dir_name),
                    f'cluster_{epoch}.csv'
                )
                width = 500
                height = 500
                output_path = os.path.join(self.output_dir, self.output_dir_name, f'raster_{epoch}.tiff')
                save_dataframe_as_tiff(
                    os.path.join(
                        self.output_dir, 
                        self.output_dir_name, 
                        f'cluster_{epoch}.csv'
                    ),
                    width, height, output_path
                )
                print(f"===== experiment results exported | epoch {epoch} =====")

        return self.W

    def export_training_result(self, path, clustered_name):

        # normalize each columns
        N, D = self.dataset.X.shape
        for i in tqdm(range(N), desc="Normalizing", unit="row"):
            temp = np.sum(np.multiply(self.dataset.X[i], self.dataset.X[i]))
            self.dataset.X[i] /= np.sqrt(temp)

        # feed forward
        train_Y = self.dataset.X.dot(self.W)
        winner = np.array([np.argmax(train_Y, axis=1).tolist()]).transpose()

        # encode cluster
        encoder = LabelEncoder()
        encoded_winner = np.array([encoder.fit_transform(winner.flatten())]).transpose()
        
        # merge normalized data with original data
        merged = np.hstack((self.dataset.X_origin, self.dataset.X, encoded_winner))
        columns = [f"x{i+1}" for i in range(0, self.dataset.X.shape[1])] + [f"x{i+1}_normalized" for i in range(0, self.dataset.X.shape[1])] + ['cluster']
        
        # export clustered data
        df = pd.DataFrame(merged, columns=columns).rename_axis("id")
        df.to_csv(os.path.join(path, clustered_name), encoding="utf-8")


def export_combination(columns, categories, df, group, path):
    for i in tqdm(range(len(columns)), desc="Exporting", unit="column"):
        comb = columns[i]
        columnX = comb[0]
        columnY = comb[1]
        for category in categories:
            x1 = df[df['cluster'] == category][columnX].tolist()
            x2 = df[df['cluster'] == category][columnY].tolist()
            combined = list(map(list, zip(x1, x2)))
            group[int(category)]=combined

        with open(os.path.join(path, f'{columnX}_{columnY}.json'), 'w') as json_file:
            json.dump(group, json_file, indent=2)

def generate_c3_2(out_columns_origin):
    # 檢查原始列是否至少包含3個元素
    # if len(out_columns_origin) < 3:
    #     print("原始列至少需要包含3個元素")
    #     return
    
    # 生成C3取2的所有組合
    c3_2_combinations = list(combinations(out_columns_origin, 2))
    
    return c3_2_combinations

def normal_X(X):
    """
    :param X:二維矩陣，N*D，N個D維的數據
    :return: 將X歸一化的結果
    """
    N, D = X.shape
    for i in range(N):
        temp = np.sum(np.multiply(X[i], X[i]))
        X[i] /= np.sqrt(temp)
    return X
def normal_W(W):
    """
    :param W:二維矩陣，D*(n*m)，D個n*m維的數據
    :return: 將W歸一化的結果
    """
    for i in range(W.shape[1]):
        temp = np.sum(np.multiply(W[:,i], W[:,i]))
        W[:, i] /= np.sqrt(temp)
    return W

# #畫圖
# def draw(C):
#     colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
#     for i in range(len(C)):
#         coo_X = []    #x座標列表
#         coo_Y = []    #y座標列表
#         for j in range(len(C[i])):
#             coo_X.append(C[i][j][0])
#             coo_Y.append(C[i][j][1])
#         pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i%len(colValue)], label=i)

#     pl.legend(loc='upper right')
#     pl.show()

