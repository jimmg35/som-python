import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class SOM(object):
    def __init__(self, dataset, output, iteration, batch_size):
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
        print(f"output dimension: {self.W.shape} \n")

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
        count = 0
        while self.iteration > count:
            train_X = self.dataset.X[np.random.choice(self.dataset.X.shape[0], self.batch_size)]
            normal_W(self.W)
            normal_X(train_X)
            train_Y = train_X.dot(self.W)
            winner = np.argmax(train_Y, axis=1).tolist()
            self.updata_W(train_X, count, winner)
            count += 1
        return self.W

    def train_result(self):
        normal_X(self.dataset.X)
        train_Y = self.dataset.X.dot(self.W)
        winner = np.array([np.argmax(train_Y, axis=1).tolist()]).transpose()
        encoder = LabelEncoder()
        encoded_winner = np.array([encoder.fit_transform(winner.flatten())]).transpose()
        merged = np.hstack((self.dataset.X_origin, self.dataset.X, encoded_winner))
        columns = [f"x{i+1}" for i in range(0, self.dataset.X.shape[1])] + [f"x{i+1}_normalized" for i in range(0, self.dataset.X.shape[1])] + ['cluster']
        df = pd.DataFrame(merged, columns=columns).rename_axis("id")
        return df

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

