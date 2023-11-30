import pandas as pd
from scipy.io import arff

class Dataset():

    # self.X 會經過正規化的資料集
    # self.X_origin 原始資料

    def __init__(
        self, dataset_path: str
    ):
        if ".csv" in dataset_path:
            self.loadCSV(dataset_path)
        elif ".arff" in dataset_path:
            self.loadAARF(dataset_path)
    
    def loadAARF(self, dataset_path):
        data, meta = arff.loadarff(dataset_path)

        # 將ARFF資料轉換成Pandas DataFrame
        df = pd.DataFrame(data)
        self.X = df.drop('CLASS', axis=1).values
        self.X_origin = df.drop('CLASS', axis=1).values

        # print(self.X)
        # print(self.X_origin)

    def loadCSV(self, dataset_path):
        # load data using pandas
        data = pd.read_csv(dataset_path, dtype='float64')

        # transform pandas dataframe to numpy 2D array
        if 'id' in data.columns:
            self.X = data.drop('id', axis=1).values
            self.X_origin = data.drop('id', axis=1).values
        else:
            self.X = data.values
            self.X_origin = data.values


