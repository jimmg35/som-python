import pandas as pd

class Dataset():

    # self.X 會經過正規化的資料集
    # self.X_origin 原始資料

    def __init__(
        self, dataset_path: str
    ):
        # load data using pandas
        data = pd.read_csv(dataset_path)

        # transform pandas dataframe to numpy 2D array
        self.X = data.drop('id', axis=1).values
        self.X_origin = data.drop('id', axis=1).values
