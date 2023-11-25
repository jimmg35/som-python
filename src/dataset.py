import pandas as pd

class Dataset():

    def __init__(
        self, dataset_path: str
    ):
        # load data using pandas
        data = pd.read_csv(dataset_path)

        # transform pandas dataframe to numpy 2D array
        self.X = data.drop('id', axis=1).values
