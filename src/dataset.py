import pandas as pd
from scipy.io import arff

class Dataset():

    # self.X 會經過正規化的資料集
    # self.X_origin 原始資料

    def __init__(
        self, dataset_path: str
    ):
        if ".csv" in dataset_path and "apr" in dataset_path:
            self.loadAprCSV(dataset_path)
        elif ".csv" in dataset_path:
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

    def loadAprCSV(self, dataset_path):
        # load data using pandas
        data = pd.read_csv(dataset_path)
        data = data.loc[:, ['price', 'unitPrice',
                             'buildingArea', 'landTransferArea',
                             'buildingTransferArea', 'buildingType']]
        data = data.dropna()

        self.X = data.values
        self.X_origin = data.values


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
        
        print(self.X.shape)



# [
#     'OID_', 'none', 'town_raw', 'transactionTarget_raw', 'address_raw',
#     'landTransferArea_raw', 'urbanLandUse_raw', 'nonUrbanLandUse_raw',
#     'nonUrbanLandUsePlanning_raw', 'transactionTime_raw',
#     'transactionAmount_raw', 'transferFloor_raw', 'floor_raw',
#     'buildingType_raw', 'usage_raw', 'buildingMaterial_raw',
#     'completionTime_raw', 'buildingTransferArea_raw', 'roomNumber_raw',
#     'hallNumber_raw', 'bathNumber_raw', 'hasCompartment_raw',
#     'hasCommittee_raw', 'price_raw', 'unitPrice_raw',
#     'parkingSpaceType_raw', 'parkingSpaceTransferArea_raw',
#     'parkingSpacePrice_raw', 'hasNotes_raw', 'notes_raw', 'id_raw',
#     'buildingArea_raw', 'subBuildingArea_raw', 'belconyArea_raw',
#     'hasElevator_raw', 'address', 'transactionTime', 'completionTime',
#     'floor', 'hasElevator', 'hasCommittee', 'hasCompartment',
#     'buildingTransferArea', 'price', 'unitPrice',
#     'parkingSpaceTransferArea', 'parkingSpacePrice', 'landTransferArea',
#     'roomNumber', 'hallNumber', 'bathNumber', 'buildingArea',
#     'subBuildingArea', 'belconyArea', 'landAmount', 'buildingAmount',
#     'parkAmount', 'id', 'urbanLandUse', 'nonUrbanLandUse',
#     'nonUrbanLandUsePlanning', 'usage', 'transferFloorRaw', 'buildingType',
#     'parkingSpaceType', 'priceWithoutParking', 'coordinate_x',
#     'coordinate_y'
# ]