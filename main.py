import os
from src.model import SOM
from src.dataset import Dataset

os.path.exists(r'./data')
os.path.exists(r'./data')

if __name__ == '__main__':


    dataset = Dataset(r'./data/sample.csv')
    dataset_original = dataset.X.copy()

    som = SOM(dataset, (5, 5), 1, 30)
    som.train()
    output = som.train_result()        
    output.to_csv('./output/dada.csv', encoding="utf-8")
    print(output)