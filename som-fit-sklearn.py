import os
import sys
import logging
import numpy as np
from datetime import datetime
from sklearn_som.som import SOM
from src.dataset import Dataset
from src.graph import save_dataframe_as_tiff
from src.export import export_cluster_result_to_csv

# 讀取環境變數與模型超參數
dataset_name = sys.argv[sys.argv.index('--dataset') + 1]
EPOCH = int(sys.argv[sys.argv.index('--epoch') + 1])
BATCH_SIZE = int(sys.argv[sys.argv.index('--batch') + 1])
WIDTH = int(sys.argv[sys.argv.index('--width') + 1])
HEIGHT = int(sys.argv[sys.argv.index('--height') + 1])
IMG_WIDTH = int(sys.argv[sys.argv.index('--image_width') + 1])
IMG_HEIGHT = int(sys.argv[sys.argv.index('--image_height') + 1])
clustered_name = 'cluster.csv'
checkpoint_step = 5

# 建立輸出路徑
current_timestamp = int(datetime.timestamp(datetime.now()))
data_dir = r'./data'
output_dir = r'./cluster-result'
output_dir_name = f"{str(current_timestamp)}_som_sklearn"
if os.path.exists(data_dir) == False:
    os.mkdir(data_dir)

if os.path.exists(output_dir) == False:
    os.mkdir(output_dir)
os.mkdir(os.path.join(output_dir, output_dir_name))

# 設置logging (用於紀錄每次實驗的超參數)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=os.path.join(output_dir, output_dir_name, 'log.txt'), 
                    filemode='w')
logging.getLogger().addHandler(logging.StreamHandler())
escape = "\n\n============================= \n"


if __name__ == '__main__':
    
    # Loading dataset (must in csv format)
    dataset = Dataset(
        os.path.join(data_dir, dataset_name)
    )
    logging.info(f'{escape}')
    logging.info(f'Reading dataset named - {dataset_name} {escape}')
    

    # Loading hyperparameters
    som = SOM(m=HEIGHT, n=WIDTH, dim=dataset.X.shape[1])


    logging.info(f'Initializing SOM model, hyperparameters provided below')
    logging.info(f'Epoch : {EPOCH}')
    logging.info(f'Batch size : {BATCH_SIZE}')
    logging.info(f'Output dimension : ({HEIGHT}, {WIDTH}) {escape}')
    
    # Start training model
    
    for epoch in range(EPOCH):
        som.fit(dataset.X)
        if epoch != 0 and epoch % checkpoint_step == 0:
            labels = som.predict(dataset.X)
            # merge data with cluster result
            datasetWithClusterLabels = np.hstack((
                dataset.X, 
                np.array(list(labels)).reshape(-1, 1)
            ))
            export_cluster_result_to_csv(
                datasetWithClusterLabels,
                dataset.columns,
                os.path.join(output_dir, output_dir_name),
                f'cluster_{epoch}.csv'
            )
            save_dataframe_as_tiff(
                os.path.join(output_dir, output_dir_name),
                f'cluster_{epoch}.csv',
                IMG_WIDTH, 
                IMG_HEIGHT, 
            )

        logging.info(f'Epoch {epoch+1}/{EPOCH} finished')
        
    logging.info(f'Finished training{escape}')