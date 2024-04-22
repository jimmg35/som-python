import os
import sys
import logging
from src.model import SOM
from datetime import datetime
from src.dataset import Dataset

# 讀取環境變數與模型超參數
dataset_name = sys.argv[sys.argv.index('--dataset') + 1]
EPOCH = int(sys.argv[sys.argv.index('--epoch') + 1])
BATCH_SIZE = int(sys.argv[sys.argv.index('--batch') + 1])
WIDTH = int(sys.argv[sys.argv.index('--width') + 1])
HEIGHT = int(sys.argv[sys.argv.index('--height') + 1])
clustered_name = 'cluster.csv'
checkpoint_step = 5

# 建立輸出路徑
current_timestamp = int(datetime.timestamp(datetime.now()))
data_dir = r'./data'
output_dir = r'./cluster-result'
output_dir_name = f"{str(current_timestamp)}_som"
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
    som = SOM(
        dataset, 
        (HEIGHT, WIDTH), 
        EPOCH, 
        BATCH_SIZE, 
        checkpoint_step,
        output_dir, output_dir_name
    )
    logging.info(f'Initializing SOM model, hyperparameters provided below')
    logging.info(f'Epoch : {EPOCH}')
    logging.info(f'Batch size : {BATCH_SIZE}')
    logging.info(f'Output dimension : ({HEIGHT}, {WIDTH}) {escape}')
    
    # Start training model
    som.train()
    logging.info(f'Finished training{escape}')

    # Get the clustered data (pandas DataFrame)
    logging.info(f'Exporting clustered data')
    group = som.export_training_result(
        os.path.join(output_dir, output_dir_name),
        clustered_name
    )
    logging.info(f'Cluster result output at {os.path.join(output_dir, output_dir_name, clustered_name)}{escape}')

    # Log cluster info
    logging.info(f'We have {len(group)} clusters')
    for category in list(group.keys()):
        logging.info(f'Counts of cluster #{category} : {len(group[category])}')

    logging.info(f'{escape}')
    logging.info(f'Finished')


    # 假設 som 是你已經實例化並訓練好的 SOM 物件
    # visualize_weights(som, filename="som_weights.png")