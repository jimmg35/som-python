from PIL import Image
import numpy as np
import sys
import pandas as pd
from os import listdir
import os


# 讀取環境變數與模型超參數
tiff_path = sys.argv[sys.argv.index('--data') + 1]
output_path = sys.argv[sys.argv.index('--output') + 1]

file_names = [filename for filename in listdir(tiff_path)]

out_columns = [filename.replace('.tif', '') for filename in listdir(tiff_path)]

print(file_names)
print(out_columns)

# 讀取 TIFF 檔案並轉換為 NumPy 陣列
arrays = [np.array(Image.open(os.path.join(tiff_path, file))) for file in file_names]

# 將陣列重新排列並堆疊
stacked_array = np.column_stack(tuple(arr.ravel() for arr in arrays))

array_2d = stacked_array[stacked_array[:, 0] != 0]

outCSV = pd.DataFrame(data = array_2d, columns = out_columns)

outCSV.to_csv(output_path, index=False)