from PIL import Image
import numpy as np
import pandas as pd

# 定義檔案路徑和檔案名稱
file_path = r'./data/'
file_names = ['red.tif', 'blue.tif', 'green.tif', 'nir.tif',
              'Var_red.tif', 'Var_blue.tif', 'Var_green.tif', 'Var_nir.tif',
              'Cont_red.tif', 'Cont_blue.tif', 'Cont_green.tif', 'Cont_nir.tif',
              'Corelation_red.tif', 'Corelation_blue.tif', 'Corelation_green.tif', 'Corelation_nir.tif',
              'Homo_red.tif', 'Homo_blue.tif', 'Homo_green.tif', 'Homo_nir.tif',
              'Var_red_90.tif', 'Var_blue_90.tif', 'Var_green_90.tif', 'Var_nir_90.tif',
              'Cont_red_90.tif', 'Cont_blue_90.tif', 'Cont_green_90.tif', 'Cont_nir_90.tif',
              'Corelation_red_90.tif', 'Corelation_blue_90.tif', 'Corelation_green_90.tif', 'Corelation_nir_90.tif',
              'Homo_red_90.tif', 'Homo_blue_90.tif', 'Homo_green_90.tif', 'Homo_nir_90.tif',
              ]

outColumns = ['red', 'blue', 'green', 'nir',
              'Var_red', 'Var_blue', 'Var_green', 'Var_nir',
              'Cont_red', 'Cont_blue', 'Cont_green', 'Cont_nir',
              'Corelation_red', 'Corelation_blue', 'Corelation_green', 'Corelation_nir',
              'Homo_red', 'Homo_blue', 'Homo_green', 'Homo_nir',
              'Var_red_90', 'Var_blue_90', 'Var_green_90', 'Var_nir_90',
              'Cont_red_90', 'Cont_blue_90', 'Cont_green_90', 'Cont_nir_90',
              'Corelation_red_90', 'Corelation_blue_90', 'Corelation_green_90', 'Corelation_nir_90',
              'Homo_red_90', 'Homo_blue_90', 'Homo_green_90', 'Homo_nir_90',
              ]

# 讀取 TIFF 檔案並轉換為 NumPy 陣列
arrays = [np.array(Image.open(file_path + file)) for file in file_names]

# 將陣列重新排列並堆疊
stacked_array = np.column_stack(tuple(arr.ravel() for arr in arrays))


outCSV = pd.DataFrame(data = stacked_array, columns = outColumns)

outCSV.to_csv('raster.csv', index=False)
print(outCSV)

# # 儲存成 CSV 檔案
# np.savetxt(file_path + 'output.csv', stacked_array, delimiter=',', fmt='%d')



# # 定義固定的路徑
# file_path = '/your/directory/path/'

# # 讀取 TIFF 檔案
# tiff1 = Image.open(file_path + 'tiff1.tiff')
# tiff2 = Image.open(file_path + 'tiff2.tiff')
# tiff3 = Image.open(file_path + 'tiff3.tiff')

# # 將 TIFF 轉換為 NumPy 陣列
# array1 = np.array(tiff1)
# array2 = np.array(tiff2)
# array3 = np.array(tiff3)

# # 將陣列重新排列並堆疊
# stacked_array = np.column_stack((array1.ravel(), array2.ravel(), array3.ravel()))

# # 儲存成 CSV 檔案
# np.savetxt(file_path + 'output.csv', stacked_array, delimiter=',', fmt='%d')

