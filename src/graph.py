import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw
import imageio
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def visualize_weights(som, filename=None):
    """
    將SOM的權重視覺化
    :param som: 已經訓練好的Self-Organizing Map物件
    :param filename: 圖片保存的檔案名稱（如果想要保存的話）
    """
    weights = som.W
    n_rows, n_cols = som.output

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axs[i, j]
            weight_vector = weights[:, i * n_cols + j]
            ax.imshow(weight_vector.reshape((som.dataset.X.shape[1], 1)), cmap='viridis', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])

    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def save_dataframe_as_tiff(data_path, width, height, output_path):
    dataframe = pd.read_csv(data_path)
    # 创建一个新的图像对象，使用"1"表示单波段
    img = Image.new('L', (width, height), color='white')
    pixels = img.load()

    # 将DataFrame的cluster列数据写入图像像素
    cluster_data = dataframe['cluster'].values.reshape((height, width))
    for i in range(height):
        for j in range(width):
            pixels[j, i] = int(cluster_data[i, j])

    # 保存图像文件
    img.save(output_path)


def visualize_tiff(input_path, output_path, filename, cmap, norm):
    # 讀取TIFF檔案為NumPy陣列
    image = imageio.imread(os.path.join(input_path, filename))

    # 設定標題為檔案名稱
    plt.title(filename)

    # 顯示不同整數值的像素為不同顏色
    plt.imshow(image, cmap=cmap, norm=norm)
    
    # 顯示類別資料的 colorbar
    cb = plt.colorbar(ticks=np.arange(len(cmap.colors)))
    cb.set_label('Class')

    # 保存Matplotlib圖片
    plt.savefig(os.path.join(output_path, f"{os.path.splitext(filename)[0]}.png"))
    plt.close()

def create_mp4(input_path, output_path, fps=3):
    images = []

    # 讀取指定路徑下的所有TIFF檔案，按數字順序排序
    tiff_files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.tiff', '.tif'))],
                        key=lambda x: int(''.join(filter(str.isdigit, x))))

    # 創建自定義的顏色地圖和顏色標準化器
    cmap = ListedColormap(['blue', 'green', 'red', 'purple', 'yellow', 'orange', 'brown', 'pink', 'gray'])
    boundaries = np.arange(-0.5, len(cmap.colors), 1)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    for filename in tiff_files:
        print(filename)
        # 將TIFF檔案視覺化並保存Matplotlib圖片
        visualize_tiff(input_path, output_path, filename, cmap, norm)

        # 將Matplotlib圖片讀取為NumPy陣列，用於之後的MP4製作
        image_path = os.path.join(output_path, f"{os.path.splitext(filename)[0]}.png")
        image = imageio.imread(image_path)
        images.append(image)

    # 輸出MP4動畫
    output_mp4_path = os.path.join(output_path, "output_animation.mp4")
    imageio.mimsave(output_mp4_path, images, fps=fps)




if __name__ == '__main__':

    data = pd.read_csv(r'./cluster-result/1703764982_som/cluster.csv')

    width = 1045
    height = 957

    output_path = 'output.tiff'

    save_dataframe_as_tiff(data, width, height, output_path)