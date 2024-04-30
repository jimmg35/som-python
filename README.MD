：

## 指令使用方式

### 多張 TIFF 結合為 CSV 檔

`python raster-to-csv.py --data ./tiff-data/linkou --output ./data/linkou.csv`

### 訓練 SOM

`python som-fit.py --dataset linkou.csv --epoch 100000 --batch 30 --width 4 --height 4 --image_width 500 --image_height 500`

### 使用公開套件的 SOM 進行訓練

`python som-fit-sklearn.py --dataset linkou.csv --epoch 100 --batch 30 --width 4 --height 4 --image_width 500 --image_height 500`

### SOM 分群後的 TIFF 結合為影片檔

`python convert-to-video.py --input ./cluster-result/1703816192_som --output ./cluster-result/1703816192_som`

### 針對 SOM 分群結果進行敘述性統計

`python describe-cluster.py --input `
