from src.graph import create_mp4
import sys

# 讀取環境變數與模型超參數
input_path = sys.argv[sys.argv.index('--input') + 1]
output_path = sys.argv[sys.argv.index('--output') + 1]

if __name__ == "__main__":

    # 轉換為Matplotlib圖片並保存，並製作MP4動畫，每3秒換一張
    create_mp4(input_path, output_path, fps=0.5)