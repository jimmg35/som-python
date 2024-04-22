from src.graph import create_mp4

if __name__ == "__main__":
    # 指定TIFF影像所在的路徑
    input_path = r'./cluster-result/1703816192_som'

    # 指定輸出動畫的路徑
    output_path = "./cluster-result/1703816192_som"

    # 轉換為Matplotlib圖片並保存，並製作MP4動畫，每3秒換一張
    create_mp4(input_path, output_path, fps=0.5)