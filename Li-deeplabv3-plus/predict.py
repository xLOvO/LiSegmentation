import os
import time
from tqdm import tqdm
from PIL import Image
from deeplab import DeeplabV3

if __name__ == "__main__":
    deeplab = DeeplabV3()
    mode = "dir_predict"
    count           = False
    name_classes    = ["background","Feldspar", "Quartz", "Lepidolite"]
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    dir_origin_path = "val/"
    dir_save_path   = "img_out/"
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"
    if mode == "dir_predict":
        total_time = 0.0
        frame_cnt = 0
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                start_time = time.time()
                r_image = deeplab.detect_image(image)
                end_time = time.time()
                total_time += (end_time - start_time)
                frame_cnt += 1
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
        if frame_cnt > 0:
            avg_fps = frame_cnt / total_time
            print(f"\nTotal Images: {frame_cnt}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Average FPS: {avg_fps:.2f}")
        else:
            print("Error！")
    else:
        raise AssertionError("Error!")