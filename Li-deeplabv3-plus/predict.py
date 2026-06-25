import argparse
import os
import time

from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3


IMAGE_EXTENSIONS = (".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")
DEFAULT_CLASSES = ["background", "Feldspar", "Quartz", "Lepidolite"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepLabV3+ semantic segmentation on a directory.")
    parser.add_argument("--model-path", default="logs/best_epoch_weights.pth", help="Path to trained .pth weights.")
    parser.add_argument("--input", default="val", help="Directory containing input images.")
    parser.add_argument("--output", default="img_out", help="Directory used to save prediction images.")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU inference.")
    parser.add_argument("--mix-type", type=int, default=0, choices=[0, 1, 2],
                        help="0: blend mask and image, 1: mask only, 2: keep foreground only.")
    parser.add_argument("--count", action="store_true", help="Print pixel counts for each class.")
    return parser.parse_args()


def predict_directory(deeplab, input_dir, output_dir, count=False):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    total_time = 0.0
    frame_cnt = 0

    for img_name in tqdm(os.listdir(input_dir)):
        if not img_name.lower().endswith(IMAGE_EXTENSIONS):
            continue

        image_path = os.path.join(input_dir, img_name)
        image = Image.open(image_path)

        start_time = time.time()
        result = deeplab.detect_image(image, count=count, name_classes=DEFAULT_CLASSES)
        total_time += time.time() - start_time
        frame_cnt += 1

        result.save(os.path.join(output_dir, img_name))

    if frame_cnt == 0:
        print(f"No supported images found in {input_dir}.")
        return

    avg_fps = frame_cnt / total_time if total_time > 0 else 0
    print(f"\nTotal Images: {frame_cnt}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    args = parse_args()
    deeplab = DeeplabV3(
        model_path=args.model_path,
        cuda=not args.no_cuda,
        mix_type=args.mix_type,
    )
    predict_directory(deeplab, args.input, args.output, count=args.count)
