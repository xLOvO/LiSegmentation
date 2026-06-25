import os
import random

import numpy as np
import torch
from PIL import Image


def cvtColor(image):
    """Convert grayscale or other PIL modes to RGB before model preprocessing."""
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    return image.convert("RGB")


def resize_image(image, size):
    """Resize with unchanged aspect ratio and pad with gray pixels."""
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def get_lr(optimizer):
    """Return the current learning rate from the first optimizer group."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def seed_everything(seed=11):
    """Fix common random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, rank, seed):
    """Give each DataLoader worker a deterministic seed."""
    worker_seed = rank + seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def preprocess_input(image):
    """Normalize RGB image data from [0, 255] to [0, 1]."""
    image /= 255.0
    return image


def show_config(**kwargs):
    print("Configurations:")
    print("-" * 70)
    print("|%25s | %40s|" % ("keys", "values"))
    print("-" * 70)
    for key, value in kwargs.items():
        print("|%25s | %40s|" % (str(key), str(value)))
    print("-" * 70)


def download_weights(backbone, model_dir="./model_data"):
    """Download official backbone weights used when pretrained=True."""
    from torch.hub import load_state_dict_from_url

    download_urls = {
        "mobilenet": "https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar",
        "xception": "https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth",
    }
    if backbone not in download_urls:
        raise ValueError(f"Unsupported backbone: {backbone}")

    os.makedirs(model_dir, exist_ok=True)
    load_state_dict_from_url(download_urls[backbone], model_dir)
