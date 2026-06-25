import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class DeeplabDataset(Dataset):
    """VOC-style segmentation dataset.

    Expected structure:
        VOCdevkit/VOC2007/JPEGImages/{image_id}.jpg
        VOCdevkit/VOC2007/SegmentationClass/{image_id}.png
        VOCdevkit/VOC2007/ImageSets/Segmentation/{train,val}.txt
    """

    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(DeeplabDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        name = self.annotation_lines[index].split()[0]
        image_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages", name + ".jpg")
        mask_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass", name + ".png")

        jpg = Image.open(image_path)
        png = Image.open(mask_path)
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)

        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png)

        # Values outside class range are treated as ignore_index during loss calculation.
        png[png >= self.num_classes] = self.num_classes
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape(
            (int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1)
        )

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        """Apply letterbox resize for validation and random augmentation for training."""
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new("RGB", [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new("L", [w, h], 0)
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        # Random resize with aspect-ratio jitter.
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # Random horizontal flip.
        if self.rand() < .5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # Paste on a fixed-size canvas. Negative offsets crop oversized augmented images.
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_label = Image.new("L", (w, h), 0)
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image_data = np.array(new_image, np.uint8)
        label = new_label

        if self.rand() < 0.25:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        if self.rand() < 0.25:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            matrix = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data = cv2.warpAffine(
                image_data, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128)
            )
            label = cv2.warpAffine(
                np.array(label, np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
            )

        # HSV color jitter only changes the image, never the segmentation mask.
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue_data, sat_data, val_data = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((
            cv2.LUT(hue_data, lut_hue),
            cv2.LUT(sat_data, lut_sat),
            cv2.LUT(val_data, lut_val),
        ))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


def deeplab_dataset_collate(batch):
    """Merge images, class masks, and one-hot masks into tensors."""
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
