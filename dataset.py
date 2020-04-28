import os
import cv2
import numpy as np

from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn
import torchvision.transforms as transforms

import tools
from augmentation import *

def read_annotation_file(path):
    with open(path, 'r') as label:
        objects_information = []
        for line in label:
            line = line.split()
            if len(line) == 5:  # 0: class, 1:x, 2:y, 3:w, 4:h
                object_information = []
                for data in line:
                    object_information.append(float(data))
                objects_information.append(object_information)
        objects_information = np.asarray(objects_information).astype(np.float32)
        return objects_information

class YOLODataset(Dataset):
    def __init__(self,
                 path,
                 img_size=(416, 416),
                 use_augmentation=True):

        files = sorted(os.listdir(path))

        img_exts = [".png", ".jpg", ".bmp"]
        label_exts = [".txt"]

        self.imgs = [os.path.join(path, file) for file in files if file.endswith(tuple(img_exts))]
        self.labels = [os.path.join(path, file) for file in files if file.endswith(tuple(label_exts))]

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.img_size = img_size
        assert len(self.imgs) == len(self.labels), "영상의 갯수와 라벨 파일의 갯수가 서로 맞지 않습니다."

        self.use_augmentation = use_augmentation

    def __getitem__(self, idx):
        assert Path(self.imgs[idx]).stem == Path(self.labels[idx]).stem, "영상과 어노테이션 파일의 짝이 맞지 않습니다."

        img = cv2.imread(self.imgs[idx], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.img_size[0], self.img_size[1])).astype(np.float32)
        label = read_annotation_file(self.labels[idx])

        classes, bboxes_xywh = label[:, 0:1], label[:, 1:]

        if self.use_augmentation:
            img = PhotometricNoise(img)

        classes = torch.from_numpy(classes)
        bboxes_xywh = torch.from_numpy(bboxes_xywh)

        bboxes_label = torch.cat([classes, bboxes_xywh], dim=-1)

        # to rgb
        img = img[:, :, (2, 1, 0)]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img / 255.
        img = self.normalize(img)

        return img, bboxes_label

    def __len__(self):
        return len(self.imgs)

def yolo_collate(batch_data):
    imgs = []
    bboxes_label_list = []
    for img, bboxes_label in batch_data:
        imgs.append(img)
        bboxes_label_list.append(bboxes_label)
    return torch.stack(imgs, 0), bboxes_label_list