import os
import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn
from augmentation import *
import torchvision.transforms as transforms

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
                 img_size=(416, 416)):

        files = os.listdir(path)

        img_exts = [".png", ".jpg", ".bmp"]
        label_exts = [".txt"]

        self.imgs = [os.path.join(path, file) for file in files if file.endswith(tuple(img_exts))]
        self.labels = [os.path.join(path, file) for file in files if file.endswith(tuple(label_exts))]

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.img_size = img_size
        assert len(self.imgs) == len(self.labels), "영상의 갯수와 라벨 파일의 갯수가 서로 맞지 않습니다."

    def __getitem__(self, idx):
        assert Path(self.imgs[idx]).stem == Path(self.labels[idx]).stem, "영상과 어노테이션 파일의 짝이 맞지 않습니다."

        img = cv2.imread(self.imgs[idx], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0])).astype(np.float32)
        label = read_annotation_file(self.labels[idx])

        classes, bboxes_xywh = label[:, 0:1], label[:, 1:]

        classes = torch.from_numpy(classes).type(torch.float32)
        bboxes_xywh = torch.from_numpy(bboxes_xywh).type(torch.float32)

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

# class YOLODataset(Dataset):
#     def __init__(self, path, img_size=(416, 416), use_augmentation=True):
#         files = os.listdir(path)
#
#         img_exts = [".png", ".jpg", ".bmp"]
#         label_exts = [".txt"]
#
#         self.imgs = [os.path.join(path, file) for file in files if file.endswith(tuple(img_exts))]
#         self.labels = [os.path.join(path, file) for file in files if file.endswith(tuple(label_exts))]
#         self.use_augmentation = use_augmentation
#         self.aug_p = 0.0
#
#         self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#
#         self.img_size = img_size
#         self.multiscale_img_size = [self.img_size]
#
#         if self.use_augmentation:
#             for d in range(1, 8):
#                 if d == 0:
#                     continue
#                 self.multiscale_img_size.append([img_size[0]+64*d, img_size[1]+64*d])
#             self.multiscale_img_size = np.asarray(self.multiscale_img_size)
#             print(self.multiscale_img_size)
#
#         assert len(self.imgs) == len(self.labels), "영상의 갯수와 라벨 파일의 갯수가 서로 맞지 않습니다."
#
#     def __getitem__(self, idx):
#         assert Path(self.imgs[idx]).stem == Path(self.labels[idx]).stem, "영상과 어노테이션 파일의 짝이 맞지 않습니다."
#
#         img = cv2.imread(self.imgs[idx], cv2.IMREAD_COLOR)
#         resized_img, scaled_xyxy_label = resize_img(img,
#                                                     target_resize=self.img_size,
#                                                     normalized_xywh_label=open_label_file(self.labels[idx]),
#                                                     use_letterbox=False,
#                                                     padded_value=128)
#
#         img_h, img_w = resized_img.shape[0], resized_img.shape[1]
#
#         if self.use_augmentation:
#             if scaled_xyxy_label.shape[0] > 0:
#                 # label shuffling 부분 만들어주기 why? 학습시킬 때 한 그리드에 두 물체 있으면 한 물체만 찾도록 training 돼서
#                 # shuffle 시켜서 골고루 학습되게 만들어 줘야 됨
#                 np.random.shuffle(scaled_xyxy_label)
#                 aug_resized_img = resized_img.copy()
#                 aug_scaled_xyxy_label = scaled_xyxy_label.copy()
#
#                 aug_resized_img, aug_scaled_xyxy_label = horFlip(aug_resized_img,
#                                                                  aug_scaled_xyxy_label,
#                                                                  p=self.aug_p)
#                 #
#                 aug_resized_img, aug_scaled_xyxy_label = randomScale(aug_resized_img,
#                                                                      aug_scaled_xyxy_label,
#                                                                      scale=[-0.25, 0.5],
#                                                                      p=self.aug_p,
#                                                                      padded_val=128)
#
#                 #Translation 은 training 데이터셋에 대해 학습 수렴 속도를 늦추지 않음
#                 aug_resized_img, aug_scaled_xyxy_label = randomTranslation(aug_resized_img,
#                                                                            aug_scaled_xyxy_label,
#                                                                            p=self.aug_p,
#                                                                            padded_val=128)
#
#                 # aug_resized_img, aug_scaled_xyxy_label = randomShear(aug_resized_img,
#                 #                                                      aug_scaled_xyxy_label,
#                 #                                                      shear_degree=7.0,
#                 #                                                      p=self.aug_p)
#                 #
#                 # aug_resized_img, aug_scaled_xyxy_label = randomRotation(aug_resized_img,
#                 #                                                         aug_scaled_xyxy_label,
#                 #                                                         angle_degree=7.0,
#                 #                                                         p=self.aug_p)
#
#                 aug_resized_img = hsvColorSpaceJitter(aug_resized_img,
#                                                       hGain=0.05,
#                                                       sGain=0.2,
#                                                       vGain=0.1,
#                                                       p=self.aug_p)
#
#                 nL = len(aug_scaled_xyxy_label)  # number of labels
#                 if nL:
#                     resized_img = aug_resized_img
#                     scaled_xyxy_label = aug_scaled_xyxy_label
#
#                     # convert xyxy to xywh
#                     # drawBBox(aug_resized_img, aug_scaled_xyxy_label)
#                     # cv2.imwrite(str(idx) +str(self.img_size[0]) +".png", aug_resized_img)
#
#         # BGR to RGB, to 3x416x416
#         resized_img = resized_img.astype(np.float32)
#         resized_img = resized_img[:, :, ::-1].transpose(2, 0, 1)  # 416x416x3 to 3x416x416
#         resized_img = torch.from_numpy(np.ascontiguousarray(resized_img / 255.))
#         resized_img = self.normalize(resized_img)
#
#         normalized_xywh_label = xyxyToxywh(ScaledToNormalizedCoord(scaled_xyxy_label, img_w, img_h))
#         normalized_xywh_label = torch.from_numpy(normalized_xywh_label)
#
#         return resized_img, normalized_xywh_label
#
#     def __len__(self):
#         return len(self.imgs)
#
# def yolo_collate(datas):
#     imgs = []
#     labels = []
#     for data in datas:
#         img, label = data
#         imgs.append(img)
#         labels.append(label)
#     return torch.stack(imgs, 0), labels
