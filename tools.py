import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from voc0712 import *

CLASS_COLOR = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in
               range(len(VOC_CLASSES))]
# We use ignore thresh to decide which anchor box can be kept.
ignore_thresh = 0.5

class LRScheduler(object):
    def __init__(self,
                 optimizer,
                 warmup_iter,
                 total_iter,
                 target_lr):

        self.optimizer = optimizer
        self.warmup_iter = warmup_iter
        self.total_iter = total_iter
        self.target_lr = target_lr

    def warmup_lr(self, cur_iter):
        warmup_lr = self.target_lr * float(cur_iter) / float(self.warmup_iter)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def cosine_decay_lr(self, cur_iter):

        Tcur = cur_iter - self.warmup_iter
        Tmax = self.total_iter - self.warmup_iter

        warmup_lr = 0.5 * self.target_lr * (1.+np.cos((Tcur/Tmax)*np.pi))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, cur_iter):
        if cur_iter <= self.warmup_iter:
            self.warmup_lr(cur_iter)
        else:
            self.cosine_decay_lr(cur_iter)

def xywh2xyxy(box_xywh):
    box_xyxy = box_xywh.clone()
    box_xyxy[..., 0] = box_xywh[..., 0] - box_xywh[..., 2] / 2.
    box_xyxy[..., 1] = box_xywh[..., 1] - box_xywh[..., 3] / 2.
    box_xyxy[..., 2] = box_xywh[..., 0] + box_xywh[..., 2] / 2.
    box_xyxy[..., 3] = box_xywh[..., 1] + box_xywh[..., 3] / 2.
    return box_xyxy


def xyxy2xywh(box_xyxy):
    box_xywh = box_xyxy.clone()
    box_xywh[..., 0] = (box_xyxy[..., 0] + box_xyxy[..., 2]) / 2.
    box_xywh[..., 1] = (box_xyxy[..., 1] + box_xyxy[..., 3]) / 2.
    box_xywh[..., 2] = box_xyxy[..., 2] - box_xyxy[..., 0]
    box_xywh[..., 3] = box_xyxy[..., 3] - box_xyxy[..., 1]
    return box_xywh

def iou_xyxy(boxA_xyxy, boxB_xyxy):
    # determine the (x, y)-coordinates of the intersection rectangle
    x11, y11, x12, y12 = torch.split(boxA_xyxy, 1, dim=1)
    x21, y21, x22, y22 = torch.split(boxB_xyxy, 1, dim=1)

    xA = torch.max(x11, x21.T)
    yA = torch.max(y11, y21.T)
    xB = torch.min(x12, x22.T)
    yB = torch.min(y12, y22.T)

    interArea = (xB - xA).clamp(0) * (yB - yA).clamp(0)
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)
    unionArea = (boxAArea + boxBArea.T - interArea)
    iou = interArea / (unionArea+1e-6)

    # return the intersection over union value
    return iou

def iou_xywh(boxA_xywh, boxB_xywh):
    boxA_xyxy = xywh2xyxy(boxA_xywh)
    boxB_xyxy = xywh2xyxy(boxB_xywh)

    # return the intersection over union value
    return iou_xyxy(boxA_xyxy, boxB_xyxy)

def whiou(boxA_wh, boxB_wh):
    # determine the (x, y)-coordinates of the intersection rectangle
    w1, h1 = torch.split(boxA_wh, 1, dim=1)
    w2, h2 = torch.split(boxB_wh, 1, dim=1)

    innerW = torch.min(w1, w2.T).clamp(0)
    innerH = torch.min(h1, h2.T).clamp(0)

    interArea = innerW * innerH
    boxAArea = (w1) * (h1)
    boxBArea = (w2) * (h2)
    iou = interArea / (boxAArea + boxBArea.T - interArea + 1e-6)

    # return the intersection over union value
    return iou

def build_targets(model,
                  bboxes_label_list,
                  batch_size,
                  input_size,
                  dtype):

    h, w = input_size
    o = (4 + 1 + model.num_classes)
    targets = []

    for stride, yolo_layer in zip(model.strides, model.yolo_layers):
        h_ = h // stride
        w_ = w // stride
        a_ = yolo_layer.num_anchors

        targets.append(torch.zeros((batch_size, h_, w_, a_, o), dtype=dtype))

    for idx_batch in range(batch_size):
        for bbox_label in bboxes_label_list[idx_batch]:
            c = int(bbox_label[0])
            bbox_x, bbox_y = bbox_label[[1, 2]]
            bbox_w, bbox_h = bbox_label[[3, 4]]

            bbox_xywh = bbox_label[1:].view(1, -1)

            # compute the IoU
            best_iou = 0.
            best_idx_layer = 0.
            best_idx_iou = 0.
            best_anchor = None

            for idx_layer, yolo_layer in enumerate(model.yolo_layers):
                anchor_wh = yolo_layer.anchors_wh

                iou = whiou(bbox_xywh[:, 2:], anchor_wh)
                iou, idx_iou = torch.max(iou, dim=-1)

                if iou > best_iou:
                    best_iou = iou
                    best_idx_layer = idx_layer
                    best_idx_iou = idx_iou
                    best_anchor = anchor_wh[best_idx_iou]

            grid_w = w // model.strides[best_idx_layer]
            grid_h = h // model.strides[best_idx_layer]

            bbox_x_on_grid = bbox_x * grid_w
            bbox_y_on_grid = bbox_y * grid_h

            idx_x = int(bbox_x_on_grid)
            idx_y = int(bbox_y_on_grid)

            tx = bbox_x_on_grid-idx_x
            ty = bbox_y_on_grid-idx_y
            tw = torch.log(bbox_w/best_anchor[0, 0])
            th = torch.log(bbox_h/best_anchor[0, 1])

            targets[best_idx_layer][idx_batch, idx_y, idx_x, best_idx_iou, [0, 1, 2, 3]] = torch.tensor([tx, ty, tw, th])
            targets[best_idx_layer][idx_batch, idx_y, idx_x, best_idx_iou, 4] = 1.0
            targets[best_idx_layer][idx_batch, idx_y, idx_x, best_idx_iou, 5 + c] = 1.0

    for i in range(len(targets)):
        targets[i] = targets[i].view(batch_size, -1, o)

    targets = torch.cat(targets, dim=1)

    return targets
