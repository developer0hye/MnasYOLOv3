import torch
from torch import nn

import torchvision.ops as ops
import torchvision.models as models

import numpy as np

import tools

class YOLO(nn.Module):
    def __init__(self, num_classes,
                 anchors_wh=None, anchors_min_wh=None, anchors_max_wh=None):
        super(YOLO, self).__init__()

        self.anchors_wh = anchors_wh  # [number_of_anchor boxes, 0 to 1], 0 = width, 1 = height
        self.anchors_min_wh = anchors_min_wh
        self.anchors_max_wh = anchors_max_wh

        self.num_anchors = len(self.anchors_wh)
        self.num_classes = num_classes

        mean_shift = torch.tensor(0.01, dtype=torch.float).repeat(self.num_anchors)
        self.mean_shift = nn.Parameter(mean_shift)

        scale_width = torch.tensor(1.0, dtype=torch.float).repeat(self.num_anchors)
        scale_width = scale_width.view(1, 1, 1, self.num_anchors, 1)
        self.scale_width = nn.Parameter(scale_width)

        scale_height = torch.tensor(1.0, dtype=torch.float).repeat(self.num_anchors)
        scale_height = scale_height.view(1, 1, 1, self.num_anchors, 1)
        self.scale_height = nn.Parameter(scale_height)

    def forward(self, x):
        b, h, w, a, o = x.shape

        grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1)
        grid_xy = grid_xy.view(1, h, w, 1, 2).to(x.device)

        anchors_wh = self.anchors_wh.view(1, 1, 1, self.num_anchors, 2).to(x.device)
        anchors_min_wh = self.anchors_min_wh.view(1, 1, 1, self.num_anchors, 2).to(x.device)
        anchors_max_wh = self.anchors_max_wh.view(1, 1, 1, self.num_anchors, 2).to(x.device)

        # scale_width = self.scale_width.view(1, 1, 1, self.num_anchors, 1).to(x.device)
        # scale_height = self.scale_height.view(1, 1, 1, self.num_anchors, 1).to(x.device)

        # print(scale)
        # print(self.scale_width[..., 0])
        # print(self.scale_height[..., 0])

        x[..., [0, 1]] = grid_xy + torch.sigmoid(x[..., [0, 1]])
        x[..., 0] = x[..., 0] / w
        x[..., 1] = x[..., 1] / h

        #x[..., [2, 3]] = anchors_wh * torch.exp(x[..., [2, 3]])

        x[..., 2] = anchors_min_wh[..., 0] + self.scale_width[..., 0] * (anchors_max_wh[..., 0]-anchors_min_wh[..., 0]) * torch.sigmoid(x[..., 2])
        x[..., 3] = anchors_min_wh[..., 1] + self.scale_height[..., 0] * (anchors_max_wh[..., 1]-anchors_min_wh[..., 1]) * torch.sigmoid(x[..., 3])

        x[..., 4] = torch.sigmoid(x[..., 4])
        x[..., 5:] = torch.sigmoid(x[..., 5:]) if self.num_classes > 1 else 1.0

        x = x.view(b, -1, o)
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1, leakyReLU=False):
        super(Conv2d, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True) if leakyReLU else nn.ReLU(inplace=True))

    def forward(self, x):
        return self.convs(x)

class MnasYOLOv3(nn.Module):
    def __init__(self,
                 num_classes=20,
                 anchors_wh=[[0.0587, 0.1009], [0.0868, 0.2519], [0.1682, 0.1418],
                     [0.1731, 0.3962], [0.3792, 0.3031], [0.2924, 0.6362],
                     [0.7266, 0.4697], [0.5062, 0.777], [0.8672, 0.8544]],
                 anchors_min_wh=[[0.008,  0.011], [0.026,  0.132], [0.092,  0.030],
                             [0.082,  0.220], [0.232,  0.082], [0.126,  0.398],
                             [0.421,  0.149], [0.336,  0.512], [0.620,  0.568]],
                 anchors_max_wh=[[0.150,  0.224], [0.154,  0.776], [0.614,  0.244],
                             [0.278, 0.992], [0.998,  0.485], [0.438,  0.998],
                             [0.998,  0.692], [0.704,  0.998], [0.998,  0.998]],
                 anchors_mask=[[0, 1, 2], [3, 4, 5], [6, 7, 8]]):
        super(MnasYOLOv3, self).__init__()

        self.num_classes = num_classes
        self.anchors_wh = torch.tensor(anchors_wh, dtype=torch.float32)
        self.anchors_min_wh = torch.tensor(anchors_min_wh, dtype=torch.float32)
        self.anchors_max_wh = torch.tensor(anchors_max_wh, dtype=torch.float32)

        self.anchors_mask = torch.tensor(anchors_mask)
        self.anchor_number = 3
        self.strides = [8, 16, 32]

        self.backbone = models.mnasnet1_0(pretrained=True).layers[:-3]

        self.yolo_layers = nn.ModuleList([])
        for anchor_mask in self.anchors_mask:
            self.yolo_layers.append(YOLO(num_classes=self.num_classes,
                                         anchors_wh=self.anchors_wh[anchor_mask],
                                        anchors_min_wh=self.anchors_min_wh[anchor_mask],
                                        anchors_max_wh=self.anchors_max_wh[anchor_mask]))

        # s = 32
        self.conv_set_3 = nn.Sequential(
            Conv2d(320, 256, 1, leakyReLU=True),
            Conv2d(256, 512, 3, padding=1, leakyReLU=True),
            Conv2d(512, 256, 1, leakyReLU=True),
            Conv2d(256, 512, 3, padding=1, leakyReLU=True),
            Conv2d(512, 256, 1, leakyReLU=True)
        )

        self.conv_1x1_3 = Conv2d(256, 256, 1, leakyReLU=True)
        self.extra_conv_3 = Conv2d(256, 512, 3, padding=1, leakyReLU=True)
        self.pred_3 = nn.Conv2d(512, self.yolo_layers[0].num_anchors * (4 + 1 + self.num_classes), 1)

        # s = 16
        self.conv_set_2 = nn.Sequential(
            Conv2d(352, 128, 1, leakyReLU=True),
            Conv2d(128, 256, 3, padding=1, leakyReLU=True),
            Conv2d(256, 128, 1, leakyReLU=True),
            Conv2d(128, 256, 3, padding=1, leakyReLU=True),
            Conv2d(256, 128, 1, leakyReLU=True)
        )
        self.conv_1x1_2 = Conv2d(128, 128, 1, leakyReLU=True)
        self.extra_conv_2 = Conv2d(128, 256, 3, padding=1, leakyReLU=True)
        self.pred_2 = nn.Conv2d(256, self.yolo_layers[1].num_anchors * (4 + 1 + self.num_classes), 1)

        # s = 8
        self.conv_set_1 = nn.Sequential(
            Conv2d(168, 64, 1, leakyReLU=True),
            Conv2d(64, 128, 3, padding=1, leakyReLU=True),
            Conv2d(128, 64, 1, leakyReLU=True),
            Conv2d(64, 128, 3, padding=1, leakyReLU=True),
            Conv2d(128, 64, 1, leakyReLU=True)
        )
        self.extra_conv_1 = Conv2d(64, 128, 3, padding=1, leakyReLU=True)
        self.pred_1 = nn.Conv2d(128, self.yolo_layers[2].num_anchors * (4 + 1 + self.num_classes), 1)

        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        # backbone
        fmps = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 9 or i == 11 or i == 13:
                fmps.append(x)

        fmp_1, fmp_2, fmp_3 = fmps

        # detection head
        # multi scale feature map fusion
        fmp_3 = self.conv_set_3(fmp_3)
        fmp_3_up = self.upsampling(self.conv_1x1_3(fmp_3))

        fmp_2 = torch.cat([fmp_2, fmp_3_up], 1)
        fmp_2 = self.conv_set_2(fmp_2)
        fmp_2_up = self.upsampling(self.conv_1x1_2(fmp_2))

        fmp_1 = torch.cat([fmp_1, fmp_2_up], 1)
        fmp_1 = self.conv_set_1(fmp_1)

        # head
        # s = 32
        fmp_3 = self.extra_conv_3(fmp_3)
        pred_3 = self.pred_3(fmp_3)

        # s = 16
        fmp_2 = self.extra_conv_2(fmp_2)
        pred_2 = self.pred_2(fmp_2)

        # s = 8
        fmp_1 = self.extra_conv_1(fmp_1)
        pred_1 = self.pred_1(fmp_1)

        pred_all_yolo_layer_inputs = []
        pred_all_xywhoc_bboxes = []

        for pred, yolo_layer in zip([pred_1, pred_2, pred_3], self.yolo_layers):
            b, c, h, w = pred.shape

            pred = pred.permute(0, 2, 3, 1).contiguous()

            pred_yolo_layer_input = pred.clone().view(b, h * w * yolo_layer.num_anchors, -1)
            pred_all_yolo_layer_inputs.append(pred_yolo_layer_input)

            pred_xywhoc_bboxes = pred.clone().view(b, h, w, yolo_layer.num_anchors, -1)
            pred_all_xywhoc_bboxes.append(yolo_layer(pred_xywhoc_bboxes))

        pred_all_yolo_layer_inputs = torch.cat(pred_all_yolo_layer_inputs, dim=1)
        pred_all_xywhoc_bboxes = torch.cat(pred_all_xywhoc_bboxes, dim=1)

        if self.training:
            return pred_all_yolo_layer_inputs, pred_all_xywhoc_bboxes
        else:
            with torch.no_grad():
                pred_all_xyxy_bboxes = tools.xywh2xyxy(pred_all_xywhoc_bboxes[..., :4])
                pred_all_o = pred_all_xywhoc_bboxes[..., 4]
                pred_all_c_conf, pred_all_c_idx = torch.max(pred_all_xywhoc_bboxes[..., 5:], dim=2)
                pred_all_oc_conf = pred_all_o * pred_all_c_conf

                pred_all_xyxy_bboxes = pred_all_xyxy_bboxes.to('cpu')
                pred_all_oc_conf = pred_all_oc_conf.to('cpu')
                pred_all_c_idx = pred_all_c_idx.to('cpu')

                oc_conf_thresh = 0.001
                keep_bboxes = pred_all_oc_conf > oc_conf_thresh

                pred_all_xyxy_bboxes = pred_all_xyxy_bboxes[keep_bboxes]
                pred_all_oc_conf = pred_all_oc_conf[keep_bboxes]
                pred_all_c_idx = pred_all_c_idx[keep_bboxes]

                nms_iou_thresh = 0.5
                nms_pred_all_xyxy_bboxes = []
                nms_pred_all_oc_conf = []
                nms_pred_all_c_idx = []

                for c_idx in range(self.num_classes):
                    c_inds = pred_all_c_idx[:] == c_idx

                    pred_c_all_xyxy_bboxes = pred_all_xyxy_bboxes[c_inds]
                    pred_c_all_oc_conf = pred_all_oc_conf[c_inds]
                    pred_c_all_c_idx = pred_all_c_idx[c_inds]

                    keep_bboxes = ops.nms(pred_c_all_xyxy_bboxes, pred_c_all_oc_conf, nms_iou_thresh)

                    nms_pred_all_xyxy_bboxes.append(pred_c_all_xyxy_bboxes[keep_bboxes])
                    nms_pred_all_oc_conf.append(pred_c_all_oc_conf[keep_bboxes])
                    nms_pred_all_c_idx.append(pred_c_all_c_idx[keep_bboxes])

                nms_pred_all_xyxy_bboxes = torch.cat(nms_pred_all_xyxy_bboxes, 0).numpy()
                nms_pred_all_oc_conf = torch.cat(nms_pred_all_oc_conf, 0).numpy()
                nms_pred_all_c_idx = torch.cat(nms_pred_all_c_idx, 0).numpy()

                return pred_all_yolo_layer_inputs,\
                       nms_pred_all_xyxy_bboxes,\
                       nms_pred_all_oc_conf, \
                       nms_pred_all_c_idx

    def yololoss(self,
                 preds,
                 preds_bbox,
                 targets,
                 bbox_w=2.0,
                 obj_pos_w=5.0,
                 obj_neg_w=1.0):

        p_bbox_xy = preds[..., [0, 1]]
        p_bbox_wh = preds_bbox[..., [2, 3]]
        p_obj = torch.sigmoid(preds[..., 4])
        p_class = preds[..., 5:]

        t_bbox_xy = targets[..., [0, 1]]
        t_bbox_wh = targets[..., [2, 3]]
        t_obj = targets[..., 4]
        t_class = targets[..., 5:]

        loss_xy_func = nn.BCEWithLogitsLoss(reduction='none')
        loss_wh_func = nn.MSELoss(reduction='none')
        loss_obj_func = nn.MSELoss(reduction='none')
        loss_class_func = nn.BCEWithLogitsLoss(reduction='none')

        loss_xy = bbox_w * torch.mean(torch.sum(torch.sum(loss_xy_func(p_bbox_xy, t_bbox_xy), 2) * t_obj, 1))
        loss_wh = bbox_w * torch.mean(torch.sum(torch.sum(loss_wh_func(p_bbox_wh, t_bbox_wh), 2) * t_obj, 1))

        t_obj_pos = t_obj
        t_obj_neg = 1. - t_obj

        loss_obj_pos = obj_pos_w * torch.mean(torch.sum(loss_obj_func(p_obj, t_obj)*t_obj_pos, 1))
        loss_obj_neg = obj_neg_w * torch.mean(torch.sum(loss_obj_func(p_obj, t_obj)*t_obj_neg, 1))

        loss_class = torch.mean(torch.sum(torch.sum(loss_class_func(p_class, t_class), 2) * t_obj, 1))
        if self.num_classes <= 1:
            loss_class = torch.tensor(0.)

        loss = loss_xy + loss_wh + loss_obj_pos + loss_obj_neg + loss_class
        return loss


if __name__ == '__main__':
    device = "cuda"
    model = MnasYOLOv3().to(device)
    model.train()

    import cv2
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    img = cv2.imread("000012.jpg").astype(np.float32)
    img = cv2.resize(img, (416, 416))/255.
    img = img[..., (2, 1, 0)]
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = normalize(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    input_size = img.shape[2:4]  # (h, w)

    bboxes_label = np.asarray([[6, 0.50, 0.55, 0.39, 0.52]], dtype=np.float32)
    bboxes_label = torch.from_numpy(bboxes_label)
    bboxes_label_list = [bboxes_label]

    epochs = 1000
    lr = 1e-5

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    
    for epoch in range(epochs):
        model.train()

        pred_all_yolo_layer_inputs, pred_all_xywhoc_bboxes = model(img)

        targets = tools.build_targets(model,
                                      bboxes_label_list,
                                      batch_size=1,
                                      input_size=(416, 416),
                                      dtype=img.dtype)
        targets = targets.to(device)

        loss = model.yololoss(pred_all_yolo_layer_inputs, targets)
        print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            #https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/43
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

            pred_all_yolo_layer_inputs, \
            nms_pred_all_xyxy_bboxes, \
            nms_pred_all_oc_conf, \
            nms_pred_all_c_idx = model(img)
            # targets = build_targets(model, pred_all_yolo_layer_inputs, bboxes_label_list, input_size)
            # targets = targets.to(device)
            #
            # loss = yololoss(model, pred_all_yolo_layer_inputs, targets)
            # print("eval:", loss)