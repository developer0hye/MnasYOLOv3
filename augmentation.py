import cv2
import numpy as np
from numpy import random

def xywh2xyxy(box_xywh):
    box_xyxy = box_xywh.copy()
    box_xyxy[..., 0] = box_xywh[..., 0] - box_xywh[..., 2] / 2.
    box_xyxy[..., 1] = box_xywh[..., 1] - box_xywh[..., 3] / 2.
    box_xyxy[..., 2] = box_xywh[..., 0] + box_xywh[..., 2] / 2.
    box_xyxy[..., 3] = box_xywh[..., 1] + box_xywh[..., 3] / 2.
    return box_xyxy


def xyxy2xywh(box_xyxy):
    box_xywh = box_xyxy.copy()
    box_xywh[..., 0] = (box_xyxy[..., 0] + box_xyxy[..., 2]) / 2.
    box_xywh[..., 1] = (box_xyxy[..., 1] + box_xyxy[..., 3]) / 2.
    box_xywh[..., 2] = box_xyxy[..., 2] - box_xyxy[..., 0]
    box_xywh[..., 3] = box_xyxy[..., 3] - box_xyxy[..., 1]
    return box_xywh

def PhotometricNoise(img_bgr, #type must be float
                     h_delta=18.,
                     s_gain=0.5,
                     brightness_delta=32):
    if random.randint(2):

        if random.randint(2):
            brightness_delta = np.random.uniform(-brightness_delta, brightness_delta)
            img_bgr += brightness_delta
            img_bgr = np.clip(img_bgr, 0., 255.)

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # h[0, 359], s[0, 1.0], v[0, 255.]
        h_delta = np.random.uniform(-h_delta, h_delta)
        s_gain = np.random.uniform(1. - s_gain, 1. + s_gain)

        img_hsv[..., 0] = np.clip(img_hsv[..., 0] + h_delta, 0., 359.)
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] * s_gain, 0., 1.0)

        img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        return img_bgr
    return img_bgr

def HorFlip(img, bboxes_xywh):
    if random.randint(2):
        img = cv2.flip(img, 1)#1이 호리즌탈 방향 반전
        bboxes_xywh[:, 0] = 1. - bboxes_xywh[:, 0]
        return img, bboxes_xywh
    return img, bboxes_xywh


def RandomTranslation(img, bboxes_xyxy, classes):
    if random.randint(2):

        height, width = img.shape[0:2]
        max_iteration = 50

        img_org = img.copy()
        bboxes_xyxy_org = bboxes_xyxy.copy()

        for _ in range(max_iteration):
            img = img_org.copy()
            bboxes_xyxy = bboxes_xyxy_org.copy()

            tx = np.random.randint(-width, width)
            ty = np.random.randint(-height, height)

            #translation matrix
            tm = np.float32([[1, 0, tx],
                             [0, 1, ty]])  # [1, 0, tx], [1, 0, ty]

            img = cv2.warpAffine(img, tm, (width, height), borderValue=(127, 127, 127))

            tx /= width
            ty /= height

            bboxes_xyxy[:, [0, 2]] += tx
            bboxes_xyxy[:, [1, 3]] += ty

            clipped_bboxes_xyxy = np.clip(bboxes_xyxy, 0., 1.)

            clipped_bboxes_w = clipped_bboxes_xyxy[:, 2] - clipped_bboxes_xyxy[:, 0]
            clipped_bboxes_h = clipped_bboxes_xyxy[:, 3] - clipped_bboxes_xyxy[:, 1]

            valid_bboxes_inds = (clipped_bboxes_w > 1e-6) & (clipped_bboxes_h > 1e-6)

            if np.sum(valid_bboxes_inds) == 0:
                continue

            clipped_bboxes_xyxy = clipped_bboxes_xyxy[valid_bboxes_inds]
            clipped_bboxes_w = clipped_bboxes_w[valid_bboxes_inds]
            clipped_bboxes_h = clipped_bboxes_h[valid_bboxes_inds]

            bboxes_xyxy = bboxes_xyxy[valid_bboxes_inds]

            bboxes_w = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]
            bboxes_h = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]

            occlusion_proportion_w = 1. - (clipped_bboxes_w / bboxes_w)
            occlusion_proportion_h = 1. - (clipped_bboxes_h / bboxes_h)

            if np.sum(occlusion_proportion_w > 0.4) > 0 or np.sum(occlusion_proportion_h > 0.4) > 0:
                continue
            else:
                classes = classes[valid_bboxes_inds]
                return img, clipped_bboxes_xyxy, classes
        return img_org, bboxes_xyxy_org, classes

    return img, bboxes_xyxy, classes

def RandomScale(img, bboxes_xyxy, classes, scale=[-0.25, 0.25]):
    if random.randint(2):

        height, width = img.shape[0:2]
        max_iteration = 50

        img_org = img.copy()
        bboxes_xyxy_org = bboxes_xyxy.copy()

        n_bboxes = len(bboxes_xyxy_org)

        for _ in range(max_iteration):
            img = img_org.copy()
            bboxes_xyxy = bboxes_xyxy_org.copy()
            random_scale = np.random.uniform(1. + scale[0], 1. + scale[1])

            sm = cv2.getRotationMatrix2D(angle=0., center=(width / 2, height / 2), scale=random_scale)
            img = cv2.warpAffine(img, sm, (width, height), borderValue=(127, 127, 127))

            sm[0, 2] /= width
            sm[1, 2] /= height

            h_bboxes_xy_tl = np.concatenate([bboxes_xyxy[:, [0, 1]], np.ones((n_bboxes, 1))], axis=-1)
            h_bboxes_xy_br = np.concatenate([bboxes_xyxy[:, [2, 3]], np.ones((n_bboxes, 1))], axis=-1)

            h_bboxes_xy_tl = sm @ h_bboxes_xy_tl.T
            h_bboxes_xy_br = sm @ h_bboxes_xy_br.T

            bboxes_xyxy[:, [0, 1]] = h_bboxes_xy_tl.T
            bboxes_xyxy[:, [2, 3]] = h_bboxes_xy_br.T

            clipped_bboxes_xyxy = np.clip(bboxes_xyxy, 0., 1.)

            clipped_bboxes_w = clipped_bboxes_xyxy[:, 2] - clipped_bboxes_xyxy[:, 0]
            clipped_bboxes_h = clipped_bboxes_xyxy[:, 3] - clipped_bboxes_xyxy[:, 1]

            valid_bboxes_inds = (clipped_bboxes_w > 1e-6) & (clipped_bboxes_h > 1e-6)

            if np.sum(valid_bboxes_inds) == 0:
                continue

            clipped_bboxes_xyxy = clipped_bboxes_xyxy[valid_bboxes_inds]
            clipped_bboxes_w = clipped_bboxes_w[valid_bboxes_inds]
            clipped_bboxes_h = clipped_bboxes_h[valid_bboxes_inds]

            bboxes_xyxy = bboxes_xyxy[valid_bboxes_inds]

            bboxes_w = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]
            bboxes_h = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]

            occlusion_proportion_w = 1. - (clipped_bboxes_w / bboxes_w)
            occlusion_proportion_h = 1. - (clipped_bboxes_h / bboxes_h)

            if np.sum(occlusion_proportion_w > 0.4) > 0 or np.sum(occlusion_proportion_h > 0.4) > 0:
                continue
            else:
                classes = classes[valid_bboxes_inds]
                return img, clipped_bboxes_xyxy, classes
        return img_org, bboxes_xyxy_org, classes
    return img, bboxes_xyxy, classes

def RandomCrop(img, bboxes_xyxy, classes):
    if random.randint(2):

        height, width = img.shape[0:2]
        max_iteration = 50

        img_org = img.copy()
        bboxes_xyxy_org = bboxes_xyxy.copy()

        for _ in range(max_iteration):
            img = img_org.copy()
            bboxes_xyxy = bboxes_xyxy_org.copy()

            bboxes_xyxy[:, [0, 2]] *= width
            bboxes_xyxy[:, [1, 3]] *= height

            w = random.randint(width//4, width)
            h = random.randint(height//4, height)

            # aspect ratio constraint b/t .5 & 2
            if h / w < 0.5 or h / w > 2:
                continue

            rect_left_top_x = random.randint(0, width - w + 1)
            rect_left_top_y = random.randint(0, height - h + 1)

            rect_right_bottom_x = rect_left_top_x + w
            rect_right_bottom_y = rect_left_top_y + h

            img = img[rect_left_top_y:rect_right_bottom_y, rect_left_top_x:rect_right_bottom_x, :]

            clipped_bboxes_xyxy = np.zeros_like(bboxes_xyxy)
            clipped_bboxes_xyxy[:, [0, 2]] = np.clip(bboxes_xyxy[:, [0, 2]], rect_left_top_x, rect_right_bottom_x)
            clipped_bboxes_xyxy[:, [1, 3]] = np.clip(bboxes_xyxy[:, [1, 3]], rect_left_top_y, rect_right_bottom_y)

            clipped_bboxes_w = clipped_bboxes_xyxy[:, 2] - clipped_bboxes_xyxy[:, 0]
            clipped_bboxes_h = clipped_bboxes_xyxy[:, 3] - clipped_bboxes_xyxy[:, 1]

            valid_bboxes_inds = (clipped_bboxes_w > 0) & (clipped_bboxes_h > 0)

            if np.sum(valid_bboxes_inds) == 0:
                continue

            clipped_bboxes_xyxy = clipped_bboxes_xyxy[valid_bboxes_inds]
            clipped_bboxes_w = clipped_bboxes_w[valid_bboxes_inds]
            clipped_bboxes_h = clipped_bboxes_h[valid_bboxes_inds]

            bboxes_xyxy = bboxes_xyxy[valid_bboxes_inds]

            bboxes_w = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]
            bboxes_h = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]

            occlusion_proportion_w = 1. - (clipped_bboxes_w / bboxes_w)
            occlusion_proportion_h = 1. - (clipped_bboxes_h / bboxes_h)

            if np.sum(occlusion_proportion_w > 0.4) > 0 or np.sum(occlusion_proportion_h > 0.4) > 0:
                continue
            else:
                classes = classes[valid_bboxes_inds]
                clipped_bboxes_xyxy[:, [0, 2]] -= rect_left_top_x
                clipped_bboxes_xyxy[:, [1, 3]] -= rect_left_top_y

                clipped_bboxes_xyxy[:, [0, 2]] /= w
                clipped_bboxes_xyxy[:, [1, 3]] /= h

                return img, clipped_bboxes_xyxy, classes
        return img_org, bboxes_xyxy_org, classes
    return img, bboxes_xyxy, classes


def drawBBox(img, bboxes_xyxy):
    h, w = img.shape[:2]

    bboxes_xyxy[:, [0, 2]] *= w
    bboxes_xyxy[:, [1, 3]] *= h

    for bbox_xyxy in bboxes_xyxy:
        cv2.rectangle(img, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), (0, 255, 0),2)

if __name__ == '__main__':

    while(True):
        img = cv2.imread("000021.jpg", cv2.IMREAD_COLOR).astype(np.float32)

        import dataset
        label = dataset.read_annotation_file("000021.txt")
        classes, bboxes_xywh = label[:, 0:1], label[:, 1:]

        img = PhotometricNoise(img)
        img, bboxes_xywh = HorFlip(img, bboxes_xywh)

        bboxes_xyxy = xywh2xyxy(bboxes_xywh)

        img, bboxes_xyxy, classes = RandomTranslation(img, bboxes_xyxy, classes)
        img, bboxes_xyxy, classes = RandomScale(img, bboxes_xyxy, classes)
        img, bboxes_xyxy, classes = RandomCrop(img, bboxes_xyxy, classes)


        if len(bboxes_xyxy) != len(classes):
            print("bbox랑 class 수랑 일치하지 않다. augmentation 과정에서 실수가 있는 게 분명해")

        img = cv2.resize(img, (416, 416)).astype(np.float32)
        drawBBox(img, bboxes_xyxy)
        img /= 255.
        cv2.imshow("img", img)
        ch = cv2.waitKey(0)

        if ch == 27:
            break