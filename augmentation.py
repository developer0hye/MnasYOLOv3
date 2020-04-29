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
                     v_delta=32):
    if random.randint(2):
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # h[0, 359], s[0, 1.0], v[0, 255.]
        h_delta = np.random.uniform(-h_delta, h_delta)
        s_gain = np.random.uniform(1. - s_gain, 1. + s_gain)
        v_delta = np.random.uniform(-v_delta, v_delta)

        img_hsv[..., 0] = np.clip(img_hsv[..., 0] + h_delta, 0., 359.)
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] * s_gain, 0., 1.0)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + v_delta, 0., 255.0)

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
        max_iteration = 100

        img_org = img.copy()
        bboxes_xyxy_org = bboxes_xyxy.copy()

        for _ in range(max_iteration):
            img = img_org.copy()
            bboxes_xyxy = bboxes_xyxy_org.copy()

            tx = np.random.uniform(-width, width)
            ty = np.random.uniform(-height, height)

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

            valid_bboxes_inds = (clipped_bboxes_w > 0.01) & (clipped_bboxes_h > 0.01)
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

            if np.sum(occlusion_proportion_w > 0.3) > 0 or np.sum(occlusion_proportion_h > 0.3) > 0:
                continue
            else:
                classes = classes[valid_bboxes_inds]
                return img, clipped_bboxes_xyxy, classes
    return img, bboxes_xyxy, classes

    # if np.random.rand() < p:
    #     img_h, img_w = img.shape[0:2]
    #     augmented_img = img.copy()
    #     scaled_augmented_xyxy_label = scaled_xyxy_label.copy()
    #
    #     tx = np.floor(np.random.uniform(-img_w/2, img_w/2))
    #     ty = np.floor(np.random.uniform(-img_h/2, img_h/2))
    #
    #     #translation matrix
    #     tm = np.float32([[1, 0, tx],
    #                      [0, 1, ty]])  # [1, 0, tx], [1, 0, ty]
    #
    #     augmented_img = cv2.warpAffine(augmented_img, tm, (img_w, img_h), borderValue=(padded_val, padded_val, padded_val))
    #
    #     scaled_augmented_xyxy_label[:, [1,3]] += tx
    #     scaled_augmented_xyxy_label[:, [2,4]] += ty
    #
    #     cliped_scaled_augmented_xyxy_label, do_augmentation = checkApplyAugmentation(scaled_augmented_xyxy_label, img_w, img_h)
    #
    #     if do_augmentation:
    #         return augmented_img, cliped_scaled_augmented_xyxy_label
    #
    # return img, scaled_xyxy_label



def drawBBox(img, bboxes_xyxy):
    h, w = img.shape[:2]

    bboxes_xyxy[:, [0, 2]] *= w
    bboxes_xyxy[:, [1, 3]] *= h

    for bbox_xyxy in bboxes_xyxy:
        cv2.rectangle(img, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), (0, 255, 0),2)

if __name__ == '__main__':

    while(True):
        img = cv2.imread("000021.jpg", cv2.IMREAD_COLOR)
        img = cv2.resize(img, (416, 416)).astype(np.float32)

        import dataset
        label = dataset.read_annotation_file("000021.txt")
        classes, bboxes_xywh = label[:, 0:1], label[:, 1:]

        img = PhotometricNoise(img)
        img, bboxes_xywh = HorFlip(img, bboxes_xywh)

        bboxes_xyxy = xywh2xyxy(bboxes_xywh)
        img, bboxes_xyxy, classes = RandomTranslation(img, bboxes_xyxy, classes)




        drawBBox(img, bboxes_xyxy)
        img /= 255.
        cv2.imshow("img", img)
        ch = cv2.waitKey(0)

        if ch == 27:
            break



    #
    #     img_hsv[...,]
    #     img_hsv[..., [0, 1, 2]] *= [hGain, sGain, vGain]
    #     print(np.maximum(img_hsv[..., 0]))
    #     img_hsv[..., 0] = np.clip(img_hsv[..., 0], 0., 179.)
    #     img_hsv[..., [1, 2]] = np.clip(img_hsv[..., [1, 2]], 0., 255.)
    #
    #     img_hsv = img_hsv.astype(np.uint8)
    #     augmented_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    #     return augmented_img
    # return img


# def open_label_file(path):
#     with open(path, 'r') as label:
#         objects_information = []
#         for line in label:
#             line = line.split()
#             if len(line) == 5:  # 0: class, 1:x, 2:y, 3:w, 4:h
#                 object_information = []
#                 for data in line:
#                     object_information.append(float(data))
#                 objects_information.append(object_information)
#         objects_information = np.asarray(objects_information).astype(np.float32)
#         return objects_information
#
# def xywhToxyxy(xywh_label):
#     xyxy_label = xywh_label.copy()
#     xyxy_label[..., 1] = xyxy_label[..., 1] - (xyxy_label[..., 3] / 2.0)
#     xyxy_label[..., 2] = xyxy_label[..., 2] - (xyxy_label[..., 4] / 2.0)
#     xyxy_label[..., 3] = xyxy_label[..., 1] + xyxy_label[..., 3]
#     xyxy_label[..., 4] = xyxy_label[..., 2] + xyxy_label[..., 4]
#     xyxy_label[..., 1:5] = np.clip(xyxy_label[..., 1:5], 0., 1.)
#     return xyxy_label
#
# def xyxyToxywh(xyxy_label):
#     xywh_label = xyxy_label.copy()
#     xywh_label[..., 1] = (xyxy_label[..., 1] + xyxy_label[..., 3]) / 2.0
#     xywh_label[..., 2] = (xyxy_label[..., 2] + xyxy_label[..., 4]) / 2.0
#     xywh_label[..., 3] = xyxy_label[..., 3] - xyxy_label[..., 1]
#     xywh_label[..., 4] = xyxy_label[..., 4] - xyxy_label[..., 2]
#     return xywh_label
#
# def NormalizedToScaledCoord(normalized_label, img_w, img_h):
#     scaled_label = normalized_label.copy()
#     scaled_label[:, [1, 3]] *= img_w
#     scaled_label[:, [2, 4]] *= img_h
#
#     scaled_label[:, [1, 3]] = np.clip(scaled_label[:, [1, 3]], 0, img_w - 1)
#     scaled_label[:, [2, 4]] = np.clip(scaled_label[:, [2, 4]], 0, img_h - 1)
#
#     return scaled_label
#
# def ScaledToNormalizedCoord(scaled_label, img_w, img_h):
#     normalized_label = scaled_label.copy()
#     normalized_label[:, [1, 3]] /= img_w
#     normalized_label[:, [2, 4]] /= img_h
#
#     normalized_label[:, [1, 3]] = np.clip(normalized_label[:, [1, 3]], 0, 0.999)
#     normalized_label[:, [2, 4]] = np.clip(normalized_label[:, [2, 4]], 0, 0.999)
#
#     return normalized_label
#
# def CoordClip(xyxy_label, img_w, img_h):
#     cliped_xyxy_label = xyxy_label.copy()
#     cliped_xyxy_label[:, [1, 3]] = np.clip(cliped_xyxy_label[:, [1, 3]], 0., img_w-1)
#     cliped_xyxy_label[:, [2, 4]] = np.clip(cliped_xyxy_label[:, [2, 4]], 0., img_h-1)
#     return cliped_xyxy_label
#
#
# def getIndexesOuterBBoxes(scaled_xyxy_label, img_w, img_h):
#     indexes_outer_bbox = (scaled_xyxy_label[:, 1] < 0) & (scaled_xyxy_label[:, 3] < 0)
#     indexes_outer_bbox |= (scaled_xyxy_label[:, 1] > img_w-1) & (scaled_xyxy_label[:, 3] > img_w-1)
#     indexes_outer_bbox |= (scaled_xyxy_label[:, 2] < 0) & (scaled_xyxy_label[:, 4] < 0)
#     indexes_outer_bbox |= (scaled_xyxy_label[:, 2] > img_h-1) & (scaled_xyxy_label[:, 4] > img_h-1)
#     return indexes_outer_bbox
#
# def IoUbetweenWholeAndInnerBox(whole_bbox_label, inner_bbox_label):
#     bboxes_whole_area = (whole_bbox_label[:, 3] - whole_bbox_label[:, 1]) * (whole_bbox_label[:, 4] - whole_bbox_label[:, 2])
#     bboxes_inner_area = (inner_bbox_label[:, 3] - inner_bbox_label[:, 1]) * (inner_bbox_label[:, 4] - inner_bbox_label[:, 2])
#     ious_between_whole_and_inner = bboxes_inner_area / bboxes_whole_area
#     return ious_between_whole_and_inner
#
# def BBoxesArea(xyxy_label):
#     return (xyxy_label[:, 3] - xyxy_label[:, 1]) * (xyxy_label[:, 4] - xyxy_label[:, 2])
#
# def checkApplyAugmentation(scaled_augmented_xyxy_label, img_w, img_h, th_iou=0.9):
#     do_augmentation = True
#
#     # 이미지 외부에 있는 오브젝트들 삭제
#     indexesOuterBBoxes = getIndexesOuterBBoxes(scaled_augmented_xyxy_label, img_w, img_h)
#     scaled_augmented_xyxy_label = scaled_augmented_xyxy_label[indexesOuterBBoxes == False]
#     cliped_scaled_augmented_xyxy_label = CoordClip(scaled_augmented_xyxy_label, img_w, img_h)
#
#     if(len(cliped_scaled_augmented_xyxy_label) == 0):
#         return cliped_scaled_augmented_xyxy_label, False
#
#     # 너무 작은 오브젝트들 삭제
#     cliped_scaled_augmented_xywh_label = xyxyToxywh(cliped_scaled_augmented_xyxy_label)
#     indexesTooSmallBBoxesArea = (cliped_scaled_augmented_xywh_label[..., 3] <= 4) | (cliped_scaled_augmented_xywh_label[..., 4] <= 4)
#     cliped_scaled_augmented_xyxy_label = cliped_scaled_augmented_xyxy_label[indexesTooSmallBBoxesArea == False]
#     scaled_augmented_xyxy_label = scaled_augmented_xyxy_label[indexesTooSmallBBoxesArea == False]
#
#     if (len(cliped_scaled_augmented_xyxy_label) == 0):
#         return cliped_scaled_augmented_xyxy_label, False
#
#     # 어그먼테이션 적용되고 나서 일부분만 보이는 오브젝트들 삭제
#     ious = IoUbetweenWholeAndInnerBox(scaled_augmented_xyxy_label, cliped_scaled_augmented_xyxy_label)
#     indexesTooOccludedBBoxes = (ious <= th_iou)
#
#     if np.count_nonzero(indexesTooOccludedBBoxes) > 0:
#         do_augmentation = False
#
#     return cliped_scaled_augmented_xyxy_label, do_augmentation
#
# def hsvColorSpaceJitter(img, hGain=0.5, sGain=0.5, vGain=0.5, p=0.5):
#     if np.random.rand() < p:
#         augmented_img = img.copy()
#         img_hsv = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2HSV).astype(np.float32)
#
#         hGain = np.random.uniform(1. - hGain, 1. + hGain)
#         sGain = np.random.uniform(1. - sGain, 1. + sGain)
#         vGain = np.random.uniform(1. - vGain, 1. + vGain)
#
#         img_hsv[..., [0, 1, 2]] *= [hGain, sGain, vGain]
#         img_hsv[..., 0] = np.clip(img_hsv[..., 0], 0., 179.)
#         img_hsv[..., [1, 2]] = np.clip(img_hsv[..., [1, 2]], 0., 255.)
#
#         img_hsv = img_hsv.astype(np.uint8)
#         augmented_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
#         return augmented_img
#     return img
#
# def horFlip(img, scaled_xyxy_label, p=0.5):
#     if np.random.rand() < p:
#         img_h, img_w = img.shape[0:2]
#         augmented_img = cv2.flip(img, 1)#1이 호리즌탈 방향 반전
#         augmented_scaled_xyxy_label = scaled_xyxy_label.copy()
#         augmented_normalized_xyxy_label = ScaledToNormalizedCoord(augmented_scaled_xyxy_label, img_w, img_h)
#         augmented_normalized_xywh_label = xyxyToxywh(augmented_normalized_xyxy_label)
#         augmented_normalized_xywh_label[:, 1] = 1. - augmented_normalized_xywh_label[:, 1] # normalized xywh format
#
#         augmented_scaled_xyxy_label = NormalizedToScaledCoord(xywhToxyxy(augmented_normalized_xywh_label), img_w, img_h)
#
#         return augmented_img, augmented_scaled_xyxy_label
#     return img, scaled_xyxy_label
#
# def randomTranslation(img, scaled_xyxy_label, p=0.5, padded_val=127):
#     if np.random.rand() < p:
#         img_h, img_w = img.shape[0:2]
#         augmented_img = img.copy()
#         scaled_augmented_xyxy_label = scaled_xyxy_label.copy()
#
#         tx = np.floor(np.random.uniform(-img_w/2, img_w/2))
#         ty = np.floor(np.random.uniform(-img_h/2, img_h/2))
#
#         #translation matrix
#         tm = np.float32([[1, 0, tx],
#                          [0, 1, ty]])  # [1, 0, tx], [1, 0, ty]
#
#         augmented_img = cv2.warpAffine(augmented_img, tm, (img_w, img_h), borderValue=(padded_val, padded_val, padded_val))
#
#         scaled_augmented_xyxy_label[:, [1,3]] += tx
#         scaled_augmented_xyxy_label[:, [2,4]] += ty
#
#         cliped_scaled_augmented_xyxy_label, do_augmentation = checkApplyAugmentation(scaled_augmented_xyxy_label, img_w, img_h)
#
#         if do_augmentation:
#             return augmented_img, cliped_scaled_augmented_xyxy_label
#
#     return img, scaled_xyxy_label
#
# def randomRotation(img, scaled_xyxy_label, angle_degree=10., p=0.5, padded_val=127):
#     if np.random.rand() < p:
#         augmented_img = img.copy()
#         scaled_augmented_xyxy_label = scaled_xyxy_label.copy()
#
#         random_angle_degree = np.random.uniform(-angle_degree, angle_degree)
#         img_h, img_w = augmented_img.shape[:2]
#         n = len(scaled_augmented_xyxy_label)
#
#         rm = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), random_angle_degree, 1)
#         augmented_img = cv2.warpAffine(augmented_img, rm, (img_w, img_h), flags=cv2.INTER_AREA, borderValue=(padded_val, padded_val, padded_val))
#
#         bboxes_tl_br_point = scaled_augmented_xyxy_label[:, 1:5]
#         bboxes_tl_tr_bl_br_points = np.ones((n, 4, 3))
#
#         bboxes_tl_tr_bl_br_points[:, 0, [0, 1]] = bboxes_tl_br_point[:, [0, 1]]# tl
#         bboxes_tl_tr_bl_br_points[:, 1, [0, 1]] = bboxes_tl_br_point[:, [2, 1]]# tr
#         bboxes_tl_tr_bl_br_points[:, 2, [0, 1]] = bboxes_tl_br_point[:, [0, 3]]# bl
#         bboxes_tl_tr_bl_br_points[:, 3, [0, 1]] = bboxes_tl_br_point[:, [2, 3]]# br
#
#         #numpy dot 순서 공부해봐야겠다.
#         bboxes_tl_tr_bl_br_points = bboxes_tl_tr_bl_br_points.reshape((n * 4, 3))
#         bboxes_tl_tr_bl_br_points = bboxes_tl_tr_bl_br_points.dot(rm.T)
#         bboxes_tl_tr_bl_br_points = bboxes_tl_tr_bl_br_points.reshape((n, 4, 2))
#
#         scaled_augmented_xyxy_label[:, [1, 2]] = np.min(bboxes_tl_tr_bl_br_points, axis=1)
#         scaled_augmented_xyxy_label[:, [3, 4]] = np.max(bboxes_tl_tr_bl_br_points, axis=1)
#
#         cliped_scaled_augmented_xyxy_label, do_augmentation = checkApplyAugmentation(scaled_augmented_xyxy_label, img_w,
#                                                                                      img_h)
#         if do_augmentation:
#             return augmented_img, cliped_scaled_augmented_xyxy_label
#
#     return img, scaled_xyxy_label
#
# def randomShear(img, scaled_xyxy_label, shear_degree=7.5, p=0.5, padded_val=127):
#     if np.random.rand() < p:
#         augmented_img = img.copy()
#         scaled_augmented_xyxy_label = scaled_xyxy_label.copy()
#
#         img_h, img_w = augmented_img.shape[:2]
#         n = len(scaled_augmented_xyxy_label)
#
#         random_shear_hor = np.tanh(np.deg2rad(np.random.uniform(-shear_degree, shear_degree)))
#         random_shear_ver = np.tanh(np.deg2rad(np.random.uniform(-shear_degree, shear_degree)))
#
#         #shearing matrix 이건 또 반전안해도되네?
#         sm = np.float32([[1., random_shear_hor, -random_shear_hor * img_w / 2],
#                          [random_shear_ver, 1, -random_shear_ver * img_h / 2]])
#
#         augmented_img = cv2.warpAffine(augmented_img,
#                                        sm,
#                                        (img_w, img_h),
#                                        flags=cv2.INTER_AREA,
#                                        borderValue=(padded_val, padded_val, padded_val))
#
#         bboxes_tl_br_point = scaled_augmented_xyxy_label[:, 1:5]
#         bboxes_tl_tr_bl_br_points = np.ones((n, 4, 3))
#
#         bboxes_tl_tr_bl_br_points[:, 0, [0, 1]] = bboxes_tl_br_point[:, [0, 1]]  # tl
#         bboxes_tl_tr_bl_br_points[:, 1, [0, 1]] = bboxes_tl_br_point[:, [2, 1]]  # tr
#         bboxes_tl_tr_bl_br_points[:, 2, [0, 1]] = bboxes_tl_br_point[:, [0, 3]]  # bl
#         bboxes_tl_tr_bl_br_points[:, 3, [0, 1]] = bboxes_tl_br_point[:, [2, 3]]  # br
#
#         # numpy dot 순서 공부해봐야겠다.
#         bboxes_tl_tr_bl_br_points = bboxes_tl_tr_bl_br_points.reshape((n * 4, 3))
#         bboxes_tl_tr_bl_br_points = bboxes_tl_tr_bl_br_points.dot(sm.T)
#         bboxes_tl_tr_bl_br_points = bboxes_tl_tr_bl_br_points.reshape((n, 4, 2))
#
#         scaled_augmented_xyxy_label[:, [1, 2]] = np.min(bboxes_tl_tr_bl_br_points, axis=1)
#         scaled_augmented_xyxy_label[:, [3, 4]] = np.max(bboxes_tl_tr_bl_br_points, axis=1)
#
#         cliped_scaled_augmented_xyxy_label, do_augmentation = checkApplyAugmentation(scaled_augmented_xyxy_label, img_w,
#                                                                                      img_h)
#         if do_augmentation:
#             return augmented_img, cliped_scaled_augmented_xyxy_label
#
#     return img, scaled_xyxy_label
#
#
# def randomScale(img, scaled_xyxy_label, scale=[-0.25, 0.25], p=0.5, padded_val=127):
#     if np.random.rand() < p:
#         augmented_img = img.copy()
#         scaled_augmented_xyxy_label = scaled_xyxy_label.copy()
#         img_h, img_w = augmented_img.shape[:2]
#         n = len(scaled_augmented_xyxy_label)
#         random_scale = np.random.uniform(1. + scale[0], 1. + scale[1])
#
#         #scaling matrix_img
#         sm = cv2.getRotationMatrix2D(angle=0., center=(img_w / 2, img_h / 2), scale=random_scale) # 이미지 좌표 중심점 기준으로 축소 및 확대 + 0도 회전
#         augmented_img = cv2.warpAffine(augmented_img, sm, (img_w, img_h), flags= cv2.INTER_AREA,
#                                        borderValue=(padded_val, padded_val, padded_val))
#
#         bboxes_tl_br_point = scaled_augmented_xyxy_label[:, 1:5]
#         bboxes_tl_tr_bl_br_points = np.ones((n, 4, 3))
#
#         bboxes_tl_tr_bl_br_points[:, 0, [0, 1]] = bboxes_tl_br_point[:, [0, 1]]  # tl
#         bboxes_tl_tr_bl_br_points[:, 1, [0, 1]] = bboxes_tl_br_point[:, [2, 1]]  # tr
#         bboxes_tl_tr_bl_br_points[:, 2, [0, 1]] = bboxes_tl_br_point[:, [0, 3]]  # bl
#         bboxes_tl_tr_bl_br_points[:, 3, [0, 1]] = bboxes_tl_br_point[:, [2, 3]]  # br
#
#         # numpy dot 순서 공부해봐야겠다.
#         bboxes_tl_tr_bl_br_points = bboxes_tl_tr_bl_br_points.reshape((n * 4, 3))
#         bboxes_tl_tr_bl_br_points = bboxes_tl_tr_bl_br_points.dot(sm.T)
#         bboxes_tl_tr_bl_br_points = bboxes_tl_tr_bl_br_points.reshape((n, 4, 2))
#
#         scaled_augmented_xyxy_label[:, [1, 2]] = np.min(bboxes_tl_tr_bl_br_points, axis=1)
#         scaled_augmented_xyxy_label[:, [3, 4]] = np.max(bboxes_tl_tr_bl_br_points, axis=1)
#
#         cliped_scaled_augmented_xyxy_label, do_augmentation = checkApplyAugmentation(scaled_augmented_xyxy_label, img_w,
#                                                                                      img_h)
#         if do_augmentation:
#             return augmented_img, cliped_scaled_augmented_xyxy_label
#
#     return img, scaled_xyxy_label
#
# def drawBBox(img, scaled_xyxy_label):
#     for bbox in scaled_xyxy_label:
#         #print(bbox)
#         cv2.rectangle(img, (bbox[1], bbox[2]), (bbox[3], bbox[4]), (0,255,0),2)
#
# def resize_img(img, target_resize, normalized_xywh_label=None, use_letterbox=True, padded_value = (127, 127, 127)):
#     h, w = img.shape[0], img.shape[1]
#     target_h, target_w = target_resize[1], target_resize[0]
#
#     if use_letterbox:
#         if h > w:
#             h_scale = target_h / h
#             resized_w = np.clip(np.rint(w * h_scale).astype(np.uint32), 0, target_w)
#             resized_img = cv2.resize(img, (resized_w, target_h))
#             scaled_xyxy_label = NormalizedToScaledCoord(xywhToxyxy(normalized_xywh_label), resized_w, target_h)
#
#             padded_w = target_w - resized_w
#             padded_left = padded_w // 2
#             padded_right = padded_w - padded_left
#             resized_img = cv2.copyMakeBorder(resized_img,
#                                              0, 0, padded_left, padded_right,
#                                              cv2.BORDER_CONSTANT,
#                                              value=padded_value)
#
#             scaled_xyxy_label[:, [1, 3]] += padded_left
#         else:
#             w_scale = target_w / w
#             resized_h = np.clip(np.rint(h * w_scale).astype(np.uint32), 0, target_h)
#             resized_img = cv2.resize(img, (target_w, resized_h))
#             scaled_xyxy_label = NormalizedToScaledCoord(xywhToxyxy(normalized_xywh_label), target_w, resized_h)
#
#             padded_h = target_h - resized_h
#             padded_top = padded_h // 2
#             padded_bot = padded_h - padded_top
#             resized_img = cv2.copyMakeBorder(resized_img,
#                                              padded_top, padded_bot, 0, 0,
#                                              cv2.BORDER_CONSTANT,
#                                              value=padded_value)
#
#             scaled_xyxy_label[:, [2, 4]] += padded_top
#     else:
#         resized_img = cv2.resize(img, (target_w, target_h))
#         scaled_xyxy_label = NormalizedToScaledCoord(xywhToxyxy(normalized_xywh_label), target_w, target_h)
#
#     return resized_img, scaled_xyxy_label
#
#
# if __name__ == '__main__':
#     np.set_printoptions(precision=3, suppress =True)
#
#     while(True):
#         img = cv2.imread("000007.jpg", cv2.IMREAD_COLOR)
#         normalized_xywh_label = open_label_file("000007.txt")
#
#         resized_img, scaled_xyxy_label = resize_img(img,
#                                                     target_resize=(416, 416),
#                                                     normalized_xywh_label=normalized_xywh_label,
#                                                     use_letterbox=True)
#
#         # augmentation gogo
#         aug_resized_img, aug_scaled_xyxy_label = horFlip(resized_img, scaled_xyxy_label, p=0.5)
#         aug_resized_img, aug_scaled_xyxy_label = randomScale(aug_resized_img, aug_scaled_xyxy_label, scale=[-0.25, 0.5], p=0.5)
#         aug_resized_img, aug_scaled_xyxy_label = randomTranslation(aug_resized_img, aug_scaled_xyxy_label, p=0.5)
#         aug_resized_img, aug_scaled_xyxy_label = randomShear(aug_resized_img, aug_scaled_xyxy_label, shear_degree=7.0, p=0.5)
#         aug_resized_img, aug_scaled_xyxy_label = randomRotation(aug_resized_img, aug_scaled_xyxy_label, angle_degree=7.0, p=0.5)
#         aug_resized_img = hsvColorSpaceJitter(aug_resized_img, hGain=0.05, sGain=0.2, vGain=0.3, p=0.5)
#
#         drawBBox(aug_resized_img, aug_scaled_xyxy_label)
#
#         cv2.imshow("img", img)
#         cv2.imshow("aug_resized_img", aug_resized_img)
#
#         ch = cv2.waitKey(0)
#         if ch == 27:
#             break