"""VOC Dataset Classes
Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
Updated by: Ellis Brown, Max deGroot, Yonghye Kwon
"""
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def base_transform(image, size):
    x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
    return x

class BaseTransform:
    def __init__(self, size):
        self.size = size
    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size), boxes, labels

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs, 0), targets


# MULTI_ANCHOR_SIZE = [[1.02, 1.49], [1.57, 3.38], [3.96, 3.01],    # 4 *[4.08, 5.96], [6.28, 13.52], [15.84, 12.04]
#                      [2.45, 6.31], [5.57, 5.58], [4.05, 9.21],    # 2 *[4.9, 12.62], [11.14, 11.16], [8.10, 18.42]
#                      [10.37, 6.08], [7.12, 10.18], [11.42, 11.21]]   # 1 * [10.37, 6.08], [7.12, 10.18], [11.42, 11.21]

MULTI_ANCHOR_SIZE = [[0.0587, 0.1009], [0.0868, 0.2519], [0.1682, 0.1418],
                     [0.1731, 0.3962], [0.3792, 0.3031], [0.2924, 0.6362],
                     [0.7266, 0.4697], [0.5062, 0.777], [0.8672, 0.8544]]

#[0.0587 0.1009] 0.008 0.0107 0.1502 0.224 0.0613 0.1024
#[0.0868 0.2519] 0.026 0.1307 0.154 0.776 0.0889 0.2573
#[0.1682 0.1418] 0.092 0.03 0.614 0.244 0.1749 0.1482
#[0.1731 0.3962] 0.082 0.22 0.278 0.992 0.1753 0.4063
#[0.3792 0.3031] 0.232 0.082 0.998 0.4853 0.381 0.3064
#[0.2924 0.6362] 0.126 0.398 0.4375 0.998 0.2935 0.6396
#[0.7266 0.4697] 0.4209 0.1493 0.998 0.692 0.728 0.4679
#[0.5062 0.777 ] 0.336 0.512 0.704 0.998 0.5058 0.7749
#[0.8672 0.8544] 0.62 0.568 0.998 0.998 0.8661 0.8532

# 성능 낮음;
# [0.1058 0.1238] 0.008 0.0097 0.482 0.3093 0.1069 0.1272
# [0.1528 0.3072] 0.028 0.146 0.2435 0.84 0.1567 0.3134
# [0.341  0.2818] 0.2216 0.0501 0.8199 0.4305 0.3517 0.2734
# [0.2647 0.655 ] 0.0755 0.3913 0.3775 0.9976 0.2628 0.6385
# [0.4265 0.505 ] 0.2991 0.3279 0.6196 0.7226 0.4377 0.5069
# [0.7307 0.3962] 0.5032 0.0827 0.998 0.543 0.7446 0.3852
# [0.5332 0.7978] 0.3271 0.5583 0.7251 0.998 0.5308 0.8005
# [0.8186 0.6365] 0.5968 0.4627 0.998 0.79 0.813 0.6344
# [0.8713 0.9014] 0.6328 0.7251 0.998 0.998 0.8715 0.9011

# yolo-v2 config
voc_ab = {
    'num_classes': 20,
    'lr_epoch': (150, 200, 250),
    'max_epoch': 250,
    'min_dim': [416, 416],
    'ms_channels':[128, 256, 512],
    'stride': 32,
    'strides': [16, 32],
    'multi_scale': [[320, 320], [352, 352], [384, 384], [416, 416], [448, 448],
                 [480, 480], [512, 512], [544, 544], [576, 576], [608, 608]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
path_to_dir = osp.dirname(osp.abspath(__file__))
VOC_ROOT = path_to_dir + "/VOCdevkit/"

VOC_ROOT = 'E:/python_work/OD/yolo-guide/data/VOCdevkit/'


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])        
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])

            # to rgb
            img = img[:, :, (2, 1, 0)]
            img = torch.from_numpy(img).permute(2, 0, 1)
            img = img / 255.
            img = self.normalize(img)

            import tools
            boxes = torch.from_numpy(boxes).type(torch.float32)
            labels = torch.from_numpy(labels).type(torch.float32)

            boxes = tools.xyxy2xywh(boxes)
            labels = labels.view(-1, 1)

            target = torch.cat([labels, boxes], dim=-1)

        return img, target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
