# MnasYOLOv3

This is the YOLOv3 implementation using backbone as Mnasnet.

When I searched various yolov3 implementation, I've never seen the case using Mnasnet as a backbone network of YOLOv3.

So I implemented.

The inspiration for this project comes from [yjh0410/pytorch-yolo-v2-v3](https://github.com/yjh0410/pytorch-yolo-v2-v3) Thanks.

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. This implementation is a work in progress -- new features are currently being implemented.

## Training

```
python train.py --dataset "your_voc_dataset_path"
```

## Testing

```
python eval_voc.py --trained_model "your_trained_model.pth"
```

## Pretrained Weights

You can download it from [GoogleDrive](https://drive.google.com/open?id=10FFEoagSBTDfwCDz1F2i5t3q-SFtfvmB).

## Results

CPU: I7-6700

GPU: GTX 1080Ti 11GB

RAM: DDR4 16GB

| Model name | InputSize | TrainSet | TestSet | mAP | Speed |
| ----- | ------ | ------ | ------ | ----- | ----- |
| MnasYOLOv3-Mobilenet | 416x416 | VOC07+12 | VOC07 | 76.15% | 25fps |


### To do lists

Applying multi-scale training


