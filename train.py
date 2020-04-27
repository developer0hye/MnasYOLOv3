from augmentations import SSDAugmentation
import os
import time
import random
import tools
import torch
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import argparse
from voc0712 import *


parser = argparse.ArgumentParser(description='YOLO-v2 Detection')
parser.add_argument('-v', '--version', default='yolo_v2',
                    help='yolo_v2, yolo_v3, tiny_yolo_v2, tiny_yolo_v3')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('-hr', '--high_resolution', type=int, default=0,
                    help='1: use high resolution to pretrain; 0: else not.')
parser.add_argument('-ms', '--multi_scale', type=int, default=0,
                    help='1: use multi-scale trick; 0: else not')
parser.add_argument('-fl', '--use_focal', type=int, default=0,
                    help='0: use focal loss; 1: else not;')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--obj', default=5.0, type=float,
                    help='the weight of obj loss')
parser.add_argument('--noobj', default=1.0, type=float,
                    help='the weight of noobj loss')
parser.add_argument('-wp', '--warm_up', type=str, default='yes',
                    help='yes or no to choose using warmup strategy to train')
parser.add_argument('--wp_epoch', type=int, default=6,
                    help='The upper bound of warm-up')
parser.add_argument('--weights', type=str, default=None,
                    help='load weights to resume training')
parser.add_argument('--total_epoch', type=int, default=250,
                    help='total_epoch')
parser.add_argument('--dataset_root', default="./VOCdevkit",
                    help='Location of VOC root directory')
parser.add_argument('--num_classes', default=20, type=int,
                    help='The number of dataset classes')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--gpu_ind', default=0, type=int,
                    help='To choose your gpu.')
parser.add_argument('--save_folder', default='./weights', type=str,
                    help='Gamma update for SGD')
parser.add_argument('--fine_tune', default=0, type=int,
                    help='fine tune the model trained on MSCOCO.')

args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(20)
def train(model, device):
    global cfg, hr

    # set GPU
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    import dataset
    train_dataset = dataset.YOLODataset(path=args.dataset_root)
    data_loader = data.DataLoader(train_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=dataset.yolo_collate,
                                  pin_memory=True)

    # dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim']))
    # data_loader = data.DataLoader(dataset, args.batch_size,
    #                               num_workers=args.num_workers,
    #                               shuffle=True, collate_fn=detection_collate,
    #                               pin_memory=True)

    print("----------------------------------------Object Detection--------------------------------------------")
    print("Let's train OD network !")
    net = model
    net = net.to(device)

    lr = args.lr
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    iter_per_epoch = int(np.ceil(len(train_dataset)/args.batch_size))
    total_iter = iter_per_epoch * args.total_epoch
    warmup_iter = iter_per_epoch * 6

    scheduler = tools.LRScheduler(optimizer=optimizer,
                                  warmup_iter=warmup_iter,
                                  total_iter=total_iter,
                                  target_lr=lr)

    # loss counters
    print("----------------------------------------------------------")
    print('Loading the dataset...')
    # print('Training on:', dataset.name)
    print('The dataset size:', len(train_dataset))
    print('The obj weight : ', args.obj)
    print('The noobj weight : ', args.noobj)
    print("----------------------------------------------------------")

    # create batch iterator
    iteration = 0
    start_epoch = 0

    if args.weights is not None:
        chkpt = torch.load(args.weights, map_location=device)

        start_epoch = chkpt['epoch']
        iteration = chkpt['iteration']
        chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        chkpt['optimizer'] = {k: v for k, v in chkpt['optimizer'].items()}

        model.load_state_dict(chkpt['model'], strict=False)
        optimizer.load_state_dict(chkpt['optimizer'])

    # start training
    for epoch in range(start_epoch, args.total_epoch):
        for images, bboxes_label_list in data_loader:
            iteration += 1
            scheduler.step(iteration)

            images = images.to(device)
            input_size = images.shape[2:]

            # forward
            t0 = time.time()
            out, _ = net(images)

            targets = tools.build_targets(model,
                                          bboxes_label_list,
                                          batch_size=len(images),
                                          input_size=input_size,
                                          dtype=out.dtype)
            targets = targets.to(device)

            loss = model.yololoss(out, targets)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t1 = time.time()

            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('Epoch[%d / %d]' % (epoch + 1, args.total_epoch) + ' || iter[%d / %d] ' % (iteration, total_iter) + \
                      ' || Loss: %.4f ||' % (loss.item()) + ' || lr: %.8f ||' % (optimizer.param_groups[0]['lr']) + ' || input size: %d ||' %
                      input_size[0], end=' ')

        if (epoch + 1) % 10 == 0:
            chkpt = {'epoch': epoch + 1,
                     'iteration': iteration,
                     'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()}

            print('Saving state, epoch:', epoch + 1)

            torch.save(chkpt,
                       args.save_folder + '/' + 'yolov3tiny' + '_' +
                       repr(epoch + 1) + '.pth')

if __name__ == '__main__':
    global hr, cfg

    hr = False
    device = 'cuda'

    if args.high_resolution == 1:
        hr = True

    cfg = voc_ab

    from model import MnasYOLOv3
    yolo_net = MnasYOLOv3()

    print('Let us train tiny-yolo-v3 on the VOC0712 dataset ......')

    train(yolo_net, device)
