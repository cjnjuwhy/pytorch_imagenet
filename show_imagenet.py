#!/usr/bin/python

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

from nets import net_factory

MEAN_COEF = [0.485, 0.456, 0.406]
DIV_COEF = [0.229, 0.224, 0.225]

# functions to show an image
def imshow(img):
    npimg = img.numpy() * np.array(DIV_COEF).reshape([3,1,1]) \
          + np.array(MEAN_COEF).reshape([3,1,1])            # Un-normalize
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def evaluate(model, args, input, target):
    checkpoint_path = get_checkpoint(args.checkpoint_dir)
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (best_acc1 {})"
              .format(checkpoint_path, best_acc1))

    model.eval()
    with torch.no_grad():
        output = model(input)
        acc1, acc5, pred = accuracy(output, target, topk=(1, 5))
        print(pred[0])
        print(target)
        print(' * Acc@1 {:.3f} Acc@5 {:.3f}'.format(acc1[0], acc5[0]))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        res.append(pred)
        return res


def get_checkpoint(cp_dir):
    checkpoint_path = '0'
    for files in os.listdir(cp_dir):
        if files.split('.')[-1] == 'pth':
            if files > checkpoint_path:
                checkpoint_path = files
    checkpoint_path = cp_dir + checkpoint_path
    return checkpoint_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ImageNet Loader Test')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-a', '--arch', default='resnet18', type=str,
                        metavar='architecture', help='nets')
    parser.add_argument('--checkpoint-dir', default='./', type=str,
                        metavar='DIR', help='checkpoint dir')

    args = parser.parse_args()

    model = net_factory.get_net(args.arch)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=MEAN_COEF,
                                     std=DIV_COEF)

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # Sample training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    evaluate(model, args, images, labels)
    # show images
    imshow(torchvision.utils.make_grid(images))
    plt.show()

    # Sample validation images
    dataiter = iter(val_loader)
    images, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))
    plt.show()