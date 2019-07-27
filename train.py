import os
import random
import sys
import argparse
import shutil
import numpy as np
from PIL import Image

import torchvision.transforms as standard_transforms
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import dataset

train_live_rgb_dir   = './data/live_train_face_rgb'
train_live_depth_dir = './data/live_train_face_depth'
train_fake_rgb_dir   = './data/fake_train_face_rgb'

test_live_rgb_dir    = './data/live_test_face_rgb'
test_fake_rgb_dir    = './data/fake_test_face_rgb'

parser = argparse.ArgumentParser(description='PyTorch Liveness Training')
parser.add_argument('-s', '--scale', default=1.0, type=float,
                    metavar='N', help='net scale')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


class Net(nn.Module):
    def __init__(self, scale = 1.0,expand_ratio=1):
        super(Net, self).__init__()
        def conv_bn(inp, oup, stride = 1):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.PReLU(oup)
            )
        def conv_dw(inp, oup, stride = 1):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.PReLU(inp),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.PReLU(oup),
            )
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout()
        self.head = conv_bn(3, (int)(32 * scale))
        self.step1 = nn.Sequential(
            conv_dw((int)(32 * scale), (int)(64 * scale), 2),
            conv_dw((int)(64 * scale), (int)(128 * scale)),
            conv_dw((int)(128 * scale), (int)(128 * scale)),
        )
        self.step1_shotcut = conv_dw((int)(32 * scale), (int)(128 * scale), 2)

        self.step2 = nn.Sequential(
            conv_dw((int)(128 * scale), (int)(128 * scale), 2),
            conv_dw((int)(128 * scale), (int)(256 * scale)),
            conv_dw((int)(256 * scale), (int)(256 * scale)),
        )
        self.step2_shotcut = conv_dw((int)(128 * scale), (int)(256 * scale), 2)
        self.depth_ret = nn.Sequential(
            nn.Conv2d((int)(256 * scale), (int)(256 * scale), 3, 1, 1, groups=(int)(256 * scale), bias=False),
            nn.BatchNorm2d((int)(256 * scale)),
            nn.Conv2d((int)(256 * scale), 2, 1, 1, 0, bias=False),
        )


    def forward(self, x):
        head = self.head(x)
        step1 = self.step1(head) + self.step1_shotcut(head)
        step2 = self.dropout(self.step2(step1) + self.step2_shotcut(step1))
        depth = self.softmax(self.depth_ret(step2))
        return depth

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DepthFocalLoss(nn.Module):
    def __init__(self, gamma = 1, eps = 1e-7):
        super(DepthFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.MSELoss(reduction='none')

    def forward(self, input, target):
        loss = self.ce(input, target)
        loss = (loss) ** self.gamma
        return loss.mean()

def main(args):
    device = torch.device('cuda:5')
    net = Net(args.scale)
    net = nn.DataParallel(net, device_ids = [5, 6, 7, 8])
    net = net.to(device)
    print(net)
    print("start load train data")
    normalize = standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])


    random_input_transform = standard_transforms.Compose([
        standard_transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1),
        standard_transforms.RandomResizedCrop((128, 128), scale=(0.9, 1), ratio=(1, 1)),
        standard_transforms.ToTensor(),
        normalize
    ])

    target_transform = standard_transforms.Compose([
        standard_transforms.Resize((32, 32)),
        standard_transforms.ToTensor()
    ])

    train_set = dataset.Dataset('train', train_live_rgb_dir, train_live_depth_dir, train_fake_rgb_dir,
        random_transform = random_input_transform, target_transform = target_transform)
    train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers = 4, shuffle = True, drop_last=True)

    val_set = dataset.Dataset('test', test_live_rgb_dir, None, test_fake_rgb_dir,
        random_transform = random_input_transform, target_transform = target_transform)
    val_loader = DataLoader(val_set, batch_size = 1, num_workers = 4, shuffle = False)

    criterion_depth = DepthFocalLoss()
    optimizer = torch.optim.Adam(net.parameters())

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            g_err_rate = checkpoint['best_err_rate']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            net.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    if args.evaluate:
        validate(device, net, val_loader, args.arch)
        return

    for epoch in range(args.start_epoch, args.epochs):

        train(device, net, train_loader, criterion_depth, optimizer, epoch)
        validate(device, net, val_loader)
       
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict()
        })


def validate(device, net, val_loader, depth_dir = './depth_predict'):
    try:
        shutil.rmtree(depth_dir)
    except:
        pass
    try:
        os.makedirs(depth_dir)
    except:
        pass
    toImage = standard_transforms.ToPILImage(mode='L')
    net.eval()

    for i, data in enumerate(val_loader):
        input, label = data
        input = input.cuda(device)
        output = net(input)
        out_depth = output[:,0,:,:]
        out_depth = out_depth.detach().cpu()
        image = toImage(out_depth)
        if label == 0:
            name = '' + depth_dir + '/fake-' + str(i) + '.bmp'
            image.save(name)

        if label == 1:
            name = '' + depth_dir + '/live-' + str(i) + '.bmp'
            image.save(name)


def conv_loss(device, out_depth, label_depth, criterion_depth):
    loss0 = criterion_depth(out_depth, label_depth)
    filters1 = torch.tensor([[[[-1, 0, 0],[0, 1, 0],[0, 0, 0]]]], dtype=torch.float).cuda(device)
    filters2 = torch.tensor([[[[0, -1, 0],[0, 1, 0],[0, 0, 0]]]], dtype=torch.float).cuda(device)
    filters3 = torch.tensor([[[[0, 0, -1],[0, 1, 0],[0, 0, 0]]]], dtype=torch.float).cuda(device)
    filters4 = torch.tensor([[[[0, 0, 0],[-1, 1, 0],[0, 0, 0]]]], dtype=torch.float).cuda(device)
    filters5 = torch.tensor([[[[0, 0, 0],[0, 1, -1],[0, 0, 0]]]], dtype=torch.float).cuda(device)
    filters6 = torch.tensor([[[[0, 0, 0],[0, 1, 0],[-1, 0, 0]]]], dtype=torch.float).cuda(device)
    filters7 = torch.tensor([[[[0, 0, 0],[0, 1, 0],[0, -1, 0]]]], dtype=torch.float).cuda(device)
    filters8 = torch.tensor([[[[0, 0, 0],[0, 1, 0],[0, 0, -1]]]], dtype=torch.float).cuda(device)

    loss1 = criterion_depth(nn.functional.conv2d(out_depth, filters1, padding = 1),
        nn.functional.conv2d(label_depth, filters1, padding = 1))
    loss2 = criterion_depth(nn.functional.conv2d(out_depth, filters2, padding = 1),
        nn.functional.conv2d(label_depth, filters2, padding = 1))
    loss3 = criterion_depth(nn.functional.conv2d(out_depth, filters3, padding = 1),
        nn.functional.conv2d(label_depth, filters3, padding = 1))
    loss4 = criterion_depth(nn.functional.conv2d(out_depth, filters4, padding = 1),
        nn.functional.conv2d(label_depth, filters4, padding = 1))
    loss5 = criterion_depth(nn.functional.conv2d(out_depth, filters5, padding = 1),
        nn.functional.conv2d(label_depth, filters5, padding = 1))
    loss6 = criterion_depth(nn.functional.conv2d(out_depth, filters6, padding = 1),
        nn.functional.conv2d(label_depth, filters6, padding = 1))
    loss7 = criterion_depth(nn.functional.conv2d(out_depth, filters7, padding = 1),
        nn.functional.conv2d(label_depth, filters7, padding = 1))
    loss8 = criterion_depth(nn.functional.conv2d(out_depth, filters8, padding = 1),
        nn.functional.conv2d(label_depth, filters8, padding = 1))

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
    return loss


def train(device, net, train_loader, criterion_depth, optimizer, epoch):
    losses_depth = AverageMeter()
    net.train()
    for i, data in enumerate(train_loader):
        input, depth, label = data
        input = input.cuda(device)
        depth = depth.cuda(device)
        label = label.cuda(device)
        output = net(input)

        out_depth = output[:,0,:,:]
        loss_depth = conv_loss(device, torch.reshape(out_depth, (-1, 1, 32, 32)), depth, criterion_depth)
        losses_depth.update(loss_depth.data, input.size(0))
        loss = loss_depth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("epoch:{} batch:{} depth loss:{:f} depth avg loss:{:f}".format(
                epoch, i, loss_depth.data.cpu().numpy(), losses_depth.avg.cpu().numpy()))


if __name__ == '__main__':
    main(parser.parse_args())
