from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, VOC_Config, AnnotationTransform, VOCDetection, detection_collate, BaseTransform, preproc
from models.RFB_Net_vgg import build_net
from layers.modules import MultiBoxLoss
from layers.functions import PriorBox
import time
from datetime import datetime
from utils.visualize import *
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-max','--max_epoch', default=120,
                    type=int, help='max epoch for retraining')
parser.add_argument('-b', '--batch_size', default=32,
                    type=int, help='Batch size for training')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=0.08, type=float, help='initial learning rate')
parser.add_argument('--save_folder', default='./weights/',
                    help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

img_dim = 300
p = 0.5
train_sets = [('2007', 'person_trainval'), ('2012', 'person_trainval')]
cfg = VOC_Config
rgb_means = (104, 117, 123)
batch_size = args.batch_size

# tensorboard log directory
# LOG_DIR = 'runs'
log_path = os.path.join('runs', datetime.now().isoformat())
if not os.path.exists(log_path):
    os.makedirs(log_path)
writer = SummaryWriter(log_dir=log_path)


net = build_net('train', img_dim, num_classes=2)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net)

net.cuda()
cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-4)

criterion = MultiBoxLoss(num_classes=2,
                         overlap_thresh=0.4,
                         prior_for_matching=True,
                         bkg_label=0,
                         neg_mining=True,
                         neg_pos=3,
                         neg_overlap=0.3,
                         encode_target=False)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()


def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')

    dataset = VOCDetection(VOCroot, train_sets, preproc(img_dim, rgb_means, p), AnnotationTransform())

    epoch_size = len(dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    stepvalues = (80 * epoch_size, 95 * epoch_size, 105 * epoch_size)
    step_index = 0
    start_iter = 0
# wangsong sing a song!
    lr = args.lr
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            if (epoch > 10 and epoch % 10 == 0) or (epoch > 105 and epoch % 2 == 0):
                torch.save(net.state_dict(), args.save_folder + 'epoches_' +
                           repr(epoch).zfill(3) + '.pth')
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                                  shuffle=True, num_workers=8, collate_fn=detection_collate))
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, 0.2, epoch, step_index, iteration, epoch_size)


        images, targets = next(batch_iterator)

        images = Variable(images.cuda())
        targets = [Variable(anno.cuda()) for anno in targets]

        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        load_t1 = time.time()

        # visualization
        visualize_total_loss(writer, loss.item(), iteration)
        visualize_loc_loss(writer, loss_l.item(), iteration)
        visualize_conf_loss(writer, loss_c.item(), iteration)

        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f||' % (
                loss_l.item(),loss_c.item()) +
                'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))

    torch.save(net.state_dict(), args.save_folder + 'epoches_' +
               repr(epoch).zfill(3) + '.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 11:
        lr = 1e-8 + (args.lr-1e-8) * iteration / (epoch_size * 10)
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
