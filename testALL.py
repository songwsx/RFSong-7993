from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot
from data import AnnotationTransform,VOCDetection, BaseTransform, VOC_Config
from models.RFB_Net_vgg import build_net
import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer
from utils.visualize import print_info
from tqdm import tqdm

# weights/epoches_112.pth
# Finished loading model!
# 100%|███████████████████████████████████████| 2007/2007 [00:58<00:00, 34.07it/s]
# Evaluating detections
# Writing person VOC results file
# VOC07 metric? Yes
# AP for person = 0.7993
# Mean AP = 0.7993
# ~~~~~~~~
# Results:
# 0.799
# 0.799
# ~~~~~~~~
parser = argparse.ArgumentParser(description='Receptive Field Block Net')
# 进行批量测试
parser.add_argument('--weights_path', default='weights',
                    help='weights path.')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool,
                    help='Use cpu nms')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

cfg = VOC_Config

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.01):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    num_classes = 2
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    for i in tqdm(range(num_images)):
        img = testset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            if cuda:
                x = x.cuda()
                scale = scale.cuda()

        _t['im_detect'].tic()
        out = net(x)      # forward pass
        boxes, scores = detector.forward(out,priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores=scores[0]

        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = nms(c_dets, 0.45, force_cpu=args.cpu)
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets

        nms_time = _t['misc'].toc()

        # if i % 20 == 0:
        #     print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
        #         .format(i + 1, num_images, detect_time, nms_time))
        #     _t['im_detect'].clear()
        #     _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder)


if __name__ == '__main__':
    # load net
    img_dim = 300
    num_classes = 2
    rgb_means = (104, 117, 123)
    start_epoch = 20
    trained_model_list = os.listdir(args.weights_path)
    trained_model_list.sort()
    net = build_net('test', img_dim, num_classes)  # initialize detector
    for trained_model in trained_model_list:
        start_epoch += 10
        if start_epoch < 80+10:
            continue
        trained_model  = os.path.join(args.weights_path, trained_model)
        print_info(trained_model, ['yellow', 'bold'])
        state_dict = torch.load(trained_model)
        # create new OrderedDict that does not contain `module.`

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        net.eval()
        print('Finished loading model!')
        # load data
        testset = VOCDetection(VOCroot, [('2007', 'person_test')], None, AnnotationTransform())

        if args.cuda:
            net = net.cuda()
            cudnn.benchmark = True
        else:
            net = net.cpu()

        top_k = 200
        detector = Detect(num_classes,0,cfg)
        save_folder = os.path.join(args.save_folder, 'VOC')
        test_net(save_folder, net, detector, args.cuda, testset,
                 BaseTransform(img_dim, rgb_means, (2, 0, 1)),
                 top_k, thresh=0.01)
