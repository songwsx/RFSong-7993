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
import cv2

parser = argparse.ArgumentParser(description='Receptive Field Block Net')
parser.add_argument('--img_dir', default='images', type=str,
                    help='Dir to save results')
parser.add_argument('-m', '--trained_model', default='weights/epoches_260.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool,
                    help='Use cpu nms')
args = parser.parse_args()


cfg = VOC_Config
img_dim = 300
num_classes = 2
rgb_means = (104, 117, 123)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


class ObjectDetector:
    def __init__(self, net, detection, transform, num_classes=2, thresh=0.1, cuda=True):
        self.net = net
        self.detection = detection
        self.transform = transform
        self.num_classes = num_classes
        self.thresh = thresh
        self.cuda = cuda

    def predict(self, img):
        _t = {'im_detect': Timer(), 'misc': Timer()}
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])

        with torch.no_grad():
            x = self.transform(img).unsqueeze(0)
            if self.cuda:
                x = x.cuda()
                scale = scale.cuda()

        _t['im_detect'].tic()
        out = net(x)  # forward pass
        boxes, scores = self.detection.forward(out, priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        # scale each detection back up to the image
        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        _t['misc'].tic()
        all_boxes = [[] for _ in range(num_classes)]

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > self.thresh)[0]
            if len(inds) == 0:
                all_boxes[j] = np.zeros([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            #print(scores[:, j])
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            # keep = nms(c_bboxes,c_scores)

            keep = nms(c_dets, 0.45, force_cpu=args.cpu)
            c_dets = c_dets[keep, :]
            all_boxes[j] = c_dets

        nms_time = _t['misc'].toc()
        total_time = detect_time+nms_time

        #print('total time: ', total_time)
        return all_boxes, total_time

if __name__ == '__main__':
    # load net
    net = build_net('test', img_dim, num_classes)    # initialize detector
    state_dict = torch.load(args.trained_model)
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

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()

    detector = Detect(num_classes,0,cfg)

    transform = BaseTransform(img_dim, rgb_means, (2, 0, 1))
    object_detector = ObjectDetector(net, detector, transform)

    img_list = os.listdir(args.img_dir)
    for i, img in enumerate(img_list):
        img_name = img
        img = os.path.join(args.img_dir, img)
        image = cv2.imread(img)
        detect_bboxes, tim = object_detector.predict(image)

        for class_id,class_collection in enumerate(detect_bboxes):
            if len(class_collection)>0:
                for i in range(class_collection.shape[0]):
                    if class_collection[i,-1]>0.6:
                        pt = class_collection[i]
                        cv2.rectangle(image, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (0, 255, 0), 2)

        cv2.imshow('result',image)
        cv2.waitKey()