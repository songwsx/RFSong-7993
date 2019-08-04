# config.py
import os.path


VOCroot = '/home/common/wangsong/VOC/VOCdevkit'

#RFB CONFIGS
VOC_Config = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [26, 60, 111, 162, 213, 264],

    'max_sizes' : [60, 111, 162, 213, 264, 315],

    'aspect_ratios' : [[0.4, 1.0, 1.5],
                       [0.4, 1.0, 1.6],
                       [0.5, 1.1, 1.6],
                       [0.5, 1.1, 1.6],
                       [0.5, 1.1, 1.6],
                       [0.7, 1.4]],

    'max_ratios' : [0.8, 0.8, 0.8, 0.9, 1, 1],

    'variance' : [0.1, 0.2],

    'clip' : True,
}


