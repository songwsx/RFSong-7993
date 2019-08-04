import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from math import sqrt as sqrt
from itertools import product as product


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.max_ratios = cfg['max_ratios']

        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = self.min_sizes[k]/self.image_size

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime*sqrt(self.max_ratios[k]), s_k_prime/sqrt(self.max_ratios[k])]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]


        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

if __name__ == '__main__':
    # RFB CONFIGS
    VOC_Config = {
        'feature_maps': [38, 19, 10, 5, 3, 1],

        'min_dim': 300,

        'steps': [8, 16, 32, 64, 100, 300],

        'min_sizes': [26, 60, 111, 162, 213, 264],

        'max_sizes': [60, 111, 162, 213, 264, 315],

        'aspect_ratios': [[0.4, 1.0, 1.5],
                          [0.4, 1.0, 1.6],
                          [0.5, 1.1, 1.6],
                          [0.5, 1.1, 1.6],
                          [0.5, 1.1, 1.6],
                          [0.7, 1.4]],

        'max_ratios': [0.8, 0.8, 0.8, 0.9, 1, 1],

        'variance': [0.1, 0.2],

        'clip': True,
    }

    priorbox = PriorBox(VOC_Config)

    with torch.no_grad():
        priors = priorbox.forward()
    print(priors.shape)