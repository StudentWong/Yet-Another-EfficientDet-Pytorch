# Author: Zylo117

import math

import torch
from torch import nn

from efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet
from efficientdet.utils import Anchors


class EfficientDetBackbone(nn.Module):
    def __init__(self, num_anchors=9, num_classes=80, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        self.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        self.num_scales = 3
        conv_channel_coef = {
            # TODO: I have only tested on D0~5, if you want to try it on other coefficients,
            #  fill it in with the channels of P3/P4/P5 like this.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
        }

        new_num_anchors = len(kwargs.get('ratios', [])) * len(kwargs.get('scales', []))
        if new_num_anchors > 0:
            num_anchors = new_num_anchors
        else:
            num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False) for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef])
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef])

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef], **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.backbone_net = EfficientNet(compound_coef, load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs, inputs.dtype)

        return features, regression, classification, anchors

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')
