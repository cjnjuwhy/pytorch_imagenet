#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:44:46 2019

@author: huanyu_wang
"""

import torch
import torchvision
import torch.nn as nn

class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        pass

    def forward(self, x):
        # model details
        return x


def get_model():
    return Mynet()
