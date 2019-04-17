#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:31:05 2019
This is a factory to provide the models
add models in this file
@author: huanyu_wang
"""
import torchvision.models as models
#from pvanet import pvanet
from nets import mynet1
import warnings

# "pvanet":pvanet(),
# there is something wrong with the code in pvanet
Net = dict({"whynet":mynet1.get_model(),
        })

def get_net(name):
    if name in models.__dict__.keys():
        return models.__dict__[name]()
    elif name in Net.keys():
        return Net[name]
    else:
        warnings.warn("You don't define the net: "+name)