# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:03:17 2020

@author: rashe
"""
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')