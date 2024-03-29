# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 09:21:29 2021

@author: lenovo
"""

import torch
import torch.nn as nn

from torch import optim

import numpy as np
from model.MGNet import MGNet
import cv2

from utils import JointTransform2D, ImageToImage2D, Image2D
from torch.utils import data
from losses import DiceLoss
from utils.utils import *
from utils import metrics
from optparse import OptionParser
import SimpleITK as sitk
from torch.utils.tensorboard import SummaryWriter
import time
import math
import os
import sys
import pdb
import warnings
from torch.utils.data import DataLoader
import torch.nn.functional as F
warnings.filterwarnings("ignore", category=UserWarning)
DEBUG = False


          


def validation(net, test_loader, options):

    net.eval()

    with torch.no_grad():
        for i, (data_i, label, *rest) in enumerate(test_loader):
            if isinstance(rest[0][0], str):
                        image_filename = rest[0][0]
            else:
                        image_filename = '%s.png' % str(i + 1).zfill(3) 
                        
            inputs, labels = data_i.float().cuda(), label.long().cuda()
            pred = net(inputs)
            pred = pred[0]
            pred = F.softmax(pred, dim=1)
            _, label_pred = torch.max(pred, dim=1)
            
            label_pred1 = label_pred.view(-1, 1)
            label_true1 = labels.view(-1, 1)         

 

            tmp2 = labels.detach().cpu().numpy()
            tmp = label_pred.detach().cpu().numpy()
            tmp[tmp>=0.5] = 1
            tmp[tmp<0.5] = 0
            tmp2[tmp2>0] = 1
            tmp2[tmp2<=0] = 0
            tmp2 = tmp2.astype(int)
            tmp = tmp.astype(int)
        
            yHaT = tmp
            yval = tmp2
    
            
            del inputs, label_true1,tmp,tmp2, label_pred1
            image_filename2=image_filename[0:-4]+'gt_.png'
            
            yHaT[yHaT==1] =255
            yval[yval==1] =255
            fulldir =options.pred + options.unique_name
            if not os.path.isdir(fulldir):
                
                os.makedirs(fulldir)


            cv2.imwrite( fulldir+'\\'+image_filename, yHaT[0,:,:])#[0,:,:,:]
            cv2.imwrite( fulldir+'\\'+image_filename2, yval[0,:,:])#[0,:,:,:]


if __name__ == '__main__':
    parser = OptionParser()
    def get_comma_separated_int_args(option, opt, value, parser):
        value_list = value.split(',')
        value_list = [int(i) for i in value_list]
        setattr(parser.values, option.dest, value_list)


    parser.add_option('-c', '--resume', type='str', dest='load', default='./checkpoint/best.pth', help='path of pretrained model')
    parser.add_option('-pred', '--pred-path', type='str', dest='pred_path', default='./pred/', help='pred path')
    parser.add_option('--test_data_path', type='str', dest='test_data_path', default=r'F:\DATA\\test', help='dataset path')
    parser.add_option('-o', '--log-path', type='str', dest='log_path', default='./log/', help='log path')
    parser.add_option('-m', type='str', dest='model', default='UTNet', help='use which model')
    parser.add_option('--num_class', type='int', dest='num_class', default=2, help='number of segmentation classes')
    parser.add_option('--base_chan', type='int', dest='base_chan', default=32, help='number of channels of first expansion in UNet')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name', default='test', help='unique experiment name')
    parser.add_option('--rlt', type='float', dest='rlt', default=1, help='relation between CE/FL and dice')
    parser.add_option('--weight', type='float', dest='weight',
                      default=[0.5,1] , help='weight each class in loss function')
    parser.add_option('--weight_decay', type='float', dest='weight_decay',
                      default=0.0001)
    parser.add_option('--scale', type='float', dest='scale', default=0.30)
    parser.add_option('--rotate', type='float', dest='rotate', default=180)
    parser.add_option('--crop_size', type='int', dest='crop_size', default=384)
    parser.add_option('--fuse', dest='aux_loss', action='store_true', help='using feature fusion method')
    parser.add_option('--reduce_size', dest='reduce_size', default=8, type='string')
    parser.add_option('--block_list', dest='block_list', default='1234', type='str')
    parser.add_option('--aux_loss', dest='aux_loss', action='store_true', help='using aux loss for deep supervision')

    
    parser.add_option('--gpu', type='str', dest='gpu', default='0')
    options, args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

    print('Using model:', options.model)

    net = MGNet(3, options.base_chan, options.num_class, reduce_size=options.reduce_size, block_list=options.block_list, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=options.aux_loss,fuse=options.fuse, maxpool=True)

    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))

    net.cuda()
    

    tf_test = JointTransform2D(crop=(384,384), p_flip=0, color_jitter_params=None, long_mask=True)
    test_dataset = ImageToImage2D(options.val_data_path, tf_test)
    test_loader = DataLoader(tf_test, 1, shuffle=True)
    
    validation(net, test_loader, options)

    print('done')

    sys.exit(0)
