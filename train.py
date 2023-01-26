# -*- coding: utf-8 -*-
"""
Created on Mon May  9 09:33:24 2022

@author: lenovo
"""

import torch
import torch.nn as nn

from torch import optim
from torchstat import stat
import torchvision.models as models
import numpy as np
from model.MGNet import MGNet
import cv2

from utils import JointTransform2D, ImageToImage2D, Image2D
from torch.utils import data
from losses import DiceLoss,FocalLoss
from utils.utils import *
from utils import metrics
from utils import evaluate 
from optparse import OptionParser
import SimpleITK as sitk
from torch.utils.tensorboard import SummaryWriter
import time
import math
import os
import sys
from torch.autograd import Variable
import pdb
import warnings
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from thop import profile

import pickle
from pytorch_model_summary import summary 

warnings.filterwarnings("ignore", category=UserWarning)
DEBUG = False


def train_net(net, options):
    

    tf_train = JointTransform2D(crop=(384,384), p_flip=0, color_jitter_params=None, long_mask=True)
    tf_val = JointTransform2D(crop=(384,384), p_flip=0, color_jitter_params=None, long_mask=True)
    train_dataset = ImageToImage2D(options.data_path, tf_train)
    val_dataset = ImageToImage2D(options.val_data_path, tf_val)
    trainLoader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True)
    valLoader = DataLoader(val_dataset, 1, shuffle=True)

    writer = SummaryWriter(options.log_path + options.unique_name)

    optimizer = optim.SGD(net.parameters(), lr=options.lr, momentum=0.9, weight_decay=options.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(options.weight).cuda())
    criterion_dl = FocalLoss(options.num_class)
    
    
    loss_list=[]
    loss_test=[]
    scores_list=[]
    class_scores_list=[]
    precision_list=[]
    recall_list=[]
    f1_list=[]
    iou_list=[]
    avg_scores_per_epoch=[]
    avg_iou_per_epoch=[]
    avg_loss_per_epoch=[]
    thymoma_list=[]
    loss_test_list=[]
    best_dice = 0
    for epoch in range(options.epochs):
        print('Starting epoch {}/{}'.format(epoch+1, options.epochs))
        epoch_loss = 0

        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=options.lr, epoch=epoch, warmup_epoch=5, max_epoch=options.epochs)

        print('current lr:', exp_scheduler)
        for i, (img, label, *rest) in enumerate(trainLoader):

            img = img.cuda()
            label = label.cuda()

            end = time.time()
            net.train()

            optimizer.zero_grad()
            result = net(img)
            loss = 0

            if isinstance(result, tuple) or isinstance(result, list):
                for j in range(len(result)):
                    loss += options.aux_weight[j] * (criterion(result[j], label.squeeze(1)))+ options.aux_weight[j] *criterion_dl(result[j], label)
            else:
                loss = criterion(result, label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_time = time.time() - end
            if i% 500 == 0:
                print('batch loss: %.5f, batch_time:%.5f'%(loss.item(), batch_time))
        print('[epoch %d] epoch loss: %.5f'%(epoch+1, epoch_loss/(i+1)))
        loss_list.append(epoch_loss/(i+1))
        avg_loss_per_epoch.append(loss_list[epoch])
        print("training loss = ", loss_list[epoch])
        writer.add_scalar('Train/Loss', epoch_loss/(i+1), epoch+1)
        writer.add_scalar('LR', exp_scheduler, epoch+1)

        if os.path.isdir('%s%s/'%(options.cp_path, options.unique_name)):
            pass
        else:
            
            os.mkdir('%s%s/'%(options.cp_path, options.unique_name))

        if epoch % 10 == 0 or epoch > options.epochs-10:
            torch.save(net.state_dict(), '%s%s/CheckPoints%d.pth'%(options.cp_path, options.unique_name, epoch))
        
        if (epoch+1) >0:
            print("validation begain")
            accuracy, class_accuracies, prec, rec, f1, iou,loss_test = validation(net, valLoader, epoch, options)
            print("validation end")

            print("testing loss = ",loss_test)
            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)            
            loss_test_list.append(loss_test)

            if class_accuracies.mean() >= best_dice:
                best_dice = class_accuracies.mean()
                torch.save(net.state_dict(), '%s%s/best.pth'%(options.cp_path, options.unique_name))

            print('save done')
            print('dice: %.5f/best dice: %.5f'%(class_accuracies.mean(), best_dice))
        avg_score = np.mean(scores_list)
        avg_scores_per_epoch.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        avg_iou_per_epoch.append(avg_iou)
        thymoma_list.append(class_accuracies)
        print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
        print("Average per class validation accuracies for epoch # %04d:"% (epoch))

        print("Validation thymoma iou = ", thymoma_list[epoch])
        print("Validation precision = ", avg_precision)
        print("Validation recall = ", avg_recall)
        print("Validation F1 score = ", avg_f1)
        print("Validation IoU score = ", avg_iou)

        fulldir = r""

        fig1, ax1 = plt.subplots(figsize=(11, 8))
    
        ax1.plot(range(epoch+1), thymoma_list)
        ax1.set_title("testing set dice vs epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("testing dice")
    
    
        plt.savefig(fulldir+r'\dice_vs_epochs.png')
    
        plt.clf()
    
        fig2, ax2 = plt.subplots(figsize=(11, 8))
    
        ax2.plot(range(epoch+1), avg_loss_per_epoch, 'b',range(epoch+1), loss_test_list, 'r')
        ax2.set_title("training/testing loss vs epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Current loss")
    
        plt.savefig(fulldir+r'\loss_vs_epochs.png')
    
        plt.clf()


def validation(net, test_loader,epoch, options):

    net.eval()

    accuracy_list=[]
    loss_test_list=[]
    class_accuracies_list=[]
    counter = 0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(options.weight).cuda())
    criterion_dl = FocalLoss(options.num_class)
    
    with torch.no_grad():

        for i, (data_i, label, *rest) in enumerate(test_loader):

            loss=0

            image_filename = 'image_''%s.png' % str(i + 1).zfill(5) 
    
            inputs, labels = data_i.float().cuda(), label.long().cuda()
            label2 = label.cuda()
            pred= net(inputs)

            for j in range(len(pred)):
                loss += options.aux_weight[j] * (criterion(pred[j], label2.squeeze(1)))+options.aux_weight[j] *criterion_dl(pred[j], label2)

            pred = pred[0]

            pred = F.softmax(pred, dim=1)
            _, label_pred = torch.max(pred, dim=1)
            label_pred1 = label_pred.view(-1, 1)
            label_true1 = labels.view(-1, 1)         

            counter += 1
            
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

            accuracy, class_accuracies, prec, rec, f1, iou = evaluate.evaluate_segmentation(pred=yHaT, label=yval, num_classes=2)

            
            loss_test_list.append(loss.item())
            accuracy_list.append(accuracy)
            class_accuracies_list.append(class_accuracies[1])
            del inputs, label_true1,tmp,tmp2, label_pred1
            image_filename2=image_filename[0:-4]+'_gt.png'
            
            yHaT[yHaT==1] =255
            yval[yval==1] =255
            epoch2 = '%s' % epoch
            fulldir =options.pred
            if not os.path.isdir(fulldir):
                
                os.makedirs(fulldir)

            if not os.path.isdir(fulldir+epoch2):
                
                os.makedirs(fulldir+epoch2)

            cv2.imwrite( fulldir+epoch2+'\\'+image_filename, yHaT[0,:,:])#[0,:,:,:]
            cv2.imwrite( fulldir+epoch2+'\\'+image_filename2, yval[0,:,:])#[0,:,:,:]

    accuracy=np.mean(accuracy_list)
    class_accuracies=np.mean(class_accuracies_list)
    loss_test=np.mean(loss_test_list)
    
    return accuracy, class_accuracies, prec, rec, f1, iou,loss_test




def cal_distance(label_pred, label_true, spacing):
    label_pred = label_pred.squeeze(1).cpu().numpy()
    label_true = label_true.squeeze(1).cpu().numpy()
    spacing = spacing.numpy()[0]

    ASD_list = np.zeros(3)
    HD_list = np.zeros(3)

    for i in range(3):
        tmp_surface = metrics.compute_surface_distances(label_true==(i+1), label_pred==(i+1), spacing)
        dis_gt_to_pred, dis_pred_to_gt = metrics.compute_average_surface_distance(tmp_surface)
        ASD_list[i] = (dis_gt_to_pred + dis_pred_to_gt) / 2

        HD = metrics.compute_robust_hausdorff(tmp_surface, 100)
        HD_list[i] = HD

    return ASD_list, HD_list



def transfer_model(pretrained_file, model):

    pretrained_dict = torch.load(pretrained_file)  # get pretrained dict
    model_dict = model.state_dict()  # get model dict
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    return model


def transfer_state_dict(pretrained_dict, model_dict):

    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict



if __name__ == '__main__':
    parser = OptionParser()
    def get_comma_separated_int_args(option, opt, value, parser):
        value_list = value.split(',')
        value_list = [int(i) for i in value_list]
        setattr(parser.values, option.dest, value_list)

    parser.add_option('-e', '--epochs', dest='epochs', default=150, type='int', help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=2, type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.0001, type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False, help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path', default='./checkpoint/', help='checkpoint path')
    parser.add_option('--data_path', type='str', dest='data_path', default=r'F:\DATA\train', help='dataset path')
    parser.add_option('--val_data_path', type='str', dest='val_data_path', default=r'F:\DATA\val', help='dataset path')
    parser.add_option('-o', '--log-path', type='str', dest='log_path', default='./log/', help='log path')
    parser.add_option('-pred', '--pred-path', type='str', dest='pred_path', default='./pred/', help='pred path')
    parser.add_option('-m', type='str', dest='model', default='MGNet', help='use which model')
    parser.add_option('--num_class', type='int', dest='num_class', default=2, help='number of segmentation classes')
    parser.add_option('--base_chan', type='int', dest='base_chan', default=32, help='number of channels of first expansion in UNet')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name', default='train', help='unique experiment name')
    parser.add_option('--rlt', type='float', dest='rlt', default=1, help='relation between CE/FL and dice')
    parser.add_option('--weight', type='float', dest='weight',default=[0.5,1] , help='weight each class in loss function')
    parser.add_option('--weight_decay', type='float', dest='weight_decay', default=0.0001)
    parser.add_option('--scale', type='float', dest='scale', default=0.30)
    parser.add_option('--rotate', type='float', dest='rotate', default=180)
    parser.add_option('--crop_size', type='int', dest='crop_size', default=384)
    parser.add_option('--fuse', dest='aux_loss', action='store_true', help='using feature fusion method')
    parser.add_option('--aux_weight', type='float', dest='aux_weight', default=[1, 0.8, 0.4, 0.2, 0.1])
    parser.add_option('--reduce_size', dest='reduce_size', default=8, type='string')
    parser.add_option('--block_list', dest='block_list', default='234', type='str')
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
    print('done')

    sys.exit(0)




