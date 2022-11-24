import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import pdb 


import os


from skimage import io,color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F2

from typing import Callable

import cv2
import pandas as pd

from numbers import Number
from typing import Container
from collections import defaultdict
import datetime
import random


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()

def load_image(input_path,output_path,train_input_name,train_out_name ):#, cv2.COLOR_BGR2RGB
        image = cv2.imread(os.path.join(input_path, train_input_name))
        # print(image.shape)
        # read mask image
       
        mask = cv2.imread(os.path.join(output_path, train_out_name),0)
               
        mask[mask<=127] = 0
        mask[mask>127] = 1
        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
#        print(image.shape)
#
#        image=T.ToTensor()(image)
#        mask=T.ToTensor()(mask)

        # mask = np.swapaxes(mask,2,0)
        print(image.shape)
        print(mask.shape)
        # mask = np.transpose(mask,(2,0,1))
        # image = np.transpose(image,(2,0,1))
        # print(image.shape)
        # print(mask.shape)

        return image, mask


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images
def prepare_data(dataset_dir):
    train_input_names=[]
    train_output_names=[]
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]
    for file in os.listdir(dataset_dir + "//train"):#获取文件夹内的文件
#        cwd = os.getcwd()#获取路径
        train_input_names.append(dataset_dir + "//train//" + file)#文件路径"CamVid" cwd + "/" + 
    for file in os.listdir(dataset_dir + "//train_labels"):
#        cwd = os.getcwd()
        train_output_names.append( dataset_dir + "//train_labels//" + file)
    for file in os.listdir(dataset_dir + "//test"):
#        cwd = os.getcwd()
        val_input_names.append( dataset_dir + "//test//" + file)
    for file in os.listdir(dataset_dir + "//test_labels"):
#        cwd = os.getcwd()
        val_output_names.append( dataset_dir + "//test_labels//" + file)
    for file in os.listdir( dataset_dir + "//test"):
#        cwd = os.getcwd()
        test_input_names.append( dataset_dir + "//test//" + file)
    for file in os.listdir(dataset_dir + "//test_labels"):
#        cwd = os.getcwd()
        test_output_names.append(dataset_dir + "//test_labels//" + file)
#        获取训练集测试集验证集的图像及label路径
    train_input_names.sort(),train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()
    
#  
    return train_input_names,train_output_names,val_input_names,val_output_names,test_input_names, test_output_names
def data_augmentation(input_image, output_image,h_flip=1,v_flip=1,brightness=0.5,rotation=15):
    # Data augmentation
    input_image, output_image = random_crop(input_image, output_image, crop_height=384, crop_width=384)

    if h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if brightness and random.randint(0,1):
        factor = 1.0 + random.uniform(-1.0*brightness, brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if rotation and random.randint(0,1):
        angle = random.uniform(-1*rotation, rotation)
#    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)
#    input_image = np.transpose(input_image,(2,0,1))
#    output_image = np.transpose(output_image,(2,0,1))
    return input_image, output_image

def random_crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)
        
        if len(label.shape) == 3:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (crop_height, crop_width, image.shape[0], image.shape[1]))
class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """
    def __init__(self, crop=False, p_flip=0, color_jitter_params=False, #(0.1, 0.1, 0.1, 0.1),
                 p_random_affine=180, long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        # transforming to PIL image
        image, mask = F2.to_pil_image(image), F2.to_pil_image(mask)

        # random crop
#        if self.crop:
#            if (image.shape[0] != mask.shape[0]) or (image.shape[1] != mask.shape[1]):
#                raise Exception('Image and label must have the same dimensions!')
#                
#            if (self.crop <= image.shape[1]) and (self.crop <= image.shape[0]):
#                x = random.randint(0, image.shape[1]-self.crop)
#                y = random.randint(0, image.shape[0]-self.crop)
#                
#                if len(mask.shape) == 3:
#                    return image[y:y+self.crop, x:x+self.crop, :], mask[y:y+self.crop, x:x+self.crop, :]
#                else:
#                    return image[y:y+self.crop, x:x+self.crop, :], mask[y:y+self.crop, x:x+self.crop]
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F2.crop(image, i, j, h, w), F2.crop(mask, i, j, h, w)  

           
        if np.random.rand() < self.p_flip:
            image, mask = F2.hflip(image), F2.hflip(mask)

        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)

##         random affine transform
#        if np.random.rand() < self.p_random_affine:
#            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), (256,256))
#            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        # transforming to tensor
        image = F2.to_tensor(image)
        if not self.long_mask:
            mask = F2.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        return image, mask


class ImageToImage2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        #print(image_filename[: -3])
        # read image
        # print(os.path.join(self.input_path, image_filename))
        # print(os.path.join(self.output_path, image_filename[: -3] + "png"))
        # print(os.path.join(self.input_path, image_filename))
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        # print(image.shape)
        # read mask image
        if os.path.exists(os.path.join(self.output_path, image_filename[: -3] + "png")):
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "png"),0)
        else:
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "bmp"),0)   
            
        
        mask[mask<=127] = 0
        mask[mask>127] = 1
        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
#        print(image.shape)

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
#            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
            mask=to_long_tensor(mask)
        # mask = np.swapaxes(mask,2,0)
        # print(image.shape)
        # print(mask.shape)
        # mask = np.transpose(mask,(2,0,1))
        # image = np.transpose(image,(2,0,1))
        # print(image.shape)
        # print(mask.shape)

        return image, mask, image_filename


class Image2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them. As opposed to ImageToImage2D, this
    reads a single image and requires a simple augmentation transform.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as a prediction
           dataset.

    Args:
        
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        transform: augmentation transform. If bool(joint_transform) evaluates to False,
            torchvision.transforms.ToTensor will be used.
    """

    def __init__(self, dataset_path: str, transform: Callable = None):

        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'img')
        self.images_list = os.listdir(self.input_path)

        if transform:
            self.transform = transform
        else:
            self.transform = T.ToTensor()

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]

        image = cv2.imread(os.path.join(self.input_path, image_filename))

        # image = np.transpose(image,(2,0,1))

        image = correct_dims(image)

        image = self.transform(image)

        # image = np.swapaxes(image,2,0)

        return image, image_filename

def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.

    Args:        
        paths: Container of paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)
        
        
class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)


class MetricList:
    def __init__(self, metrics):
        assert isinstance(metrics, dict), '\'metrics\' must be a dictionary of callables'
        self.metrics = metrics
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def __call__(self, y_out, y_batch):
        for key, value in self.metrics.items():
            self.results[key] += value(y_out, y_batch)

    def reset(self):
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def get_results(self, normalize=False):
        assert isinstance(normalize, bool) or isinstance(normalize, Number), '\'normalize\' must be boolean or a number'
        if not normalize:
            return self.results
        else:
            return {key: value/normalize for key, value in self.results.items()}


def log_evaluation_result(writer, dice_list, ASD_list, HD_list, name, epoch):
    
    writer.add_scalar('Test_Dice/%s_AVG'%name, dice_list.mean(), epoch+1)
    for idx in range(3):
        writer.add_scalar('Test_Dice/%s_Dice%d'%(name, idx+1), dice_list[idx], epoch+1)
    writer.add_scalar('Test_ASD/%s_AVG'%name, ASD_list.mean(), epoch+1)
    for idx in range(3):
        writer.add_scalar('Test_ASD/%s_ASD%d'%(name, idx+1), ASD_list[idx], epoch+1)
    writer.add_scalar('Test_HD/%s_AVG'%name, HD_list.mean(), epoch+1)
    for idx in range(3):
        writer.add_scalar('Test_HD/%s_HD%d'%(name, idx+1), HD_list[idx], epoch+1)


def multistep_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, lr_decay_epoch, max_epoch, gamma=0.1):

    if epoch >= 0 and epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    flag = False
    for i in range(len(lr_decay_epoch)):
        if epoch == lr_decay_epoch[i]:
            flag = True
            break

    if flag == True:
        lr = init_lr * gamma**(i+1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    else:
        return optimizer.param_groups[0]['lr']

    return lr

def exp_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, max_epoch):

    if epoch >= 0 and epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    else:
        lr = init_lr * (1 - epoch / max_epoch)**0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr



def cal_dice(pred, target, C): 
    N = pred.shape[0]
    target_mask = target.data.new(N, C).fill_(0)
    target_mask.scatter_(1, target, 1.) 

    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred, 1.) 

    intersection= pred_mask * target_mask
    summ = pred_mask + target_mask

    intersection = intersection.sum(0).type(torch.float32)
    summ = summ.sum(0).type(torch.float32)
    
    eps = torch.rand(C, dtype=torch.float32)
    eps = eps.fill_(1e-7)

    summ += eps.cuda()
    dice = 2 * intersection / summ

    return dice, intersection, summ

def cal_asd(itkPred, itkGT):
    
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(itkGT, squaredDistance=False))
    reference_surface = sitk.LabelContour(itkGT)

    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())

    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(itkPred, squaredDistance=False))
    segmented_surface = sitk.LabelContour(itkPred)

    seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0])
    seg2ref_distances = seg2ref_distances + \
                        list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0])
    ref2seg_distances = ref2seg_distances + \
                        list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
    
    all_surface_distances = seg2ref_distances + ref2seg_distances

    ASD = np.mean(all_surface_distances)

    return ASD

