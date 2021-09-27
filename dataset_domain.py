import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import SimpleITK as sitk
from skimage.measure import label, regionprops
import math
import pdb

class CMRDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', domain='A', crop_size=256, scale=0.1, rotate=10, debug=False):

        self.mode = mode
        self.dataset_dir = dataset_dir
        self.crop_size = crop_size
        self.scale = scale
        self.rotate = rotate

        if self.mode == 'train':
            pre_face = 'Training'
            if 'C' in domain or 'D' in domain:
                print('No domain C or D in Training set')
                raise StandardError

        elif self.mode == 'test':
            pre_face = 'Testing'

        else:
            print('Wrong mode')
            raise StandardError
        if debug:
            # validation set is the smallest, need the shortest time for load data.
           pre_face = 'Testing'

        path = self.dataset_dir + pre_face + '/'
        print('start loading data')
        
        name_list = []

        if 'A' in domain:
            df = pd.read_csv(self.dataset_dir+pre_face+'_A.csv')
            name_list += np.array(df['name']).tolist()
        if 'B' in domain:
            df = pd.read_csv(self.dataset_dir+pre_face+'_B.csv')
            name_list += np.array(df['name']).tolist()
        if 'C' in domain:
            df = pd.read_csv(self.dataset_dir+pre_face+'_C.csv')
            name_list += np.array(df['name']).tolist()
        if 'D' in domain:
            df = pd.read_csv(self.dataset_dir+pre_face+'_D.csv')
            name_list += np.array(df['name']).tolist()



        
        img_list = []
        lab_list = []
        spacing_list = []

        for name in name_list:
            for name_idx in os.listdir(path+name):
                if 'gt' in name_idx:
                    continue
                else:
                    idx = name_idx.split('_')[2].split('.')[0]
                    
                    itk_img = sitk.ReadImage(path+name+'/%s_sa_%s.nii.gz'%(name, idx))
                    itk_lab = sitk.ReadImage(path+name+'/%s_sa_gt_%s.nii.gz'%(name, idx))
                    
                    spacing = np.array(itk_lab.GetSpacing()).tolist()
                    spacing_list.append(spacing[::-1])

                    assert itk_img.GetSize() == itk_lab.GetSize()
                    img, lab = self.preprocess(itk_img, itk_lab)

                    img_list.append(img)
                    lab_list.append(lab)

       
        self.img_slice_list = []
        self.lab_slice_list = []
        if self.mode == 'train':
            for i in range(len(img_list)):
                tmp_img = img_list[i]
                tmp_lab = lab_list[i]

                z, x, y = tmp_img.shape

                for j in range(z):
                    self.img_slice_list.append(tmp_img[j])
                    self.lab_slice_list.append(tmp_lab[j])

        else:
            self.img_slice_list = img_list
            self.lab_slice_list = lab_list
            self.spacing_list = spacing_list

        print('load done, length of dataset:', len(self.img_slice_list))
        
    def __len__(self):
        return len(self.img_slice_list)

    def preprocess(self, itk_img, itk_lab):
        
        img = sitk.GetArrayFromImage(itk_img)
        lab = sitk.GetArrayFromImage(itk_lab)

        max98 = np.percentile(img, 98)
        img = np.clip(img, 0, max98)
            
        z, y, x = img.shape
        if x < self.crop_size:
            diff = (self.crop_size + 10 - x) // 2
            img = np.pad(img, ((0,0), (0,0), (diff, diff)))
            lab = np.pad(lab, ((0,0), (0,0), (diff,diff)))
        if y < self.crop_size:
            diff = (self.crop_size + 10 -y) // 2
            img = np.pad(img, ((0,0), (diff, diff), (0,0)))
            lab = np.pad(lab, ((0,0), (diff, diff), (0,0)))

        img = img / max98
        tensor_img = torch.from_numpy(img).float()
        tensor_lab = torch.from_numpy(lab).long()

        return tensor_img, tensor_lab


    def __getitem__(self, idx):
        tensor_image = self.img_slice_list[idx]
        tensor_label = self.lab_slice_list[idx]
       
        if self.mode == 'train':
            tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)
            tensor_label = tensor_label.unsqueeze(0).unsqueeze(0)
            
            # Gaussian Noise
            tensor_image += torch.randn(tensor_image.shape) * 0.02
            # Additive brightness
            rnd_bn = np.random.normal(0, 0.7)#0.03
            tensor_image += rnd_bn
            # gamma
            minm = tensor_image.min()
            rng = tensor_image.max() - minm
            gamma = np.random.uniform(0.5, 1.6)
            tensor_image = torch.pow((tensor_image-minm)/rng, gamma)*rng + minm

            tensor_image, tensor_label = self.random_zoom_rotate(tensor_image, tensor_label)
            tensor_image, tensor_label = self.randcrop(tensor_image, tensor_label)
        else:
            tensor_image, tensor_label = self.center_crop(tensor_image, tensor_label)
        
        assert tensor_image.shape == tensor_label.shape
        
        if self.mode == 'train':
            return tensor_image, tensor_label
        else:
            return tensor_image, tensor_label, np.array(self.spacing_list[idx])

    def randcrop(self, img, label):
        _, _, H, W = img.shape
        
        diff_H = H - self.crop_size
        diff_W = W - self.crop_size
        
        rand_x = np.random.randint(0, diff_H)
        rand_y = np.random.randint(0, diff_W)
        
        croped_img = img[0, :, rand_x:rand_x+self.crop_size, rand_y:rand_y+self.crop_size]
        croped_lab = label[0, :, rand_x:rand_x+self.crop_size, rand_y:rand_y+self.crop_size]

        return croped_img, croped_lab


    def center_crop(self, img, label):
        D, H, W = img.shape
        
        diff_H = H - self.crop_size
        diff_W = W - self.crop_size
        
        rand_x = diff_H // 2
        rand_y = diff_W // 2

        croped_img = img[:, rand_x:rand_x+self.crop_size, rand_y:rand_y+self.crop_size]
        croped_lab = label[:, rand_x:rand_x+self.crop_size, rand_y:rand_y+self.crop_size]

        return croped_img, croped_lab

    def random_zoom_rotate(self, img, label):
        scale_x = np.random.random() * 2 * self.scale + (1 - self.scale)
        scale_y = np.random.random() * 2 * self.scale + (1 - self.scale)


        theta_scale = torch.tensor([[scale_x, 0, 0],
                                    [0, scale_y, 0],
                                    [0, 0, 1]]).float()
        angle = (float(np.random.randint(-self.rotate, self.rotate)) / 180.) * math.pi

        theta_rotate = torch.tensor( [  [math.cos(angle), -math.sin(angle), 0], 
                                        [math.sin(angle), math.cos(angle), 0], 
                                        ]).float()
        
    
        theta_rotate = theta_rotate.unsqueeze(0)
        grid = F.affine_grid(theta_rotate, img.size())
        img = F.grid_sample(img, grid, mode='bilinear')
        label = F.grid_sample(label.float(), grid, mode='nearest').long()
    
        return img, label


