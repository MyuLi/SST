"""generate testing mat dataset"""
import os
import numpy as np
import h5py
from os.path import join, exists
from scipy.io import loadmat, savemat

from util import crop_center, Visualize3D, minmax_normalize, rand_crop
from PIL import Image

def create_big_apex_dataset():
    total_num = 20
    print('processing---')
    all_data = loadmat('/data/HSI_Data/Hyperspectral_Project/apex_210.mat')['data']
    print(all_data.shape)
    save_dir = '/data/HSI_Data/Hyperspectral_Project/apex_crop/'
    for i in range(total_num):
        data = rand_crop(all_data, 512, 512)
        savemat(save_dir+str(i)+'.mat',{'data': data})
        print(i)

if __name__ == '__main__':
    create_big_apex_dataset()

