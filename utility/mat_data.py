"""generate testing mat dataset"""
import os
import numpy as np
import h5py
from os.path import join, exists
from scipy.io import loadmat, savemat

from util import crop_center, Visualize3D, minmax_normalize, rand_crop
from PIL import Image
from utility.util import minmax_normalize,BandMinMaxQuantileStateful
import skimage
import torch
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


def splitWDC():
    imgpath = '/data/HSI_Data/Hyperspectral_Project/dc.tif'
    imggt = skimage.io.imread(imgpath)
    # 转为mat
    imggt = torch.tensor(imggt, dtype=torch.float)
    test = imggt[:, 600:800, 50:250].clone()
    train_0 = imggt[:, :600, :].clone()
    train_1 = imggt[:, 800:, :].clone()
    val = imggt[:, 600:656, 251:].clone()

    normalizer = BandMinMaxQuantileStateful()

    # fit train
    normalizer.fit([train_0, train_1])
    train_0 = normalizer.transform(train_0).cpu().numpy()
    train_1 = normalizer.transform(train_1).cpu().numpy()

    # fit test
    normalizer.fit([test])
    test = normalizer.transform(test).cpu().numpy()

    # val test
    normalizer.fit([val])
    val = normalizer.transform(val).cpu().numpy()

    savemat("/WDC/train/train_0.mat", {'data': train_0})
    savemat("WDC/train/train_1.mat", {'data': train_1})
    savemat("WDC/test/test.mat", {'data': test})
    savemat("WDC/val/val.mat", {'data': val})

if __name__ == '__main__':
    create_big_apex_dataset()

