import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utility import *
from hsi_setup import Engine, train_options, make_dataset


if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising (Complex noise)')
    opt = train_options(parser)
    print(opt)
    

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.net.use_2dconv)

   
    target_transform = HSI2Tensor()


    """Test-Dev"""
    basefolder = '/data/HSI_Data/Hyperspectral_Project/'
    
    
    mat_datasets = [MatDataFromFolder(
        basefolder, size=1,fns=['Urban_304.mat']) ]

    
    if not engine.get_net().use_2dconv:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='input',
                    transform=lambda x:x[ ...][None], needsigma=False),
        ])
    else:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='input', needsigma=False),
        ])

    mat_datasets = [TransformDataset(mat_dataset, mat_transform)
                    for mat_dataset in mat_datasets]

 
    mat_loaders = [DataLoader(
        mat_dataset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt.no_cuda
    ) for mat_dataset in mat_datasets]       
    
    engine.test(mat_loaders[0], basefolder)
        
