# Code for Spatial-Spectral Transformer for Hyperspectral Image Denoising

The testing data of ICVL are availabel at https://pan.baidu.com/s/1GqjTFCtNJkkqG4ENyNUFhQ?pwd=azx0 
提取码：azx0 
checkpoints are available at https://drive.google.com/drive/folders/1Rd4L7YsEoHolVcPxaD8kND3fRxviMHay?usp=sharing

## Prepare Dataset:

### Training
```
python hsi_denoising_single.py -a sst -p sst_gaussian -b 8
```
### Testing
```
python hsi_denoising_test.py -a sst -p sst_gaussian -r -rp checkepoints_guassian.pth 
```