# Code for Spatial-Spectral Transformer for Hyperspectral Image Denoising


## Prepare Dataset:
### ICVL dataset
The entire ICVL dataset download link: https://icvl.cs.bgu.ac.il/hyperspectral/

1. split the entire dataset into training, testing and validation. The files used in training are listed in utility/icvl_train_list.txt. The files used in testing can be found in [Baidu Disk](https://pan.baidu.com/s/1GqjTFCtNJkkqG4ENyNUFhQ?pwd=azx0) code:azx0


2. generate lmdb dataset for training

```
python utility/lmdb_data.py
```

3. download the testing data from BaiduDisk or generate them by yourself through

```
python utility/mat_data.py
```

## Experiement:

checkpoints are available at [Baidu Disk](https://pan.baidu.com/s/1GqjTFCtNJkkqG4ENyNUFhQ?pwd=azx0) code:azx0
### Training on ICVL dataset
```
#for gaussian noise
python hsi_denoising_single.py -a sst -p sst_gaussian -b 8 

#for comlpex noise
python hsi_denoising_complex.py -a sst -p sst_gaussian -b 8 

```
### Testing on ICVL dataset
```
#for guassian noise
python hsi_denoising_test.py -a sst -p sst_gaussian -r -rp checkepoints_guassian.pth 

#for complex noise
python hsi_denoising_test.py -a sst -p sst_complex -r -rp checkpoints/checkpoints_complex.pth
```