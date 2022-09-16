# Code for Spatial-Spectral Transformer for Hyperspectral Image Denoising


## Prepare Dataset
All the testing data are avalibal at [Baidu Disk](https://pan.baidu.com/s/1GqjTFCtNJkkqG4ENyNUFhQ?pwd=azx0) code:azx0. You can also generate testing samples byyourself through utility/dataset.py.

### ICVL dataset
* The entire ICVL dataset download link: https://icvl.cs.bgu.ac.il/hyperspectral/
1. split the entire dataset into training samples, testing samples and validating samples. The files used in training are listed in utility/icvl_train_list.txt.

2. generate lmdb dataset for training

```
python utility/lmdb_data.py
```

3. download the testing data from BaiduDisk or generate them by yourself through

```
python utility/mat_data.py
```

### WDC dataset
* The entire WDC dataset download link: https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html

The codes for split it to traning, testing, validating are available at utility/mat_data.py splitWDC().  Run the createDCmall() function in utility/lmdb_data.py to generate training lmdb dataset.
### Urban dataset
* The training dataset are from link: https://apex-esa.org/. The origin Urban dataset are from link: 

1. Run the create_big_apex_dataset() funtion in utility/mat_data.py to generate training samples.

2. Run the createDCmall() function in utility/lmdb_data.py to generate training lmdb dataset.


## Experiement:

checkpoints are available at https://rslab.ut.ac.ir/data.

* [Baidu Disk](https://pan.baidu.com/s/1GqjTFCtNJkkqG4ENyNUFhQ?pwd=azx0) code:azx0  

* [Google Driver](https://drive.google.com/drive/folders/1Rd4L7YsEoHolVcPxaD8kND3fRxviMHay?usp=sharing)
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
python hsi_denoising_test.py -a sst -p sst_gaussian -r -rp checkpoints/checkepoints_guassian.pth  --testdir  /data/HSI_Data/icvl_noise_50/512_random/

#for complex noise
python hsi_denoising_test.py -a sst -p sst_complex -r -rp checkpoints/checkpoints_complex.pth --testdir  /data/HSI_Data/icvl_noise_50/512_noniid/
```
***
### Training on Wdc dataset
```
#for gaussian noise
python hsi_denoising_single.py -a sst_wdc -p sst_gaussian -b 8 

#for comlpex noise
python hsi_denoising_complex.py -a sst_wdc -p sst_gaussian -b 8 

```
### Testing on Wdc dataset
```
#for guassian noise
python hsi_denoising_test.py -a sst_wdc -p sst_gaussian -r -rp checkpoints/wdc_gaussian.pth --testdir /data/HSI_Data/Hyperspectral_Project/WDC_noise/fixed


#for complex noise
python hsi_denoising_test.py -a sst_wdc -p sst_complex -r -rp checkpoints/wdc_complex.pth --testdir /data/HSI_Data/Hyperspectral_Project/WDC_noise/mix
```
***
### Training for real dataset
```
python hsi_denoising_single.py -a sst -p sst_gaussian -b 8 
```
### Testing on real dataset
```

python hsi_denoising_test.py -a sst -p sst_complex -r -rp checkpoints/checkpoints_complex.pth

```