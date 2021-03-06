# WiTeacher

This is the PyTorch source code for the WiTeacher paper. The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

WiTeacher aims at boosting activity recognition on cross-domain scenarios based on  WiFi.

## Dataset

The two public datasets used in the paper are shown below.

### DeepSeg Dataset

The data that we extract from raw CSI data for our experiments can be downloaded from Baidu Netdisk or Google Drive:

Data of CSI amplitudes: Data_CsiAmplitudeCut
Baidu Netdisk: https://pan.baidu.com/s/12DwlT58PzlVAyBc-lYx1lw (Password: k8yp)
Google Drive: https://drive.google.com/drive/folders/1PLzV6ZWAauMQLf08NUkd5UeKrqyGMHgv

Manually marked Labels for CSI amplitude data: Label_CsiAmplitudeCut
Baidu: https://pan.baidu.com/s/1nY5Og4NlLb7VH5oBQ-LH9w (Password: xnra)
Google: https://drive.google.com/drive/folders/1855zX-93QjmAt2wSeJk0rTJRiPaFMGBd
(1	boxing; 2	hand swing; 3	picking up; 4	hand raising; 5	running; 6	pushing; 7	squatting; 8	drawing O; 9	walking; 10 drawing X)

Also the raw CSI data we collected can be downloaded via Baidu or Google: Data_RawCSIDat. Note that there is no need to download the raw CSI data for running our experiments. 
Downloading Data_CsiAmplitudeCut and Label_CsiAmplitudeCut is enough for our experiments.
Baidu: https://pan.baidu.com/s/1FpA2u_fzFIh4FuNIcWOPdQ (Password: hhcv)
Google: https://drive.google.com/drive/folders/1vUeJYChsDgBzv7bJbiKDEfAHQje3SW9G

### SignFi Dataset

The SignFi dataset comes from the link below:
https://github.com/yongsen/SignFi

## Requirement

Python3.7

Pytorch 1.8.0

The codes are tested under window7 and it should be ok for Ubuntu.

## Folder descriptions:

*01Data_PreProcess:* This is used to extract the data in CSI format from the original WiFi and convert it into PNG format in order to make better use of the data.

*02Training_Generator:* This is used to train the generator G1 and G2 which are proposed in the article so as to generate target-like samples based on the source data by StyleGAN.

*03Activity_Recognition:* Based on the semi-supervised classification framework: Mean Teacher,  the label smoothing-based classification loss and the sample relation-based
consistency regularization term are added to significantly improve the performance of activity recognition.

## Motivation for WiTeacher

Despite of significant success on cross-domain activity recognition, the few-shot learning-based and data augmentation-based approaches still exist some shortages. First, few-shot learning-based methods still need a few labeled samples from the terminal users. 
But collecting a few labeled samples is still difficult, especially for old terminal users. Second, data augmentation-based methods generally consider all the generated data to possess the same quality. However, GANs are typically unstable and prone to failure, and correspondingly generated
samples may exhibit various levels of quality, i.e., some may be like real samples and others may be quite noised. Third, existing methods only consider each sample separately during model training, and ignore the relationships between samples, which can be explored to enhance model robustness.

To address these issues, we present a Mean Teacher-based cross-domain activity recognition framework using WiFi CSI, WiTeacher.

## WiTeacher Overview

In this framework, we designed a adaptive label smoothing method to produce proper soft labels for target-like samples generated by StyleGAN. Based on these
target-like samples with soft labels, we built a label smoothing-based classification loss to promote the generalization capacity of the model. Further, we presented a sample relation-based
consistency regularization term to force the distance of two samples to be consistent with the augmented ones, which can make the model more robust.
