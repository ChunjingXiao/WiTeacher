# CLAR


This is the PyTorch source code for the CLAR paper. 
The code runs on Python 3. 
Install the dependencies and prepare the datasets with the following commands:


CLAR use a diffusion model-based Contrastive self-supervised Learning framework for Activity Recognition using WiFi CSI.


##Dataset


The two public datasets used in the paper are shown below.\


###DeepSeg Dataset


The data that we extract from raw CSI data for our experiments can be downloaded from Baidu Netdisk or Google Drive:


Data of CSI amplitudes: Data_CsiAmplitudeCut Baidu Netdisk: https://pan.baidu.com/s/12DwlT58PzlVAyBc-lYx1lw (Password: k8yp) Google Drive: https://drive.google.com/drive/folders/1PLzV6ZWAauMQLf08NUkd5UeKrqyGMHgv


Manually marked Labels for CSI amplitude data: Label_CsiAmplitudeCut Baidu: https://pan.baidu.com/s/1nY5Og4NlLb7VH5oBQ-LH9w (Password: xnra) Google: https://drive.google.com/drive/folders/1855zX-93QjmAt2wSeJk0rTJRiPaFMGBd (1 boxing; 2 hand swing; 3 picking up; 4 hand raising; 5 running; 6 pushing; 7 squatting; 8 drawing O; 9 walking; 10 drawing X)



Also the raw CSI data we collected can be downloaded via Baidu or Google: Data_RawCSIDat. Note that there is no need to download the raw CSI data for running our experiments. Downloading Data_CsiAmplitudeCut and Label_CsiAmplitudeCut is enough for our experiments. Baidu: https://pan.baidu.com/s/1FpA2u_fzFIh4FuNIcWOPdQ (Password: hhcv) Google: https://drive.google.com/drive/folders/1vUeJYChsDgBzv7bJbiKDEfAHQje3SW9G




###SignFi Dataset

The SignFi dataset comes from the link below: https://github.com/yongsen/SignFi


##Requirement

Python 3.7

Tensorflow 2.4.1

The codes are tested under window10.


##Folder descriptions:

01Data_PreProcess: This is used to extract the data in CSI format from the original WiFi and convert it into PNG format in order to make better use of the data.


02DataGenerator: This is used to generate augmented samples based on the source data  through the augmentation method proposed in this paper.


03ActivityRecognition: Based on the self-supervised learning framework: SimCLR, DDPM-based time series-specific augmentation method and the adaptive weight algorithm are added to significantly improve the performance of activity recognition.


##Motivation for CLAR



While most of the models are typically powered by supervised machine learning algorithms, where a large training dataset with annotations is needed to maintain an acceptable performance, makes the training phase time consuming, labor intensive, and expensive.
Consequently, collecting numerous labeled data is one of the major hurdles in applying these methods for practical applications. Contrastive learning, in which models learn to extract representations by contrasting positive pairs (samples which are deemed to be similar) against negative pairs, has shown superior performance in the image processing and natural language processing.
However, directly applying contrastive learning to activity recognition tasks is confronted with two additional issues.
First, general data augmentation operations in contrastive learning might be ineffective for CSI data.
Prevailing augmentation approach, such as Gaussian blur and color distortion, are particularly designed for image data, which can hardly change the shape of the CSI waveform, a kind of time series data.
Second, typical contrastive learning models fail to consider the difference of the sample importance during model training. 
In contrastive learning, the same weights are generally assigned to all the positive sample pairs for model training.
However, for CSI-based activity recognition, different positive sample pairs might provide various clues for learning data representation. 



To address these issues, we propose a diffusion model-based Contrastive self-supervised Learning framework for human Activity Recognition (CLAR) using WiFi CSI.






##CLAR Overview

In this framework, we designed a DDPM-based time series-specific augmentation method, which can combine two samples from users with different motion habits to generate augmented samples with compromised characteristics for amplifying training data and enhancing generalization capacity.
Also, we presented an adaptive weight algorithm, which can adaptively adjust the weights of positive sample pairs for learning better data representations.








