# Medical Segmentation Decathlon: U-net with Generalized Dice Coefficient

<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>***Liver & Tumors***
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>***Spleen***

<p align="center">
  <img src="https://user-images.githubusercontent.com/88136596/136297704-f872e328-096d-4a7d-96b0-a55f8bd420de.gif">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img src="https://user-images.githubusercontent.com/88136596/136198937-c88385bb-a741-4115-89dc-d07ae7051649.gif">
</p>
<br/>

## Background

With recent advances in machine learning, semantic segmentation algorithms are becoming increasingly general-purpose and translatable to unseen tasks. Many key algorithmic advances in the field of medical imaging are commonly validated on a small number of tasks, limiting our understanding of the generalisability of the proposed contributions. A model which works out-of-the-box on many tasks, in the spirit of AutoML (Automated Machine Learning), would have a tremendous impact on healthcare. The field of medical imaging is also missing a fully open source and comprehensive benchmark for general-purpose algorithmic validation and testing covering a large span of challenges, such as: small data, unbalanced labels, large-ranging object scales, multi-class labels, and multimodal imaging, etc.

<br/>

## Objective
To address these problems, in this project, as part of the MSD challenge, I propose a generic **machine learning segmentation algorithm** which I applied on two organs: ***liver & tumors, spleen***. I propose an **unsupervised generic multi-class model by implementing U-net CNN architecture with Generalized Dice Coefficient** as loss function and also as a metric.

<br/>

## MSD Datasets
In general, the Decathlon challenge made ten datasets available online, where each dataset had between one and three region-of-interest (ROI) targets. All 3D images (2,633 in total) were acquired across multiple institutions, anatomies, and modalities during real-world clinical applications. The images were de-identified and reformatted to the Neuroimaging Informatics Technology Initiative (NIfTI) format. All images were transposed (without resampling) to the most approximate right-anterior-superior coordinate frame, ensuring the data matrix x-y-z direction was consistent. For each segmentation task, a pixel-level label annotation was provided depending on the definition of each specific task.

<br/>

### Liver & Tumors Dataset
The data set consists of 201 contrast-enhanced CT 3D images from patients with primary cancers and metastatic liver disease, as a consequence of colorectal, breast, and lung primary cancers. The corresponding target ROIs were the segmentation of the liver and tumors inside the liver. This data set was selected due to the challenging nature of having a significant label unbalance between large (liver) and small (tumor) target ROIs. The data was acquired in the IRCAD Hôpitaux Universitaires, Strasbourg, France and contained a subset of patients from the 2017 Liver Tumor Segmentation (LiTS) challenge. 

#### Training data: 131 pairs of 3D image-mask.
#### Test data: 70 3D images.
#### Target: Liver and tumors.
#### Mask labels: {‘0’ - background , ‘1’ - liver , ‘2’ - tumors}
<br/>
<p align="left">
  <img src="https://user-images.githubusercontent.com/88136596/136630170-b90d9feb-9913-4ff1-9430-8829b78a0b3b.png" width="45%" hight="45%">
</p>
<br/>

### Spleen Dataset
The dataset consists of 61 portal venous phase CT scans from patients undergoing chemotherapy treatment for liver metastases. The corresponding target ROI was the spleen. This data set was selected due to the large variations in the field-of-view. The data was acquired in the Memorial Sloan Kettering Cancer Center, New York, US. 

#### Training data: 131 pairs of 3D image-mask.
#### Test data: 70 3D images.
#### Target: Liver and tumors.
#### Mask labels: {‘0’ - background , ‘1’ - spleen}
<br/>
<p align="left">
  <img src="https://user-images.githubusercontent.com/88136596/136631736-e9c5ddf6-5fec-433d-86bc-67f9d69b398e.png" width="45%" hight="45%">
</p>
<br/>

## Preprocess

### From 3D to 2D
Each dataset consists of dozens of medical examinations in 3D, we’ll transform the 3-dimensional data into 2-d cuts as an input of our U-net. Namely, we’ll take each 3-dimensional volume and divide it into slices, hence, we’ll take the slices of all the 3D images and concatenated them together to a stack and thus we get a new augmented dataset.

### Downsampling the Data
The training dataset consists of an enormous amount of 2D slices, so in order to overcome overfitting we’ll downsample our data with a factor of 2 by downsampling each image (slice) in a half, i.e., take every second pixel lengthwise and widthwise of the image (in my case; 512/2). Also, in this way memory saving increased runtime speed.

### Data Normalization
It is necessary to transfer the range of values of the training and test images to match the range of values of the masks, hence we will perform data normalization, by applying this transformation:
<p align="left">
  <img src="https://user-images.githubusercontent.com/88136596/136672510-c2e84702-6dc7-4f27-9ace-b8c525db4825.png" width="25%" hight="25">
</p>
<br/>

## Generalized Dice Score as Metric and Generalized Dice Loss as Loss Function
<br/>
<p align="left">
  <img src="https://user-images.githubusercontent.com/88136596/136677128-0672f440-227b-413f-a4c7-a0823f7e0122.png" width="75%" hight="75%">
</p>
<br/>

## 'Adam' Optimizer
<br/>
<p align="left">
  <img src="https://user-images.githubusercontent.com/88136596/136677159-ca245859-b12a-4a88-b49e-38bf6dfe1c89.png" width="60%" hight="60%">
</p>

Source: [here](https://www.deeplearningbook.org/)

<br/>

## Model Architecture 
<br/>
<p align="left">
  <img src="https://user-images.githubusercontent.com/88136596/136843152-0492fd61-08a7-4c66-900b-dd140a5f6610.PNG">
</p>

<br/>

Experimental results show that my generic model based on U-net and Generalized Dice Coefficient algorithm leads to high segmentation accuracy for each organ (liver and tumors, spleen), separately, without human interaction, with a relatively short run time compared to traditional segmentation methods.
