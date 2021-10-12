# Medical Segmentation Decathlon: Multiclass Segmentation with U-Net and Generalized Dice Coefficient

<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>***Liver & Tumors***
<span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>***Spleen***

<p align="center">
  <img src="https://user-images.githubusercontent.com/88136596/136297704-f872e328-096d-4a7d-96b0-a55f8bd420de.gif">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img src="https://user-images.githubusercontent.com/88136596/136198937-c88385bb-a741-4115-89dc-d07ae7051649.gif">
</p>
<br/>

## System

### Hardware
#### CPU: Intel(R) Core(TM) i7-10510U.
#### GPU: NVIDIA GeForce MX250 (2GB)
#### RAM: 16GB
<br/>

### Environment and Requirements
#### Windows 10
#### Spyder 5.0.5
#### Python 3.8.10
#### TensorFlow-GPU 2.5.0 
#### cudatoolkit 11.0.221
#### cudnn 8.2.1
#### imageio 2.9.0
#### matplotlib 3.3.4
#### nibabel 3.2.1
#### numpy 1.19.5
#### scikit-image 0.18.1
#### opencv-python 4.5.3.56
<br/>

## Background

With recent advances in machine learning, semantic segmentation algorithms are becoming increasingly general-purpose and translatable to unseen tasks. Many key algorithmic advances in the field of medical imaging are commonly validated on a small number of tasks, limiting our understanding of the generalisability of the proposed contributions. A model which works out-of-the-box on many tasks, in the spirit of AutoML (Automated Machine Learning), would have a tremendous impact on healthcare. The field of medical imaging is also missing a fully open source and comprehensive benchmark for general-purpose algorithmic validation and testing covering a large span of challenges, such as: small data, unbalanced labels, large-ranging object scales, multi-class labels, and multimodal imaging, etc.

<br/>

## Objective
To address these problems, in this project, as part of the **MSD challenge**, I propose a generic **machine learning segmentation algorithm** which I applied on two organs: ***liver & tumors, spleen***. I propose an **unsupervised generic multi-class model by implementing U-net CNN architecture with Generalized Dice Coefficient** as metric and also for loss function.

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

#### Training data: 41 pairs of 3D image-mask.
#### Test data: 20 3D images.
#### Target: Spleen.
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
  <img src="https://user-images.githubusercontent.com/88136596/136672510-c2e84702-6dc7-4f27-9ace-b8c525db4825.png" width="30%" hight="30">
</p>
<br/>

## Generalized Dice Score as Metric and Generalized Dice Loss as Loss Function
<br/>
<p align="left">
  <img src="https://user-images.githubusercontent.com/88136596/136677128-0672f440-227b-413f-a4c7-a0823f7e0122.png" width="75%" hight="75%">
</p>

**Note**: *Generalized Dice Score metric much better than Sparse Categorical Cross-Entropy metric. When I tried to implement the model on the liver task with the 3 labels (label 0 is background) with Sparse Categorical Cross-Entropy I got training accuracy artificially high (> 98%), hence, I got overfitting. That because label 0 was being included in the loss calculation. So I decided to implement a Generalized Dice Score on our model.
Much better than.*

<br/>

## 'Adam' Optimizer
<br/>
<p align="left">
  <img src="https://user-images.githubusercontent.com/88136596/136677159-ca245859-b12a-4a88-b49e-38bf6dfe1c89.png" width="60%" hight="60%">
</p>

**Note**: *It's critical to use a very low learning rate because we have a large model. As a result, we are at risk of overfitting very quickly if we apply large weight updates. Here, we only want to readapt the pre-trained weights in an incremental way.*

<br/>

Source: [here](https://www.deeplearningbook.org/)

<br/>

## Model Architecture: Multi-Class U-Net
<br/>
<p align="left">
  <img src="https://user-images.githubusercontent.com/88136596/136843152-0492fd61-08a7-4c66-900b-dd140a5f6610.PNG">
</p>

### U-net Advantages

• This architecture contains links between its first “increasing features resolution/decrease image resolution” and its second “decreasing feature resolution/upscaling image resolution”. These links consist of saving snapshots of the weights during the first phase of the network and copying them to the second phase of the network. This makes the network combining features from different spatial regions of the image and allows it to localize more precisely regions of interests.

• The “U-Net” doesn’t need multiple runs to perform image segmentation and can learn with very few labeled images that are well suited for image segmentation in biology or medicine.

• The “U-Net” doesn’t need multiple runs to perform image segmentation and can learn with very few labeled images that are well suited for image segmentation in biology or medicine.

• In this architecture, the network is input image size agnostic since it does not contain fully connected layers. This also leads to a smaller model weight size.

• Can be easily scaled to have multiple classes.

• Relatively easy to understand why architecture works, if you have a basic understanding of how convolutions work.

• Architecture works well with a small training set, thanks to the robustness provided with data augmentation.

<br/>

## Experiments and Results

#### Training duration: 8 hours (for each segmentation task).
#### Batch size: 5 (due to lack of memory)
#### Epochs: 20
#### Validation split: 20%
 
<br/>

### Liver & Tumors

#### Training Process

<br/>
<p align="left">
  <img src="https://user-images.githubusercontent.com/88136596/136902983-c61e02f2-9329-4b99-aa4f-833e8332f30d.png" width="75%" hight="75%">
</p>
<br/>

#### Exemples
<br/>
<p align="left">
  <img src="https://user-images.githubusercontent.com/88136596/136909118-9ef79a78-dc06-4a78-9f46-c25835bd6ff0.png" width="75%" hight="75%">
</p>
<br/>

### Spleen

#### Training Process
<br/>
<p align="left">
  <img src="https://user-images.githubusercontent.com/88136596/136905028-a027e411-eb64-4a27-9441-a7d943e5d9f7.png" width="75%" hight="75%">
</p>
<br/>

#### Exemples
<br/>
<p align="left">
  <img src="https://user-images.githubusercontent.com/88136596/136909740-627568f5-8fec-4c26-8c3e-4ce4f0a2fe94.png" width="75%" hight="75%">
</p>
<br/>

## Conclusions and Summary
In this project, I developed a single generic algorithm, that can be able to generalize and work accurately across multiple different medical segmentation tasks, without the need for any human interaction. I introduced the efficiency of our algorithm by applying it to two tasks; segmentation of spleen and liver with tumors. Experimental results show that our generic model based on U-net and Generalized Dice Coefficient algorithm leads to high segmentation performance, the test Generalized Dice Coefficient reached 89% for liver and tumors and 91% for the spleen which is satisfying. All this was done without human interaction and with a relatively short run time compared to traditional segmentation methods.

<br/>

## Final Notes
### What's in the files?

• Python code files (.py)

• NifTI Data files (.nii.gz) folders (.tar)

• Pridected masks images (.png) folders (.zip)

• NumPy files of the images and labels (.npy)

• Animations files (.gif)

• Best model weights files (.h5)

<br/>

#### Contributes are welcome !
#### Thank you :)
