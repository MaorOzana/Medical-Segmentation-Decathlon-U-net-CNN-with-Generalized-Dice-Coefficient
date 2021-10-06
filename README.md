# Medical Segmentation Decathlon: U-net CNN with Generalized Dice Coefficient





<p align="center">
  <img src="https://user-images.githubusercontent.com/88136596/136197398-a584b511-a82e-4b7a-a80c-bfac32c19428.gif" alt="my boat" />
  <img src="https://user-images.githubusercontent.com/88136596/136198937-c88385bb-a741-4115-89dc-d07ae7051649.gif" hspace="20" alt="dfsa" />
</p>

With recent advances in machine learning, semantic segmentation algorithms are becoming increasingly general-purpose and translatable to unseen tasks. Many key algorithmic advances in the field of medical imaging are commonly validated on a small number of tasks, limiting our understanding of the generalisability of the proposed contributions. A model which works out-of-the-box on many tasks, in the spirit of AutoML (Automated Machine Learning), would have a tremendous impact on healthcare. The field of medical imaging is also missing a fully open source and comprehensive benchmark for general-purpose algorithmic validation and testing covering a large span of challenges, such as: small data, unbalanced labels, large-ranging object scales, multi-class labels, and multimodal imaging, etc.

To address these problems, in this project, as part of the MSD challenge, I propose a generic machine learning algorithm which I applied on two organs: ***liver & tumors, spleen***. I propose an **unsupervised generic multi-class model by implementing U-net CNN architecture with Generalized Dice Coefficient** as loss function and also as a metric. 

The MSD dataset consists of dozens of medical examinations in 3D (per organ), I’ll transform the 3-dimensional data into 2-d cuts as an input of our U-net. 

Experimental results show that our generic model based on U-net and Generalized Dice Coefficient algorithm leads to high segmentation accuracy for each organ (liver and tumors, spleen), separately, without human interaction, with a relatively short run time compared to traditional segmentation methods.
