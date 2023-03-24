# MRI_Dictionary_Learning

# Overview

Let's describe the whole procedure. We follow, in general, the steps of Tian et al 2018, "A NOVEL IMAGE-SPECIFIC TRANSFER APPROACH FOR PROSTATE SEGMENTATION IN
MR IMAGES".

We do, however, work on 3D MRI Knee scans, having 4 classes (TB, FB, FAC, TAC). We sample a lot of local SIFT-descriptors from every MRI scan (~4000 per patient, weighted sampling to have more cartilage samples) and create an image specific SVM (with the labels given by the segmentation mask).

That means, that for each model we provide all the model's descriptors into the SVM along with their label. We train the SVM and save the best weights for each image to create the SVM weight. 
 

 
