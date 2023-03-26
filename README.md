# MRI_Dictionary_Learning

# Overview

Let's describe the whole procedure. We follow, in general, some of the steps of Tian et al 2018, "A NOVEL IMAGE-SPECIFIC TRANSFER APPROACH FOR PROSTATE SEGMENTATION IN MR IMAGES".

We do, however, work on 3D MRI Knee scans, having 4 classes (TB, FB, FAC, TAC). We sample a lot of local SIFT-descriptors from every MRI scan (~4000 per patient, weighted sampling to have more cartilage samples) and create an image specific SVM (with the labels given by the segmentation mask).

That means, that for each model we provide all the model's descriptors into the SVM along with their label. We train the SVM and save the best weights for each image to create the SVM weight. 
>>This is the point where I currently face the difficulty of getting the best results. More on that on our personal meeting.

Next, we follow a variation of the Bag of Words algorithm. We have our set of every descriptor of every image, and we do a K-Means algorithm to define the codewords/cluster centers. 
Before that, we have defined some subregions in our image: we take the 3D ROI space and do a 2-step K-Means of coordinates to find
the (3,4,5 or whatever works best) number of subregion centers and their area.
For each of these subregions, we do a soft assignment K-Means to create a histogram of the frequency of each codeword in the subregion in a dense grid of voxels. 
Then we concat and normalize this frequency histograms to create the global image descriptor.

The whole procedure till this point is done with the functions define in **get_parameters.py** as is also how we actually extract using this methodology the atlas of weights of the SVM for each image and the atlas of global image descriptors.


 
