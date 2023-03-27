# MRI_Dictionary_Learning

# Overview

Let's describe the whole procedure. We follow, in general, some of the steps of Tian et al 2018, "A NOVEL IMAGE-SPECIFIC TRANSFER APPROACH FOR PROSTATE SEGMENTATION IN MR IMAGES".

We do, however, work on 3D MRI Knee scans, having 4 classes (TB, FB, FAC, TAC). We sample a lot of local SIFT-descriptors from every MRI scan (~4000 per patient, weighted sampling to have more cartilage samples) and create an image specific SVM (with the labels given by the segmentation mask).

That means, that for each model we provide all the model's descriptors into the SVM along with their label. We train the SVM and save the best weights for each image to create the SVM weight. 
>>This is the point where I currently face the difficulty of getting the best results. More on that on our personal meeting.

Next, we follow a variation of the Bag of Words algorithm. We have our set of every descriptor of every image, and we do a K-Means algorithm to define the codewords/cluster centers. 
Before that, we have defined some subregions in our image: we take the 3D ROI space and do a 2-step K-Means of coordinates to find
the (3,4,5 or whatever works best) number of subregion centers and their area.
For each of these subregions, we do a soft assignment K-Means to create a histogram of the frequency of each codeword in the subregion in a dense grid of voxels. That way, we encorporate the spatial information.
Then we concat and normalize this frequency histograms to create the global image descriptor.

The whole procedure till this point is done with the functions define in **get_parameters.py** as is also how we actually extract using this methodology the atlas of weights of the SVM for each image and the atlas of global image descriptors.


 
Next up, we proceed to our main problem.
We have to create a dictionary of Weights and a dictionary of Global Descriptors and learn an intrinsic relationship between them.

Firstly, we have to initialize: 1. The Dx, Dw (dictionaries), 2. Λw (coding coefficients), 3. M (mapping matrix between the two).
Λx is computed based on those.

We initialize the dictionaries with K-Means algorithm on the Global Descriptors and Weights respectively, the coding coefficients 
by finding a least-squares solution to the linear matrix equation W=Λw*Dw and ensuring L1 norm = 1, and the mapping matrix by learning PCA components from Dx, Dw and multiplying them to obtain a transformation matrix that maps one dataset to the other.

The above procedure is done using the functions in **InitializeDict_M_Coeff.py**. 
We now have our Dx, Dw, Λw, and M, and we can now go on to the optimization problem, which is a procedure with 4 steps:
1. update Λx: we want to minimize the ||X-Dx*Λx||(F) ^2 + γ *||Λw - M * Λx||(F) ^2 + λ *||Λx||(F) ^2.
              We do this by solving its closed form solution: Λx = ( (Dx.T * Dx + γ * M.T *M + λ * I) ^−1 ) * (Dx.T * X + γ * M.T * Λw).
2. update Λw: **this is yet to be done**
3. update Dx, Dw: we solve the quadratically constrained quadratic program problem ||X-Dx*Λx||(F) ^2 + ||W - Dw * Λw||(F) ^2, with cvxpy.
4. update M: We update M by solving the closed form solution M = Λw * Λx.T *  (( Λx* Λx.T + (λ/γ) * Ι) ^-1 ).
**repeat untιl convergency**

The above procedure is done iwth the functions in **optimization_problem.py**.

Lastly, we have to transfer the classifiers to the testing images: so, we compute the image's global descriptor and we compute based on the optimal parameters found above.





The functions are there but we use pipeline.ipynb to do our small changes while we run the experiment and save results.
