from get_parameters import *
from InitializeDict_M_Coeff import *
from optimization_problems import * 
from transfer_classifiers import * 

segmentation_masks, roi_masks= get_masks('./masks/')
images=np.load('MRI_Images1.npy')


image_global_descriptors = []
image_weights = []
gamma=1
lamda=1


#for every image we will do the following:

local_descriptors, ListOfCoordinates = compute_descriptors(images[0], segmentation_masks[0])

image_weight = image_specific_SVM(ListOfCoordinates, local_descriptors, segmentation_masks[0])

#subregions!!

image_global_descriptor = global_descriptor(images[0], local_descriptors)



#and then we initiate the dictionaries:

Dx, Dw = Initialize_Dictionaries(image_global_descriptor, image_weights, 50)
Coeff_W = Initialize_Coefficients(image_weights, Dw)
M=Initilize_Mapping(Dx, Dw, 50)


#now we optimize Dx, Dw, Coeff_X, Coeff_W, M

Dx, Dw, M, Coeff_X, Coeff_W = \
    optimization_problem(image_weights, image_global_descriptor, Dx, Dw, M, Coeff_W, gamma, lamda)


#now for new/testing images

testing_images=np.load('MRI_Images2.npy')
image_descriptor, weights =transfer_classifiers(Dx, Dw, M, testing_images[0], lamda, all_descriptors)