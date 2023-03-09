
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2 
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans 
from numpy.linalg import norm
import os


sift = cv2.xfeatures2d.SIFT_create()



#This function will be used to access the file location of the MRI's
#and the masks

def get_lists_of_of_paths(directory , ):
    file_list = os.listdir(directory)
    file_paths = []
    for file in file_list:
        file_path = os.path.join(directory, file)
        file_paths.append(file_path)

    return file_paths  




# We first we convert all the image files from mhd/raw formats to 
# npy arrays, using SimpleITK

def get_images(path_to_images):    
    mri_files=get_lists_of_of_paths(path_to_images)
    image_paths=[]
    mri_images=[]
    for j in range(len(mri_files)):
        mid_file=get_lists_of_of_paths(mri_files[j])
        image_paths.append(get_lists_of_of_paths(mid_file[0]))
        for i in range(160):
            if i==0:    
                image = sitk.ReadImage(image_paths[j][0])
                image_array = (sitk.GetArrayFromImage(image)).reshape(384,384)
            else:  
                image = sitk.ReadImage(image_paths[j][i])
                image_staging= sitk.GetArrayFromImage(image).reshape(384,384)
                image_array=np.dstack((image_array,image_staging))
        mri_images.append(image_array)


    return mri_images




# We don the same for the masks, using SimpleITK for the segmentation
# masks and np.load for the Region_Of_Interest
# (roi) masks, all in ./masks/<sample_id>/ 

def get_masks(path_to_images): 

    masks_paths=get_lists_of_of_paths(path_to_images)

    roi_masks=[]
    segm_masks=[]
    for j in range(len(masks_paths)):
        mid_file=get_lists_of_of_paths(masks_paths[j])
        roi_masks.append(np.load(mid_file[3]))

        segm_mask = sitk.ReadImage(mid_file[1], sitk.sitkFloat32)
        segm_masks.append(sitk.GetArrayViewFromImage(segm_mask))

    return segm_masks, roi_masks
  
  
  


# we use this function for our sampling of voxels to create local 
# descriptors (feature vectors) datasets for our classification algorithmm 

def append_to_coordinates(array1, array2, starting_index):
    for i in range(starting_index,starting_index+100):
        array1.append(array2[i])




# This function computes the SIFT descriptor on the 2D axis, along each
# 2D axis combination in XYZ, using openCV, sums 3 2D feature vectors normalizes
# that to use it as the voxel 3D descriptor.
# It takes as input the 3D MRI scan numpy array, and the coordinate in each of 
# x, y, z axis and returns the local feature vector. To do that it passes along 
# the 2D coordinates each time as the keypoint for the SIFT algorithm.

def makeshiftSIFT(array3d, x_coordinate, y_coordinate, z_coordinate ):

    XYaxis=array3d[:,:,z_coordinate]
    XZaxis=array3d[:,y_coordinate,:]
    YZaxis=array3d[x_coordinate,:,:]

    gray = cv2.cvtColor(XYaxis, cv2.COLOR_BGR2GRAY)
    keypoint1= cv2.KeyPoint(x_coordinate, y_coordinate, 1)
    _, descriptor1 = sift.compute(gray, keypoint1)


    gray = cv2.cvtColor(XZaxis, cv2.COLOR_BGR2GRAY)
    keypoint2= cv2.KeyPoint(x_coordinate, z_coordinate, 1)
    _, descriptor2 = sift.compute(gray, keypoint2)

    gray = cv2.cvtColor(XZaxis, cv2.COLOR_BGR2GRAY)
    keypoint3= cv2.KeyPoint(y_coordinate, z_coordinate, 1)
    _, descriptor3 = sift.compute(gray, keypoint3)


    descriptor = descriptor1 + descriptor2 + descriptor3 
    descriptor = descriptor / 3

    return descriptor




#We use the makeshiftSIFT function to extract the local descriptors doing 
#a weighted sampling based on the labels mask, which we use to identify to 
#which class each voxel of the MRI scan belongs to. We focus the sampling on 
#the cartilage classes.
#It takes as input the MRI scan array and the segmentation mask array that 
#were created with the previous funcitons, and returns the full dataset of the
#local descriptors.

def compute_descriptors(image, segmetation_mask):

    FBlistofCoordinates=[]
    FAClistofCoordinates=[]
    TBlistofCoordinates=[]
    TAClistofCoordinates=[]

    for y in range(150,275,3):
        for x in range(70,250,3):
            for z in range(30,120,3):
                if segmetation_mask[x,y,z]==1:
                    FBlistofCoordinates.append([x,y,z])
                elif segmetation_mask[x,y,z]==2:
                    FAClistofCoordinates.append([x,y,z])
                elif segmetation_mask[x,y,z]==3:
                    TBlistofCoordinates.append([x,y,z])
                elif segmetation_mask[x,y,z]==4:
                    TAClistofCoordinates.append([x,y,z])

    ListOfCoordinates=[]
    for i in (0, 7000, 13000):
        append_to_coordinates(ListOfCoordinates,TBlistofCoordinates, i) 
    for i in (0, 120, 350, 600, 800 ,920):
        append_to_coordinates(ListOfCoordinates,TAClistofCoordinates, i)
    for i in (0, 120, 500, 900, 1300 ,1800):
        append_to_coordinates(ListOfCoordinates,FAClistofCoordinates, i)
    for i in (0, 10000, 20000):
        append_to_coordinates(ListOfCoordinates,FAClistofCoordinates, i)


    all_descriptors=[]
    for coordinate in ListOfCoordinates:
        all_descriptors.append(makeshiftSIFT(image, coordinate[0], coordinate[1], coordinate[2]))

    return all_descriptors, ListOfCoordinates


    

#The function below takes the list of descriptors as the training/testing 
#datasets and, after doing a Grid Search in the first run, we select the best 
#hypermarameters to find the W vector for each MRI scan. We do that by getting 
#the classes from the segmentation mask that has the info of which class
#each voxel belongs to.

def image_specific_SVM(coordinates, descriptors, array_with_labels):

    X=descriptors
    Y=[array_with_labels[coordinate[0], coordinate[1], coordinate[2]] for coordinate in coordinates]

    x_train=X[0:1400]
    x_test=X[1400:1800]
    y_train=Y[0:1400]
    y_test=Y[1400:1800]

    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['rbf']} 
    
    grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3, cv=5)
    
    # fitting the model for grid search
    grid.fit(x_train, y_train)

    classifier = grid.best_estimator_
    classifier.fit(x_train, y_train)
    W = classifier.coeff_

    return W
# weights=image_specific_SVM(ListOfCoordinates, all_descriptors, classes_array)



#This function is core to the implementation of our variation of the Bag Of Words
#algorithm. It implements KMeans to find the 50 'most frequently' used local descriptors
#across all scans and creates a (normalized) histogram with soft assignment to class centers 
#from local descriptors.

def descriptor_soft_assign(feature_vectors, no_of_words, descriptors, a=1):
    

    random_state=1
    kmeans_model=KMeans(n_clusters=no_of_words, verbose=False, init='random', random_state=random_state, n_init=3)
    kmeans_model.fit(descriptors)

    assignments=np.zeros(no_of_words)
    for feature_vector in feature_vectors:
        diffs=[]
        for cluster in kmeans_model.cluster_centers_:
            diff = norm(feature_vector-cluster)
            diffs.append(diff)
            sum_of_diffs=sum(diffs)
        for i in range(len(diffs)) in diffs:
            assignment= np.exp(-a*diffs[i])/np.exp(-a*sum_of_diffs)
            assignments[i]+=assignment
    
    assignment_histograms=assignments/sum(assignments)
    
    return assignment_histograms




# We do a double KMeans clustering in the coodinates of the ROI to find the cluster
# centers of the spatial information. That way, there is a number of subregions 
# defined by these cluster centers. We will use these subregions as a tool to 
# not lose the aforementioned spatial information when computing the global 
# descriptors.

def define_subregions(roi_mask):
    indices = np.where(roi_mask == 1)

    Coordinates = np.array(indices).T
    n_clusters = 100
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Coordinates)
    labels = kmeans.predict(Coordinates)

    staging_Clusters = kmeans.cluster_centers_

    kmeans_2 = KMeans(n_clusters=6)
    kmeans_2.fit(staging_Clusters)
    labels_2 = kmeans_2.predict(staging_Clusters)

    Final_Cluster_Centers = kmeans_2.cluster_centers_


    # Create a dictionary to store the coordinates for each of the Final_Cluster_Centers
    Final_Cluster_Dict = {}
    for i in range(len(Final_Cluster_Centers)):
        Final_Cluster_Dict[i] = []


    # Assign each Original_Coordinate to one of the Final_Cluster_Centers via their relation to the Hundred_Clusters
    for i, label in enumerate(labels):
        Hundred_Cluster_Label = label
        Final_Cluster_Label = labels_2[Hundred_Cluster_Label]
        Final_Cluster_Dict[Final_Cluster_Label].append(Coordinates[i])


    labeled_array = np.zeros_like(roi_mask)

    # Loop through each cluster center label and assign the corresponding points in the labeled array
    for label, points in Final_Cluster_Dict.items():
        for point in points:
            x, y, z = point
            if roi_mask[x, y, z] == 1:
                labeled_array[x, y, z] = label + 1  # Add 1 to the label to shift the range from 0-5 to 1-6


    return labeled_array, Final_Cluster_Dict


# We apply the process of our variation of the Bag Of Words using the soft
# assignment method to create the histogram/descriptor of each of the subregions.
# We concatenate the histograms to create the global descriptor.

def global_descriptor(image, all_descriptors, subregion1_indeces, subregion2_indeces, subregion3_indeces):

    subr1_descriptors=[]
  
    for index in subregion1_indeces:
        subr1_descriptor=makeshiftSIFT(image, index[0], index[1], index[2])
        subr1_descriptors.append(subr1_descriptor)

    subregion1_histogram= descriptor_soft_assign(subr1_descriptors, 50, all_descriptors)



    subr2_descriptors=[]
    for index in subregion2_indeces:
        subr2_descriptor=makeshiftSIFT(image, index[0], index[1], index[2])
        subr2_descriptors.append(subr2_descriptor)

    subregion2_histogram= descriptor_soft_assign(subr2_descriptors, 50, all_descriptors)

    
    
    subr3_descriptors=[]
    for index in subregion3_indeces:
        subr3_descriptor=makeshiftSIFT(image, index[0], index[1], index[2])
        subr3_descriptors.append(subr3_descriptor)

    subregion3_histogram= descriptor_soft_assign(subr3_descriptors, 50, all_descriptors)
    



    global_image_descriptor = np.concatenate([subregion1_histogram, subregion2_histogram, subregion3_histogram])

    return global_image_descriptor




#We create all our grobal descriptors as well as all our weight vectors to create 
#the Global Descriptor dataset and our Weight dataset, between which we will try to 
#find an intrinsic relationship in the next steps.

def get_atlas(images_array, roi_array, classes_array):

  
    subregion1_indeces=[]
    subregion2_indeces=[] 
    subregion3_indeces=[]

    Weights=[]
    Global_Descriptors=[]

    for i in range((len(images_array))):

        

        local_descriptors, sampled_coordinates = compute_descriptors(images_array[i], roi_array[i], classes_array[i])


        weight=image_specific_SVM(sampled_coordinates, local_descriptors, classes_array) 
        Weights.append(weight)
 
        global_descriptor=global_descriptor(images_array[i], local_descriptors, subregion1_indeces, subregion2_indeces, subregion3_indeces)
        Global_Descriptors.append(global_descriptor)

    return  Global_Descriptors, Weights