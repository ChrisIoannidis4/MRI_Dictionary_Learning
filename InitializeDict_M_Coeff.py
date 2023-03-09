from get_parameters import * 
from sklearn.decomposition import PCA



def Initialize_Dictionaries(Global_Descriptors, Weights, Dictionary_Size):
    
    k=Dictionary_Size

    descr_kmeans=KMeans(n_clusters=k, random_state=0)
    descr_kmeans.fit(Global_Descriptors)
    descriptors_dictionary = descr_kmeans.cluster_centers_      

    weights_kmeans=KMeans(n_clusters=k, random_state=0)
    weights_kmeans.fit(Weights)
    weights_dictionary = weights_kmeans.cluster_centers_   


    return descriptors_dictionary, weights_dictionary


def Initialize_Coefficients(Global_Descriptors, Weights, Dictionary_Size, weights_dictionary):
    #To initiate the Λx and Λy to sparsely represent any given instance of X or W from the columns of
    # the Dx and Dw, we will
    #Note: note: every column should have a sum of one
    #We measure sparsity with L0 norm
    # Coeff_X = np.ones((1, len(Global_Descriptors))) * (1/Dictionary_Size)


    Coeff_W = np.linalg.lstsq(weights_dictionary, Weights, rcond=None)
    Coeff_W /= np.sum(Coeff_W, axis=0)

    return Coeff_W  #, Coeff_X


def Initilize_Mapping(descriptors_dictionary, weights_dictionary, Dictionary_Size):
    
    pca = PCA(n_components=Dictionary_Size)
    pca.fit(descriptors_dictionary)
    source_components = pca.components_
    pca.fit(weights_dictionary)
    target_components = pca.components_
    mapping_function = target_components.T @ source_components

    return mapping_function                                                             


