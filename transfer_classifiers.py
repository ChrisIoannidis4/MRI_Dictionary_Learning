from optimization_problems import *


def transfer_classifiers(Dx, Dw, M, image, lamda, all_descriptors):

    image_descriptor = global_descriptor(image, all_descriptors)

    staging1 = np.matmul(Dx.transpose, Dx) + lamda * np.identity(len(Dx))
    staging2 = np.matmul(Dx.transpose, image_descriptor)

    coeff_x = np.matmul(inv(staging1), staging2)
    coeff_w = np.matmul(M, coeff_x)

    weights = np.matmul(Dw, coeff_w)

    return image_descriptor, weights

