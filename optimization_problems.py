from get_parameters import * 
from InitializeDict_M_Coeff import * 
from numpy.linalg import inv, norm
import cvxpy as cp

#UPDATE Λx

def update_coeff_X(Weights, Global_Descriptors, Descriptors_Dict, Weights_Dict, Mapping_Func, Coeff_W, gamma, lamda):
    staging1 = np.matmul(Weights_Dict.transpose ,Weights_Dict) + gamma * np.matmul(Mapping_Func.transpose ,Mapping_Func) + lamda * np.identity(len(Weights_Dict.transpose))
    staging2 =  np.matmul(Descriptors_Dict.transpose ,Global_Descriptors) + gamma * np.matmul(Mapping_Func.transpose, Coeff_W)

    Coeff_X = np.matmul(inv(staging1),staging2)

    return Coeff_X



#UPDATE Λw





#UPDATE Dx, Dw


def update_dictionaries(Global_Descriptors, Descriptors_Dict, coeff_X, Weights, Weights_Dict, coeff_W):

    Dx = Descriptors_Dict
    Dw = Weights_Dict
    X = Global_Descriptors
    W = Weights 
    k = len(Dx)

    Dx = cp.Variable((k, k))
    Dw= cp.Variable((k, k))

    objective = cp.Minimize(cp.norm(X - np.matmul(Dx, coeff_X), "fro") + cp.norm(W - np.matmul(Dw, coeff_W), "fro"))

    constraints = [cp.abs(Dx) <= 1, cp.abs(Dw) <= 1]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    Dx = Dx.value
    Dw = Dw.value

    return Dx, Dw


#UPDATE M

def update_mapping(Coeff_X, Coeff_W, lamda, gamma):
    staging_1=np.matmul(Coeff_W, Coeff_X.transpose)
    staging_2=np.matmul(Coeff_X,Coeff_X.transpose) + (lamda / gamma) * np.identity(len(Coeff_X))

    Mapping_Function = staging_1 * inv(staging_2)

    return Mapping_Function





def optimization_problem(Weights, Global_Descriptors, Descriptors_Dict, Weights_Dict, Mapping_Func, Coeff_W, gamma, lamda):
    
    Coeff_X = update_coeff_X(Weights, Global_Descriptors, Descriptors_Dict, Weights_Dict, Mapping_Func, Coeff_W, gamma, lamda)
    Coeff_W = update_coeff_W(Weights, Global_Descriptors, Descriptors_Dict, Weights_Dict, Mapping_Func, Coeff_W, gamma, lamda)
    Descriptors_Dict = update_dictionaries()[0]
    Weights_Dict = update_dictionaries()[1]
    Mapping_Function = update_mapping(Coeff_X, Coeff_W, lamda, gamma)



    max_iterations=100
    i=0

    while True:

        Coeff_X_old = Coeff_X
        Coeff_X = update_coeff_X(Weights, Global_Descriptors, Descriptors_Dict, Weights_Dict, Mapping_Func, Coeff_W, gamma, lamda)

        Coeff_W_old = Coeff_W
        Coeff_W = update_coeff_W(Weights, Global_Descriptors, Descriptors_Dict, Weights_Dict, Mapping_Func, Coeff_W, gamma, lamda)

        Descriptors_Dict_old = Descriptors_Dict
        Weights_Dict_old = Weights_Dict
        Descriptors_Dict = update_dictionaries()[0]
        Weights_Dict = update_dictionaries()[1]

        Mapping_Function_old = Mapping_Function
        Mapping_Function = update_mapping(Coeff_X, Coeff_W, lamda, gamma)

        Lxdiff= norm(Coeff_X - Coeff_X_old)
        Lwdiff= norm(Coeff_W - Coeff_W_old)
        Dxdiff= norm(Weights_Dict - Weights_Dict_old)
        Dwdiff= norm(Descriptors_Dict - Descriptors_Dict_old)
        Mdiff = norm(Mapping_Function - Mapping_Function_old)
        i += 1

        if (i > max_iterations) or ((Lxdiff<=0.01*norm(Coeff_X_old)) & (Lwdiff<=0.01*norm(Coeff_W_old))) \
                                & (Dxdiff<=0.01*norm(Descriptors_Dict_old)) & (Dwdiff<=0.01*norm(Descriptors_Dict_old)) & (Mdiff<=0.01*norm(Mapping_Function_old)) : 
            break
                                           
        


    return Descriptors_Dict, Weights_Dict, Mapping_Func, Coeff_X, Coeff_W
