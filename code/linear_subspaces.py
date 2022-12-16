# SVD
# factorisation
# reduce_factorisation_dimensions (see if can merge with above function)
# projectors + projections + compare_projections
# get basis with x% representativity
import numpy as np

def SVD(data):
    
    # D=[d_1 d_2 ... d_{12}]= U * Sigma * V^T
    
    # optimal decomposition given the rank for the subspace. U is the base for the columns and v^T for the rows 
    # U(:,1) --> best direction se se quiser rank = 1 subspace 
    
    # sigma are the singular values of the D, giving the importance of each base 
    
    return np.linalg.svd(data)
    

def factorisation(data, rank):
    
    D = SVD(data)
    
    basis = D[0][:, :rank]
    sigma = D[1]
    vT = D[2][:rank, :]
    
    return basis, sigma, vT


def get_basis_projector(basis):
    return basis @ basis.T # PI = U @ U^T


def get_null_projector(basis):
    return (np.eye(basis.shape[0]) - get_basis_projector(basis)) # I - PI


def get_basis_projectors(basis):
    return get_basis_projector(basis), get_null_projector(basis)


def project(projector, projected):
    return projector @ projected


def compare_projections(basis, features, keep_first_n, ord=2):
    projector, null_projector = get_basis_projectors(basis)
    
    proj = project(projector, features)
    orth_proj = project(null_projector, features)
    
    print("All original points are close to proj + orth_proj: {0}".format(np.isclose(features, proj+orth_proj).all()))

    norm_proj = np.linalg.norm(proj, axis=0, ord=ord)/np.linalg.norm(features, axis=0, ord=ord)
    norm_orth_proj = np.linalg.norm(orth_proj, axis=0, ord=ord)/np.linalg.norm(features, axis=0, ord=ord)
    
    larger_projnorm = sorted(norm_proj, reverse=True)[0:keep_first_n ]
    larger_orthprojnorm = sorted(norm_orth_proj, reverse=True)[0:keep_first_n ]

    idx_larger_projnorm = np.argsort(-norm_proj)[0:keep_first_n]
    idx_larger_orthprojnorm = np.argsort(-norm_orth_proj)[0:keep_first_n]

    return (norm_proj, norm_orth_proj), (idx_larger_projnorm, idx_larger_orthprojnorm)


def reduce_factorisation_dimensions(factorisation, rank):
    print("Reducing original {0}-dim factorisation to {1} dimensions...".format(factorisation[1].shape[0], rank))
    return factorisation[0][:,:rank], factorisation[1][:rank], factorisation[2][:rank, :]

# def get_coefficients()


def get_representativity(sigma):
    return np.cumsum(sigma)/np.sum(sigma) * 100


def cumulative_representativity(sigma, index):
    representativity = get_representativity(sigma)
    return representativity[index]


def minimum_representativity(representativity, threshold):
    return np.argmax(representativity > threshold)


def get_singular_value(sigma, index):
    return sigma[index]


def get_rank(sigma, threshold):
    representativity = get_representativity(sigma)
    min_repr = minimum_representativity(representativity, threshold)    
    cumulative_repr = cumulative_representativity(sigma, min_repr)
    return min_repr, cumulative_repr
