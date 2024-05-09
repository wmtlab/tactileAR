import numpy as np
import torch


def gamma2Cov1(gamma):
    """
    Generate a covariance matrix using a single parameter, ensuring symmetry.
    Input gamma should be a torch tensor with `requires_grad = True`.
    """
    return torch.diag(torch.tensor([1, 1], dtype=gamma.dtype)) * gamma

def gamma2Cov3(gamma_x, gamma_y, gamma_z):
    """
    Ensure positive definiteness by constructing a lower triangular matrix 'L' and multiplying by its transpose.
    """
    L = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    L[0, 0] *= gamma_x
    L[1, 0] *= gamma_y
    L[1, 1] *= gamma_z
    return L @ L.T


def single_taxel_degradation_matrix(centre_x, centre_y, cov, HR_shape):
    """
    Generate a degradation matrix for a single taxel at given pixel coordinates in a high-resolution (HR) sensor grid.
    """
    assert len(HR_shape) == 2, "HR shape dim should be 2!"
    # X, Y    = torch.meshgrid(torch.linspace(0,HR_shape[0]-1,HR_shape[0]), torch.linspace(0,HR_shape[1]-1,HR_shape[1]), indexing="ij")
    X, Y    = torch.meshgrid(torch.linspace(0,HR_shape[0]-1,HR_shape[0]), torch.linspace(0,HR_shape[1]-1,HR_shape[1]))
    pos     = torch.dstack((X, Y))
    mean    = torch.tensor([centre_x, centre_y], dtype=pos.dtype)

    cov_inv = torch.linalg.inv(cov)
    fac = torch.einsum('...k,kl,...l->...', pos - mean, cov_inv, pos - mean)
    
    gs  = torch.exp(-fac / 2)
    # gs  = (gs - gs.min()) / (gs.max() - gs.min())     # Need Nomalization？
    gs  = gs/ gs.sum()
    return gs, gs.reshape(1, -1)


def generateDegradationMatrix(gamma_param, type, taxel_centre_list=None, HR_shape=(40, 40), device='cpu'):
    """
    Generate a degradation matrix for tactile sensors based on the type of covariance generation specified.
    Supports three types: 0, 1, and 2.
    """
    if type == 0 or type == 1:
        assert len(gamma_param) == 1, "For type 0 or 1, gamma_param should have length 1."
    elif type == 2:
        assert len(gamma_param) == 16, "For type 2, gamma_param should have length 16."
    else:
        raise NotImplementedError("Supported types are 0, 1, and 2.")
    
    H = None
    H_matrix = []
    for x_idx in range(4):
        for y_idx in range(4):
            centre_x, centre_y = taxel_centre_list[0][x_idx], taxel_centre_list[1][y_idx]
            
            if type == 0:
                cov = gamma2Cov1(gamma_param[0][0])
            elif type == 1:
                cov = gamma2Cov3(gamma_param[0][0], gamma_param[0][1], gamma_param[0][2])
            else:
                cov = gamma2Cov3(gamma_param[4*y_idx + x_idx][0], gamma_param[4*y_idx + x_idx][1], gamma_param[4*y_idx + x_idx][2])
            
            H_i_matrix, H_i_flatten = single_taxel_degradation_matrix(centre_x, centre_y, cov, HR_shape)
            
            H_matrix.append(H_i_matrix)
            H = torch.cat([H, H_i_flatten]) if H is not None else H_i_flatten
            
    return H_matrix, H.to(device)