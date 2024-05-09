import torch
import numpy as np
import time
import os, sys

sys.path.append('.')
from config import *
from degradation import generateDegradationMatrix


class TactileReconstruction:
    """
    Dimensionality of states (M), High Resolution (HR) space (N), and Low Resolution (LR) space (L).
    M = state_shape[0] * state_shape[1]
    N = hr_shape[0]    * hr_shape[1]
    L = lr_shape[0]    * lr_shape[1]
    """
    def __init__(self, state_shape=(60, 60), hr_shape=(40, 40), lr_shape=(4, 4), device='cpu'):
        self.state_shape, self.hr_shape, self.lr_shape = state_shape, hr_shape, lr_shape
        
        # Initialize position tensors for contact states in HR and LR
        self.contact_state_positions = self.positions(state_shape).to(device)  # [2, M]
        self.contact_HR_positions = self.positions(hr_shape).to(device)        # [2, N]
        self.contact_LR_positions = self.positions(lr_shape).to(device)        # [2, L]
        
        # Precompute squared position matrices for efficiency
        self.contact_state_positions2 = torch.sum(self.contact_state_positions ** 2, axis=0)  # [M]
        self.contact_HR_positions2 = torch.sum(self.contact_HR_positions ** 2, axis=0)        # [N]
        self.contact_LR_positions2 = torch.sum(self.contact_LR_positions ** 2, axis=0)        # [L]
        
        self.mmPerPixel = 2
        self.measure_cnt = 0
        self.device = device
        
        # Initialize gradient computation matrices for tactile data
        M = hr_shape[0]
        G_x = np.zeros((M * M, M * M))
        G_y = np.zeros((M * M, M * M))
        
        """ 
        [ 0 0 0]       [ 0 -1 0]
        [-1 0 1]  and  [ 0  0 0]
        [ 0 0 0]       [ 0  1 0]
        """
        # for x in range(M):
        #     for y in range(1, M-1):
        #         G_y[x*M+y, x*M+y-1] =  1
        #         G_y[x*M+y, x*M+y+1] = -1
        #         G_x[y*M+x, y*M+x  ] = -1
        #         G_x[y*M+x, y*M+x+M] =  1 

        """ 
        [-1 0 1]       [-1 -2 1]
        [-2 0 2]  and  [ 0  0 0]
        [-1 0 1]       [ 1  2 1]
        """
        for x in range(1, M-1):
            for y in range(1, M-1):
                G_y[x*M+y, (x-1)*M+y-1] =  1/5
                G_y[x*M+y, (x-1)*M+y+1] = -1/5
                G_y[x*M+y, x    *M+y-1] =  2/5
                G_y[x*M+y, x    *M+y+1] = -2/5
                G_y[x*M+y, (x+1)*M+y-1] =  1/5
                G_y[x*M+y, (x+1)*M+y+1] = -1/5
                
                G_x[y*M+x, y*M+x-1]   = -1/5
                G_x[y*M+x, y*M+x  ]   = -2/5
                G_x[y*M+x, y*M+x+1]   = -1/5
                G_x[y*M+x, y*M+M+x-1] =  1/5
                G_x[y*M+x, y*M+M+x  ] =  2/5
                G_x[y*M+x, y*M+M+x+1] =  1/5
                 
        self.G_x = torch.from_numpy(G_x).type(torch.float32).to(device)
        self.G_y = torch.from_numpy(G_y).type(torch.float32).to(device)
    

    def positions(self, shape, mode='mid'):
        """
        Returns the 2 x N coordinate tensor. Mode can be 'mid' for center-origin coordinates or any other string for top-left-origin.
        """
        p = torch.empty((2, *shape), dtype=torch.float32)
        # p[1], p[0] = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing='ij')
        p[1], p[0] = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]))
        p = p.reshape(2, -1)
        if mode == 'mid':
            p -= torch.tensor([dim // 2 for dim in shape]).reshape(2, 1)
        return p

    
    def init_state_probability(self, mu=0.5, A=0.1, r=20):
        """
        Initialize contact surface state probabilities with mean `mu` and covariance shaped by `A` and `r`.
        """
        dist = 2 * self.contact_state_positions.T @ self.contact_state_positions - self.contact_state_positions2.reshape(1, -1) - self.contact_state_positions2.reshape(-1, 1)
        mu_0 = torch.full((1, self.state_shape[0] * self.state_shape[1]), fill_value=mu)
        sigma_0 = A * torch.exp((1 / r) * dist)
        return mu_0.to(self.device), sigma_0.to(self.device)
    
    
    def transform(self, s, theta, gamma=1e0):
        """
        Transform the HR position data according to translation `s` and rotation `theta`. `gamma` controls the transformation sharpness.
        """
        R = self.get_rotation_matrix(theta)
        s = s * self.mmPerPixel
        u = R @ self.contact_HR_positions + torch.from_numpy(s).reshape(2, 1).to(self.device)
        u = u.type(torch.float32)
        dist_sq = 2 * u.T @ self.contact_state_positions        # [N, 2] x [2, M] = [N, M]
        u2      = torch.sum(u**2, axis = 0).reshape(-1, 1)         # [N, 1]
        dist_sq = (dist_sq - self.contact_state_positions2.reshape(1, -1) - u2) # [N, M]
        C       = torch.exp(dist_sq / gamma**2 )
        scale   = torch.sum(C, axis=1).reshape(-1, 1)
        scale[scale==0] = 1     
        C       = C / scale
        return C
    
    
    def get_rotation_matrix(self, theta):
        """
        Returns the 2x2 rotation matrix for a given angle `theta` in degrees.
        """
        theta_rad = torch.deg2rad(torch.tensor(theta))
        R = torch.empty((2, 2), dtype=torch.float32, device=self.device)
        R[0, 0] =  torch.cos(theta_rad)
        R[0, 1] =  torch.sin(theta_rad)
        R[1, 0] = -torch.sin(theta_rad)
        R[1, 1] =  torch.cos(theta_rad)
        return R
    
    def get_degradation_matrix(self, file_path):
        """
        Load the degradation matrix from a specified file path and convert it to a PyTorch tensor.
        """
        degradation_matrix = np.load(file_path)
        self.degradation_matrix = torch.from_numpy(degradation_matrix).to(self.device)
        return self.degradation_matrix
    
    def generateDegradationMatrix_Iso(self, gamma, HR_shape):
        """
        Generate an isotropic degradation matrix for given 'gamma' and HR (High Resolution) shape.
        Only supports HR shapes of (40, 40) or (20, 20).
        """
        assert len(HR_shape) == 2, "HR_shape should be two-dimensional"
        gamma_tensor = torch.tensor([[gamma]])
        if HR_shape == (40, 40):
            _, H = generateDegradationMatrix(gamma_tensor, 0, taxel_centre_list_x10, HR_shape, device=self.device)
        elif HR_shape == (20, 20):
            _, H = generateDegradationMatrix(gamma_tensor, 0, taxel_centre_list_x5, HR_shape, device=self.device)
        else:
            raise NotImplementedError("Supported HR shapes are (20, 20) and (40, 40)")
        return H.to(self.device)
    
    def updataMeature(self, mu_before, sigma_before, C_k, H, lr_measure, axis='z', measure_noise=0.01, dtype=torch.float32):
        """
        Update state information (`mu_before` and `sigma_before`) based on the current measurement (`lr_measure`).
        """
        start_time = time.time()
        lr_measure = lr_measure.reshape(-1, 1)
        mu_prior = mu_before.reshape(-1, 1)
        sigma_prior = sigma_before

        # postier update
        if axis == 'x':
            HC_k = H @ self.G_x @ C_k                                              # [L, M]
        elif axis == 'y':
            HC_k = H @ self.G_y @ C_k
        elif axis == 'z':
            HC_k = H @ C_k
        else:
            raise ValueError("param axis should be \'x\', \'y\' OR \'z\'! ")
        
        
        # Calculate posterior update
        noise_matrix = torch.eye(lr_measure.shape[0], dtype=lr_measure.dtype, device=self.device) * measure_noise
        temp_matrix = HC_k @ sigma_prior @ HC_k.T + noise_matrix
        K_k = sigma_prior @ HC_k.T @ torch.linalg.inv(temp_matrix)
        mu_posterior = mu_prior + K_k @ (lr_measure - HC_k @ mu_prior)
        sigma_posterior = sigma_prior - K_k @ HC_k @ sigma_prior
        
        entropy_current = self.calculateEntropy(sigma_posterior)
        # print(f'[{self.measure_cnt}th measure] Update duration: [{time.time() - start_time:.2f}s], Current entropy: {entropy_current}')
        return mu_posterior, sigma_posterior, K_k, entropy_current


    def updataMeature_xyz(self, mu_before, sigma_before, C_k, 
                          Hx=None, Hy=None, Hz=None, 
                          lr_measure_x=None, lr_measure_y=None, lr_measure_z=None, update_mode=1,
                          measure_noise_xy=0.1, measure_noise_z=0.05, dtype=torch.float32):
        t1 = time.time()
        lr_measure_x  = lr_measure_x.reshape(-1, 1)
        lr_measure_y  = lr_measure_y.reshape(-1, 1)
        lr_measure_z  = lr_measure_z.reshape(-1, 1)
        mu_prior    = mu_before.reshape(-1, 1)                         # [M, 1]
        sigma_prior = sigma_before
        
        if update_mode==0:
            # TODO: Parallel update

            # D_x = Hx @ self.G_x @ C_k
            # D_y = Hy @ self.G_y @ C_k
            # D_z = Hz @ C_k

            # Q_x_inv = torch.linalg.inv(torch.eye(lr_measure_x.shape[0]) * measure_noise_xy)
            # Q_y_inv = torch.linalg.inv(torch.eye(lr_measure_y.shape[0]) * measure_noise_xy)
            # Q_z_inv = torch.linalg.inv(torch.eye(lr_measure_z.shape[0]) * measure_noise_z)

            # sigma_posterior_inv = torch.linalg.pinv(sigma_prior) + D_x.T @ Q_x_inv @ D_x + D_y.T @ Q_y_inv @ D_y + D_z.T @ Q_z_inv @ D_z
            # sigma_posterior = torch.linalg.pinv(sigma_posterior_inv)
            
            # K_t_x = sigma_posterior @ D_x.T @ Q_x_inv
            # K_t_y = sigma_posterior @ D_y.T @ Q_y_inv
            # K_t_z = sigma_posterior @ D_z.T @ Q_z_inv
            
            # mu_posterior = mu_prior + K_t_x @ (lr_measure_x - D_x @ mu_prior) + K_t_y @ (lr_measure_y - D_y @ mu_prior) + K_t_z @ (lr_measure_z - D_z @ mu_prior)
            # mu_posterior_xyz, sigma_posterior_xyz = mu_posterior, sigma_posterior
            
            raise NotImplementedError("Not support parallel update!")

        else:    
            # postier update | X data
            Hx_Gx_Ck = Hx @ self.G_x @ C_k                                                  # [L, M]
            temp_matrix_x     = Hx_Gx_Ck @ sigma_prior @ Hx_Gx_Ck.T + torch.eye(lr_measure_x.shape[0], dtype=torch.float32, device=self.device) * measure_noise_xy
            K_k_x             = sigma_prior @ Hx_Gx_Ck.T @ torch.linalg.inv(temp_matrix_x)     # [M, L]
            mu_posterior_x    = mu_prior + K_k_x @ (lr_measure_x - Hx_Gx_Ck @ mu_prior)
            sigma_posterior_x = sigma_prior - K_k_x @ Hx_Gx_Ck @ sigma_prior
            del Hx_Gx_Ck, temp_matrix_x, K_k_x
            
            # postier update | Y data
            Hy_Gy_Ck = Hy @ self.G_y @ C_k                                                  # [L, M]
            temp_matrix_y     = Hy_Gy_Ck @ sigma_posterior_x @ Hy_Gy_Ck.T + torch.eye(lr_measure_y.shape[0], dtype=torch.float32, device=self.device) * measure_noise_xy
            K_k_y             = sigma_posterior_x @ Hy_Gy_Ck.T @ torch.linalg.inv(temp_matrix_y)     # [M, L]
            mu_posterior_y    = mu_posterior_x + K_k_y @ (lr_measure_y - Hy_Gy_Ck @ mu_posterior_x)
            sigma_posterior_y = sigma_posterior_x - K_k_y @ Hy_Gy_Ck @ sigma_posterior_x
            del Hy_Gy_Ck, temp_matrix_y, K_k_y
            
            # postier update | Z data
            Hz_Ck = Hz @ C_k                                                                   # [L, M]
            temp_matrix_z     = Hz_Ck @ sigma_posterior_y @ Hz_Ck.T + torch.eye(lr_measure_z.shape[0], dtype=torch.float32, device=self.device) * measure_noise_z
            K_k_z             = sigma_posterior_y @ Hz_Ck.T @ torch.linalg.inv(temp_matrix_z)     # [M, L]
            mu_posterior_z    = mu_posterior_y + K_k_z @ (lr_measure_z - Hz_Ck @ mu_posterior_y)
            sigma_posterior_z = sigma_posterior_y - K_k_z @ Hz_Ck @ sigma_posterior_y
            del Hz_Ck, temp_matrix_z, K_k_z
            
            mu_posterior_xyz    = mu_posterior_z
            sigma_posterior_xyz = sigma_posterior_z

            del mu_posterior_x, mu_posterior_y, mu_posterior_z
            del sigma_posterior_x, sigma_posterior_y, sigma_posterior_z

        print('[{}th measure] cost time:[{}/s]'.format(self.measure_cnt, time.time()-t1))
        self.measure_cnt += 1
        return mu_posterior_xyz, sigma_posterior_xyz, None, None


    def calculateEntropy(self, sigma_matrix):
        """ 
        calculate gaussian distribution shannon entropy:
        https://stackoverflow.com/questions/29278848/python-determinant-of-a-large-matrix
        """
        return torch.linalg.slogdet(sigma_matrix)[1]
        
        
if __name__ == "__main__":
    pass
    
    