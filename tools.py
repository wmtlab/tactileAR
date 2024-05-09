import torch
from scipy.spatial.transform import Rotation as R
import numpy as np


def calculationPSNR(pattern1, pattern2, maxValue, is_printInfo=False):
    """ 
    pattern1: (H, W)
    pattern2: (H, W)
    PSNR = 10 * log_10 (MAX**2 / MSE)
    """
    if is_printInfo:
        print("pattern1:{}, pattern2:{}".format(pattern1.shape, pattern2.shape))
        assert len(pattern1.shape) == len(pattern2),  "pattern1 and pattern2 should be same dimension!"
        assert len(pattern1.shape) == 2, "input should be two dimension!"
        assert pattern1.shape == pattern2.shape, 'pattern should be same shape!' 
    mse  = (pattern1 - pattern2)**2
    mse  = mse.sum() / (pattern1.shape[0]*pattern1.shape[1])
    PSNR = 10 * torch.log10(maxValue**2 / mse)
    return PSNR.data


def calculationSSIM(pattern1, pattern2, C1=0.01**2, C2=0.03**2, is_printInfo=False):
    """ 
    pattern1: (H, W)
    pattern2: (H, W)    
    """
    if is_printInfo:
        print("pattern1:{}, pattern2:{}".format(pattern1.shape, pattern2.shape))
        assert len(pattern1.shape) == len(pattern2),  "pattern1 and pattern2 should be same dimension!"
        assert len(pattern1.shape) == 2, "input should be two dimension!"
        assert pattern1.shape == pattern2.shape, 'pattern should be same shape!' 
    mu1, mu2 = pattern1.mean(), pattern2.mean()
    img1_sq, img2_sq, img12 = pattern1*pattern1, pattern2*pattern2, pattern1*pattern2
    # mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2     # torch
    mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2         # torch & numpy
    sigma1_sq, sigma2_sq, sigma12 = img1_sq.mean() - mu1_sq, img2_sq.mean() - mu2_sq, img12.mean() - mu1_mu2
    ssim = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim


def euler2matrix(angles=[0, 0, 0], translation=[0, 0, 0], xyz="xyz", degrees=True):
    r = R.from_euler(xyz, angles, degrees=degrees)
    pose = np.eye(4)
    pose[:3, 3] = translation
    pose[:3, :3] = r.as_matrix()
    return pose

