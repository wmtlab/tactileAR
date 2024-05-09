import os, sys
import numpy as np
import torch
import random
import cv2
from scipy import signal 
from scipy import ndimage


class RandomControlPolicy:
    """
    Random Control Policy: Generates random position information within specified constraints.
    """
    def __init__(self, Control_config):
        self.x_min, self.x_max = Control_config['x_min'], Control_config['x_max']
        self.y_min, self.y_max = Control_config['y_min'], Control_config['y_max']
        self.theta_min, self.theta_max = Control_config['theta_min'], Control_config['theta_max']
        self.xy_stepSize, self.theta_stepSize = Control_config['xy_stepSize'], Control_config['theta_stepSize']
        
        self.x_sampleCnt = int((self.x_max - self.x_min) / self.xy_stepSize)
        self.y_sampleCnt = int((self.y_max - self.y_min) / self.xy_stepSize)
        self.theta_sampleCnt = int((self.theta_max - self.theta_min) / self.theta_stepSize)
       
        self.x_space = np.linspace(self.x_min, self.x_max, self.x_sampleCnt+1)
        self.y_space = np.linspace(self.y_min, self.y_max, self.y_sampleCnt+1)
        self.theta_space = np.linspace(self.theta_min, self.theta_max, self.theta_sampleCnt+1)

    def sampler(self):
        x_sample = random.choice(self.x_space)
        y_sample = random.choice(self.y_space)
        theta_sample = random.choice(self.theta_space)
        return x_sample, y_sample, theta_sample



class CurvePerceptionPolicy:
    """
    Control strategy based on contour perception.
    """
    def __init__(self, Control_config, alpha=1.0, state_shape=(60, 60), hr_shape=(40, 40), lr_shape=(4, 4)):
        self.alpha = alpha
        self.state_shape = state_shape
        self.hr_shape = hr_shape
        self.lr_shape = lr_shape
        self.tapping_cnt = 0
        self.setup_control_config(Control_config)

        # Sobel operators for gradient calculation
        self.sobel_x3 = {'x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 5, 
                         'y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 5}
        self.sobel_x5 = {'x': np.array([[1, 2, 0, -2, -1], [4, 8, 0, -8, -4], [6, 12, 0, -12, -6], [4, 8, 0, -8, -4], [1, 2, 0, -2, -1]]), 
                         'y': np.array([[1, 4, 6, 4, 1], [2, 8, 12, 8, 2], [0, 0, 0, 0, 0], [-2, -8, -12, -8, -2], [-1, -4, -6, -4, -1]])}

        self.alpha_decay = np.exp(-self.alpha * np.arange(50))

    def setup_control_config(self, config):
        self.x_min, self.x_max = config['x_min'], config['x_max']
        self.y_min, self.y_max = config['y_min'], config['y_max']
        self.theta_min, self.theta_max = -90, 90
        self.xy_stepSize, self.theta_stepSize = config['xy_stepSize'], 10

        self.x_space = np.linspace(self.x_min, self.x_max, int((self.x_max - self.x_min) / self.xy_stepSize) + 1)
        self.y_space = np.linspace(self.y_min, self.y_max, int((self.y_max - self.y_min) / self.xy_stepSize) + 1)
        self.theta_space = np.linspace(self.theta_min, self.theta_max, int((self.theta_max - self.theta_min) / self.theta_stepSize) + 1)

    def get_gradient(self, image, mode='same'):
        """ Calculate the gradient in both x and y directions using predefined Sobel operators. """
        gx = signal.convolve2d(image, self.sobel_x3['x'], mode)
        gy = signal.convolve2d(image, self.sobel_x3['y'], mode)
        g_magnitude = np.sqrt(gx ** 2 + gy ** 2)
        return gx, gy, g_magnitude

    def get_decision_map(self, cur_mu, cur_sigma, pixelPerMm):
        """
        Generate a decision map based on current state estimation and uncertainty.
        Converts tensor data into numpy for processing.
        """
        cur_mu_img = cur_mu.reshape(self.state_shape)
        cur_mu_gx, cur_mu_gy, cur_mu_gxy = self.get_gradient(cur_mu_img)

        # Normalize uncertainty
        cur_U = np.diag(cur_sigma).reshape(self.state_shape) * 10  # Scaling factor for visibility
        cur_U = (cur_U - cur_U.min()) / (cur_U.max() - cur_U.min())

        # Normalize and apply alpha decay
        cur_mu_gxy = (cur_mu_gxy - 0) / np.sqrt(2)
        cur_mu_gxy = (1 - self.alpha_decay[self.tapping_cnt]) * cur_mu_gxy + self.alpha_decay[self.tapping_cnt]

        # Compute decision map
        D_map = cur_mu_gxy * cur_U

        # Compute optimal position and orientation
        D_map_max_pos, D_map_max_theta, mask = self.find_max_sum_region(D_map, perception_size=(30, 30))

        pos_x_pixel, pos_y_pixel = D_map_max_pos
        pos_x_mm = (pos_x_pixel - self.state_shape[0]//2) * pixelPerMm
        pos_y_mm = (pos_y_pixel - self.state_shape[1]//2) * pixelPerMm
        pos_theta = D_map_max_theta

        self.tapping_cnt += 1

        return pos_y_mm, pos_x_mm, pos_theta, D_map, mask


    def find_max_sum_region(self, D_map, perception_size, angle_perception_scale=1.5):
        """
        Determines the optimal region for sensor placement.
        Step 1: Identify the center point of the region defined by perception_size, ignoring orientation.
        Step 2: Rotate the sensor within an area scaled by angle_perception_scale around the center point to find the optimal orientation.
        Outputs the pixel coordinates.
        """
        D_map_shape_x, D_map_shape_y = D_map.shape
        perception_shape_x, perception_shape_y = perception_size
        angle_perception_shape_x = int(angle_perception_scale * perception_shape_x)
        angle_perception_shape_y = int(angle_perception_scale * perception_shape_y)

        max_sum = 0
        best_center_x = best_center_y = 0
        best_theta = 0

        # Step 1: Identify the [x, y] center point while theta is fixed
        for i in range(D_map_shape_x - perception_shape_x + 1):
            for j in range(D_map_shape_y - perception_shape_y + 1):
                current_sum = np.sum(D_map[i:i + perception_shape_x, j:j + perception_shape_y])
                if current_sum > max_sum:
                    max_sum = current_sum
                    best_center_x = i + perception_shape_x // 2
                    best_center_y = j + perception_shape_y // 2

        # Prepare the region for rotation
        clip_size_x = int(1.42 * angle_perception_shape_x)
        clip_size_y = int(1.42 * angle_perception_shape_y)
        padded_D_map = np.pad(D_map, ((clip_size_y // 2,), (clip_size_x // 2,)), 'constant')

        # Ensure no out-of-bounds indexing occurs
        crop_center_x = clip_size_x // 2 + best_center_x
        crop_center_y = clip_size_y // 2 + best_center_y
        clip_D_map = padded_D_map[
            crop_center_x - clip_size_x // 2: crop_center_x + clip_size_x // 2,
            crop_center_y - clip_size_y // 2: crop_center_y + clip_size_y // 2
        ]

        # Step 2: Determine the optimal theta within the specified angular perception area
        max_sum = 0
        for theta in range(-90, 91, 10):
            rotated_map = ndimage.rotate(clip_D_map, angle=theta, reshape=False)
            angle_D_map = rotated_map[
                (clip_size_x - angle_perception_shape_x) // 2: (clip_size_x + angle_perception_shape_x) // 2,
                (clip_size_y - angle_perception_shape_y) // 2: (clip_size_y + angle_perception_shape_y) // 2
            ]
            sum_rotated = angle_D_map.sum()
            if sum_rotated > max_sum:
                max_sum = sum_rotated
                best_theta = theta

        return [best_center_x, best_center_y], best_theta, clip_D_map


    def set_alpha(self, alpha_value):
        """ Adjust alpha and recalculate the exponential decay used for blending gradient magnitudes. """
        self.alpha = alpha_value
        self.alpha_decay = np.exp(-self.alpha * np.arange(50))   # MAX_EPOCH < 50

        
if __name__ == "__main__":
    sys.path.append(os.path.abspath('.'))
    from config import *
    
    cp = CurvePerceptionPolicy(Control_config)
    
    
    
    