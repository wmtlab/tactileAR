import os
import sys
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import io

sys.path.append(os.path.abspath('.'))
from config import *
from tactileSensor import TactileSensor
from tactileReconstruction import TactileReconstruction
from controlPolicy import RandomControlPolicy, CurvePerceptionPolicy
from tools import calculationSSIM, calculationPSNR


# For environments without a display
import platform
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'


class SimulationTest():
    def __init__(self, simulation_test_config, 
                       tactile_sensor_config,
                       Control_config):
        """
        Initialize the simulation environment with given configurations.
        """
        self.env_config = simulation_test_config
        self.tactile_sensor_config = tactile_sensor_config
        self.sim_sensor = TactileSensor(sensor_config=self.tactile_sensor_config)

        self.state_shape = self.sim_sensor.state_resoultion
        self.hr_shape = self.sim_sensor.tactile_resolution
        self.lr_shape = (4, 4)
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        if self.env_config['is_outVideo']:
            self.images_lists = []
        
        
        self.tactileKF = TactileReconstruction(state_shape=self.state_shape,
                                                hr_shape=self.hr_shape,
                                                lr_shape=self.lr_shape,
                                                device=self.device)
        
        if self.env_config["is_real_dm"]:
            self.degradation_matrix = self.tactileKF.get_degradation_matrix(self.env_config["dm_file_path"])
        else:
            self.degradation_matrix = self.tactileKF.generateDegradationMatrix_Iso(gamma=self.env_config["dm_gamma"], HR_shape=self.hr_shape)
        
        self.random_policy = RandomControlPolicy(Control_config)
        self.active_policy = CurvePerceptionPolicy(Control_config,
                                                   state_shape=self.state_shape,
                                                   hr_shape=self.hr_shape,
                                                   lr_shape=self.lr_shape)
        
        self.active_policy.set_alpha(simulation_test_config['active_alpha'])
        
        self.init_env()
    
    def run(self, is_printInfo=True):
        MAX_EPOCH = self.env_config['max_epoch']
        epoch = 0
        run_flag = True
        action_x, action_y, action_theta = 0, 0, 0
        MSE_state_list, SSIM_state_list = [], []
        while epoch < MAX_EPOCH and run_flag:
            if is_printInfo:
                print("<--------------------->")
                print("epoch:", epoch)
                print("cur pos:", action_x, action_y, action_theta)
            MSE_state, SSIM_state, SSIM_hr, action_x, action_y, action_theta = self.simulation(action_x, action_y, action_theta, is_printInfo)
            MSE_state_list.append(MSE_state.cpu().numpy())
            SSIM_state_list.append(SSIM_state.cpu().numpy())
            epoch +=1
        
        if self.env_config['is_outVideo']:
            self.output_video()
        
        return np.array(MSE_state_list), np.array(SSIM_state_list)

    def printInfo(self,):
        print(f"[{time.localtime()}]: is_active_poily :{self.env_config['active_policy']}, alpha:{self.env_config['active_alpha']}")


    def init_env(self, ):
        self.mu_prior, self.sigma_prior = self.tactileKF.init_state_probability(mu=self.env_config["mu"],
                                                                                A=self.env_config["A"],
                                                                                r=self.env_config['r'])
        
        self.state_color, self.state_depth, _ = self.sim_sensor.get_sensor_scene_image(0, 0, 0)
        self.state_depth = (self.state_depth - self.state_depth.min()) / (self.state_depth.max() - self.state_depth.min())
        self.state_depth = torch.from_numpy(self.state_depth).to(self.device)
        
        if self.env_config["is_outVideo"]:
            self.setup_video_plot()
    
    
    def setup_video_plot(self):
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(231)
        self.ax2 = self.fig.add_subplot(232)
        self.ax3 = self.fig.add_subplot(233)
        self.ax4 = self.fig.add_subplot(234)
        self.ax5 = self.fig.add_subplot(235)
        self.ax6 = self.fig.add_subplot(236)
        policy_name = 'Active Policy' if self.env_config['active_policy'] else 'Random Policy'
        alpha_name = str(self.env_config['active_alpha']) if self.env_config['active_policy'] else ''
        self.fig.suptitle(f'{policy_name} + {alpha_name}')
    
    
    def output_video(self, FPS=1, path='./out.mp4'):
        if self.env_config['is_outVideo']:
            imageio.mimsave(path, self.images_lists, fps=FPS)
        else:
            print("is_outVideo == False")
    
    
    def simulation(self, action_x, action_y, action_theta, is_printInfo=True):
        """
        Run a single simulation iteration with the provided action parameters.
        """
        _, _, tactile_depth = self.sim_sensor.get_sensor_scene_image(action_x, action_y, action_theta)
        if tactile_depth.max() != tactile_depth.min():
            tactile_depth = (tactile_depth - tactile_depth.min()) / (tactile_depth.max() - tactile_depth.min())

        C_k = self.tactileKF.transform(np.array([action_x,action_y]), [action_theta])
        tactile_depth_est_flatten = C_k @ self.state_depth.reshape(-1, 1)
        tactile_depth_est = tactile_depth_est_flatten.reshape(tactile_depth.shape)
        LR_est_flatten = self.degradation_matrix @ tactile_depth_est_flatten
        LR_est_flatten = LR_est_flatten + torch.normal(0, self.env_config['LR_noise'], size=LR_est_flatten.shape, device=self.device) #* 添加噪声
        LR_est  = LR_est_flatten.reshape((4, 4))
        
        mu_update, sigma_update, _ , entropy_current = self.tactileKF.updataMeature(self.mu_prior, self.sigma_prior, C_k, self.degradation_matrix, LR_est_flatten)
        self.mu_prior, self.sigma_prior = mu_update, sigma_update     ## ? deepcopy

        if self.env_config['active_policy']:
            if self.device == 'cpu':
                mu_update_np, sigma_update_np = mu_update.numpy().reshape(self.state_shape), sigma_update.numpy().reshape(self.state_shape[0]*self.state_shape[1], self.state_shape[0]*self.state_shape[1])
            else:
                mu_update_np, sigma_update_np = mu_update.cpu().numpy().reshape(self.state_shape), sigma_update.cpu().numpy().reshape(self.state_shape[0]*self.state_shape[1], self.state_shape[0]*self.state_shape[1])
            pos_x_mm, pos_y_mm, pos_theta, D_map, D_mask = self.active_policy.get_decision_map(mu_update_np, sigma_update_np, pixelPerMm=0.5)
        else:
            D_map = np.zeros(shape=self.state_shape)
            pos_x_mm, pos_y_mm, pos_theta = self.random_policy.sampler()

        HR_est_flattern = C_k @ mu_update
        HR_est_img = HR_est_flattern.reshape(tactile_depth.shape)
        surface_est_img = mu_update.reshape(self.state_depth.shape)
        MSE_state = torch.sum((surface_est_img-self.state_depth)**2)
        SSIM_state = calculationSSIM(surface_est_img, self.state_depth)
        SSIM_hr = calculationSSIM(tactile_depth_est, HR_est_img)
        
        if is_printInfo:
            print("state MSE:{}, SSIM:{}, | HR SSIM:{}".format(MSE_state, SSIM_state, SSIM_hr))
        
        if self.env_config["is_outVideo"]:
            self.update_video_frames(action_x, action_y, action_theta, tactile_depth_est, HR_est_img, surface_est_img, D_map)

        return MSE_state, SSIM_state, SSIM_hr, pos_x_mm, pos_y_mm, pos_theta
    
    
    def update_video_frames(self, action_x, action_y, action_theta, tactile_depth_est, HR_est_img, surface_est_img, D_map):
        self.ax1.cla(), self.ax2.cla(), self.ax3.cla(), self.ax4.cla(), self.ax5.cla(), self.ax6.cla()

        self.state_color_ = np.flipud(self.state_color)
        self.ax1.imshow(self.state_color_)
        self.ax2.imshow(self.state_depth.cpu(), vmin=0, vmax=1)
        self.ax3.imshow(tactile_depth_est.cpu(), vmin=0, vmax=1)
        self.ax4.imshow(surface_est_img.cpu(), vmin=0, vmax=1)
        self.ax5.imshow(HR_est_img.cpu(), vmin=0, vmax=1)
        self.ax6.imshow(D_map)
        self.plot_sensor(self.ax1, 2 * action_x + self.state_shape[0] // 2, self.state_shape[0] // 2 - 2 * action_y, 40, action_theta)

        self.ax1.set_xlim([0, self.state_shape[0]])
        self.ax1.set_ylim([0, self.state_shape[1]])
        self.ax1.axis('off'), self.ax2.axis('off'), self.ax3.axis('off'), self.ax4.axis('off'), self.ax5.axis('off'), self.ax6.axis('off')
        self.ax1.set_title('Sensor Position'), self.ax2.set_title('State Ground Truth'), self.ax3.set_title('HR Ground Truth'), self.ax4.set_title('State Estimation'), self.ax5.set_title('HR Estimation'), self.ax6.set_title('Decision Map')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        self.images_lists.append(np.array(img))
        buf.close()
    
    
    def plot_sensor(self, ax, x, y, H, theta):
        theta_rad = np.deg2rad(theta)  
        half_diagonal = np.sqrt(2) * H / 2  

        corners = [(x + half_diagonal * np.cos(theta_rad + np.pi / 4 * i), 
                    y + half_diagonal * np.sin(theta_rad + np.pi / 4 * i)) for i in range(1, 9, 2)]
        
        for i in range(4):
            ax.plot([corners[i][0], corners[(i + 1) % 4][0]], 
                    [corners[i][1], corners[(i + 1) % 4][1]], color='black')
        
        dx = H * 0.6 * np.cos(theta_rad)
        dy = H * 0.6 * np.sin(theta_rad)
        ax_sensor = ax.arrow(x, y, dx, dy, length_includes_head=True, head_width=2, fc='red', ec='red')
        return ax_sensor


if __name__ == "__main__":
    args = get_args()
    update_config_from_args(args)
    sim_test = SimulationTest(simulation_test_config, tactile_sensor_config, Control_config)
    MSE_state_list, SSIM_state_list = sim_test.run()
    sim_test.output_video()
    # print(SSIM_state_list)

