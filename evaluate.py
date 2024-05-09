import os,sys
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np

sys.path.append(os.path.abspath('.'))
from config import tactile_sensor_config, Control_config, simulation_test_config, root_path
from test_sim import SimulationTest


def plot_results(data):
    alphas = ['Random', 'Alpha=0.0', 'Alpha=0.5', 'Alpha=1.0']
    MSE_keys = ['random_MSE', 'active_0_MSE', 'active_05_MSE', 'active_1_MSE']
    SSIM_keys = ['random_SSIM', 'active_0_SSIM', 'active_05_SSIM', 'active_1_SSIM']
    
    MSE_means = [np.mean(data[key], axis=0) for key in MSE_keys]
    SSIM_means = [np.mean(data[key], axis=0) for key in SSIM_keys]
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))  
    for i, means in enumerate(MSE_means):
        axs[0].plot(means, label=alphas[i])
    axs[0].set_title('Mean Squared Error (MSE) Across Different Policies')
    axs[0].set_ylabel('MSE')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()

    for i, means in enumerate(SSIM_means):
        axs[1].plot(means, label=alphas[i])
    axs[1].set_title('Structural Similarity Index (SSIM) Across Different Policies')
    axs[1].set_ylabel('SSIM')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('results.png')



def main():
    res_data = {}
    
    tactile_sensor_config["contact_plate_path"] = os.path.join(root_path, 'contact_surface/demo.stl')
    simulation_test_config['is_outVideo'] = False
    simulation_test_config['max_epoch'] = 30
    simulation_test_config['LR_noise'] = 0.1
    evaluate_times = 10

    ## -- random policy -- ##
    simulation_test_config['active_policy'] = False
    random_MSE, random_SSIM = [], []
    for i in trange(evaluate_times):
        test = SimulationTest(simulation_test_config, tactile_sensor_config, Control_config)
        mse, ssim = test.run(is_printInfo=False)
        random_MSE.append(mse)
        random_SSIM.append(ssim)
    res_data['random_MSE'] = random_MSE
    res_data['random_SSIM'] = random_SSIM
    # print(random_SSIM)
    
    ## -- active alpha=0.0 -- ##
    simulation_test_config['active_policy'] = True
    simulation_test_config['active_alpha'] = 0.0
    active_0_MSE, active_0_SSIM = [], []
    for i in trange(evaluate_times):
        test = SimulationTest(simulation_test_config, tactile_sensor_config, Control_config)
        mse, ssim = test.run(is_printInfo=False)
        active_0_MSE.append(mse)
        active_0_SSIM.append(ssim)
    res_data['active_0_MSE'] = active_0_MSE
    res_data['active_0_SSIM'] = active_0_SSIM
    # print(active_0_SSIM)
    
    # ## -- active alpha=0.5 -- ##
    simulation_test_config['active_policy'] = True
    simulation_test_config['active_alpha'] = 0.5
    active_05_MSE, active_05_SSIM = [], []
    for i in trange(evaluate_times):
        test = SimulationTest(simulation_test_config, tactile_sensor_config, Control_config)
        mse, ssim = test.run(is_printInfo=False)
        active_05_MSE.append(mse)
        active_05_SSIM.append(ssim)
    res_data['active_05_MSE'] = active_05_MSE
    res_data['active_05_SSIM'] = active_05_SSIM
    # print(active_05_SSIM)
    
    
    ## -- active alpha=1.0 -- ##
    simulation_test_config['active_policy'] = True
    simulation_test_config['active_alpha'] = 1.0
    active_1_MSE, active_1_SSIM = [], []
    for i in trange(evaluate_times):
        test = SimulationTest(simulation_test_config, tactile_sensor_config, Control_config)
        mse, ssim = test.run(is_printInfo=False)
        active_1_MSE.append(mse)
        active_1_SSIM.append(ssim)
    res_data['active_1_MSE'] = active_1_MSE
    res_data['active_1_SSIM'] = active_1_SSIM
    # print(active_1_SSIM)
    
    plot_results(res_data)
    


if __name__ == "__main__":
    main()


