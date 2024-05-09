import os
import sys
import numpy as np
import argparse


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Simulation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--policy', type=str, default='random', help='Sampling policy (default: active)')
    parser.add_argument('--active_alpha', type=float, default=0.0, help='Sampling policy hyper-parameter (default: 0.0)')
    parser.add_argument('--max_epoch', type=int, default=30, help='Maximum number of epochs (default: 30)')
    parser.add_argument('--is_outVideo', type=bool, default=False, help='Output video (default: False)')
    parser.add_argument('--state_scale', type=float, default=2.0, help='State scale (default: 2.0)')
    parser.add_argument('--mmPerPixel', type=int, default=2, help='Millimeters per pixel (default: 2)')
    parser.add_argument('--contact_plate_path', type=str, default='contact_surface/demo.stl')
    parser.add_argument('--LR_noise', type=float, default=0.1, help='Tactile sensor noise (default: 0.1)')
    
    # Control configuration
    parser.add_argument('--x_min', type=float, default=-20, help='Minimum x position in mm (default: -20)')
    parser.add_argument('--x_max', type=float, default=20, help='Maximum x position in mm (default: 20)')
    parser.add_argument('--y_min', type=float, default=-20, help='Minimum y position in mm (default: -20)')
    parser.add_argument('--y_max', type=float, default=20, help='Maximum y position in mm (default: 20)')
    parser.add_argument('--theta_min', type=float, default=-90, help='Minimum theta in degrees (default: -90)')
    parser.add_argument('--theta_max', type=float, default=90, help='Maximum theta in degrees (default: 90)')
    parser.add_argument('--xy_stepSize', type=float, default=0.5, help='Step size for x and y movements in mm (default: 0.5)')
    parser.add_argument('--theta_stepSize', type=float, default=1, help='Step size for theta movements in degrees (default: 1)')
    
    # Tactile reconstruction parameters
    parser.add_argument('--dm_gamma', type=float, default=4.0, help='Degradation matrix gamma (default: 4.0)')
    parser.add_argument('--mu', type=float, default=0.0, help='Initial prior mean (default: 0.0)')
    parser.add_argument('--A', type=float, default=0.05, help='Initial prior A parameter (default: 0.05)')
    parser.add_argument('--r', type=int, default=30, help='Initial prior r parameter (default: 30)')

    return parser.parse_args()

def update_config_from_args(args):
    """Update configurations based on command line arguments."""
    tactile_sensor_config['state_scale'] = args.state_scale
    tactile_sensor_config['contact_plate_path'] = os.path.join(root_path, args.contact_plate_path)
    tactile_sensor_config['mmPerPixel'] = args.mmPerPixel

    simulation_test_config['active_policy'] = args.policy.lower() == 'active'
    simulation_test_config['dm_gamma'] = args.dm_gamma
    simulation_test_config['mu'] = args.mu
    simulation_test_config['A'] = args.A
    simulation_test_config['r'] = args.r
    simulation_test_config['max_epoch'] = args.max_epoch
    simulation_test_config['LR_noise'] = args.LR_noise
    simulation_test_config['active_alpha'] = args.active_alpha
    simulation_test_config['is_outVideo'] = args.is_outVideo

    Control_config['x_min'] = args.x_min
    Control_config['x_max'] = args.x_max
    Control_config['y_min'] = args.y_min
    Control_config['y_max'] = args.y_max
    Control_config['theta_min'] = args.theta_min
    Control_config['theta_max'] = args.theta_max
    Control_config['xy_stepSize'] = args.xy_stepSize
    Control_config['theta_stepSize'] = args.theta_stepSize


# Use absolute path to ensure reliability across different environments
root_path = os.path.dirname(os.path.abspath(__file__))
print("==> ROOT PATH:", root_path)

# Configuration for simulation tests
simulation_test_config = {
    "is_real_dm": False,
    "active_policy": True,
    "dm_file_path": None,
    "dm_gamma": 4.0,
    "mu": 0.0,
    "A": 0.05,
    "r": 30,
    "max_epoch": 30,
    "LR_noise": 0.1,
    "active_alpha": 0,
    "is_outVideo": False,
}

# Configuration for the tactile sensor setup
tactile_sensor_config = {
    "contact_plate_H": 100,
    "contact_plate_W": 100,
    "sensor_H": 20.04,  # Calculated as 5.01 * 4
    "sensor_W": 20.04,
    "state_scale": 2.0,
    "contact_plate_path": None,
    "contact_plate_angles": [90, 0, 0],
    "contact_plate_translation": [0, 0, 0],
    "mmPerPixel": 2,  # Millimeter per pixel
}

# Configuration for motion control
Control_config = {
    "x_min": -20,  # mm
    "x_max": 20,
    "y_min": -20,
    "y_max": 20,
    "theta_min": -90,  # degrees
    "theta_max": 90,
    "xy_stepSize": 0.5,
    "theta_stepSize": 1,
}

# Tactile sensor layout configurations
taxel_centre_list_x5 = np.array([[4 + 4 * i for i in range(4)], [4 + 4 * i for i in range(4)]])
taxel_centre_list_x10 = np.array([[6 + 9 * i for i in range(4)], [6 + 9 * i for i in range(4)]])


