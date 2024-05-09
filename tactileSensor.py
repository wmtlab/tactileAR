import pyrender
import trimesh
from scipy.spatial.transform import Rotation as R
import numpy as np
import os, sys

class TactileSensor():
    def __init__(self, sensor_config, degrees=True):
        self.state_scale = sensor_config["state_scale"]
        self.H, self.W = sensor_config["contact_plate_H"], sensor_config["contact_plate_W"] 
        self.sensor_H, self.sensor_W = self.state_scale*sensor_config["sensor_H"], self.state_scale*sensor_config["sensor_W"]
        
        self.mmPerPixel = sensor_config["mmPerPixel"]
        self.pixelPerMm = 1 / self.mmPerPixel
        
        self.contact_plate_resolution_height, self.contact_plate_resolution_width = self.H*self.mmPerPixel, self.W*self.mmPerPixel
        self.state_resolution_height, self.state_resolution_width = self.sensor_H*self.mmPerPixel, self.sensor_W*self.mmPerPixel
        
        self.sensor_depth_x_idx_start, self.sensor_depth_x_idx_end = int(((1-1/self.state_scale)*self.state_resolution_height)//2), int(((1+1/self.state_scale)*self.state_resolution_height)//2)
        self.sensor_depth_y_idx_start, self.sensor_depth_y_idx_end = int(((1-1/self.state_scale)*self.state_resolution_width)//2), int(((1+1/self.state_scale)*self.state_resolution_width)//2)
        
        self.contact_scene = pyrender.Scene()
        self.sensor_scene = pyrender.Scene()
        
        self.init_mesh_angles, self.init_mesh_translation = sensor_config["contact_plate_angles"], sensor_config["contact_plate_translation"]
        
        # add contact surface
        mesh = trimesh.load(sensor_config["contact_plate_path"])
        mesh_pose = self.euler2matrix(angles=self.init_mesh_angles, translation=self.init_mesh_translation, degrees=degrees)
        self.mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh), matrix=mesh_pose) # 可以利用pyrender的node更新位置信息
        self.contact_scene.add_node(self.mesh_node)
        self.sensor_scene.add_node(self.mesh_node)
        
        # add camera
        contact_scene_camera = pyrender.camera.OrthographicCamera(xmag=self.H*0.1/2,
                                                                  ymag=self.W*0.1/2,
                                                                 )
        contact_scene_camera_pose = self.euler2matrix(angles=[0, 0, 0], translation=[0, 0, 1])
        self.contact_scene_camera_node = pyrender.Node(camera=contact_scene_camera, matrix=contact_scene_camera_pose)
        self.contact_scene.add_node(self.contact_scene_camera_node)
        
        sensor_scene_camera = pyrender.camera.OrthographicCamera(xmag=self.sensor_H*0.1/2,
                                                                 ymag=self.sensor_W*0.1/2,
                                                                 )
        sensor_scene_camera_pose = self.euler2matrix(angles=[0, 0, 0], translation=[0, 0, 1])
        self.sensor_scene_camera_node = pyrender.Node(camera=sensor_scene_camera, matrix=sensor_scene_camera_pose)
        self.sensor_scene.add_node(self.sensor_scene_camera_node)
        
        scene_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        scene_light_pose = self.euler2matrix(angles=[0, 0, 0], translation=[0, 0, 10])
        self.scene_light_node = pyrender.Node(light=scene_light, matrix=scene_light_pose)
        self.contact_scene.add_node(self.scene_light_node)
        self.sensor_scene.add_node(self.scene_light_node)

    @property
    def contact_plate_resoultion(self, ):
        return (int(self.contact_plate_resolution_height), int(self.contact_plate_resolution_width))
    
    @property
    def state_resoultion(self, ):
        return (int(self.state_resolution_height), int(self.state_resolution_width))
    
    @property
    def tactile_resolution(self, ):
        return (int(self.sensor_depth_x_idx_end-self.sensor_depth_x_idx_start), int(self.sensor_depth_y_idx_end-self.sensor_depth_y_idx_start))
    
    @property
    def get_mmPerPixel(self, ):
        return self.mmPerPixel

    def get_contact_scene_image(self,):
        r = pyrender.OffscreenRenderer(self.contact_plate_resolution_width, self.contact_plate_resolution_height)
        color, depth = r.render(self.contact_scene)
        return color, depth
    
    def get_sensor_scene_image(self, x_mm=0, y_mm=0, theta_deg=0, degrees=True, is_camera_move=True):
        x, y, theta_deg = x_mm*0.1, y_mm*0.1, theta_deg
        if is_camera_move==True: # camera move, object fix
            new_camera_pose = self.euler2matrix(angles=[0, 0, theta_deg], translation=[x, y, 1], degrees=degrees)
            self.sensor_scene_camera_node.matrix = new_camera_pose
        else: # camera fix, object move
            new_mesh_pose = self.euler2matrix(angles=[self.init_mesh_angles[0], self.init_mesh_angles[1], theta_deg], translation=[x, y, self.init_mesh_translation[2]], degrees=degrees)
            self.mesh_node.matrix = new_mesh_pose
        
        r = pyrender.OffscreenRenderer(self.state_resolution_width, self.state_resolution_height)
        color, depth = r.render(self.sensor_scene)
        sensor_depth = depth[self.sensor_depth_x_idx_start:self.sensor_depth_x_idx_end, \
                             self.sensor_depth_y_idx_start:self.sensor_depth_y_idx_end]
        
        depth = 1 - (depth - depth.min()) / (depth.max() - depth.min())
        if not (sensor_depth.max() == sensor_depth.min()):
            sensor_depth = 1 - (sensor_depth - sensor_depth.min()) / (sensor_depth.max() - sensor_depth.min())
        return color, depth, sensor_depth
        
        
    def euler2matrix(self, angles=[0, 0, 0], translation=[0, 0, 0], xyz="xyz", degrees=True):
        r = R.from_euler(xyz, angles, degrees=degrees)
        pose = np.eye(4)
        pose[:3, 3] = translation
        pose[:3, :3] = r.as_matrix()
        return pose


if __name__ == "__main__":
    
    ## -- debug sensor  -- ##
    sys.path.append(os.path.abspath('.'))
    from config import tactile_sensor_config
    import matplotlib.pyplot as plt
    
    t_sensor = TactileSensor(sensor_config=tactile_sensor_config) 
    contact_plate_color, contact_surface_depth = t_sensor.get_contact_scene_image()
    
    for i in range(11):
        state_color, state_depth, tactile_depth = t_sensor.get_sensor_scene_image(theta_deg=5*i)
        plt.imshow(state_color)
        plt.pause(0.1)        
    plt.show()

