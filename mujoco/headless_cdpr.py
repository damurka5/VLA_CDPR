import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Set this BEFORE importing mujoco

import mujoco as mj
import numpy as np
import cv2
import time
from datetime import datetime
import imageio

class HeadlessCDPRController:
    def __init__(self, frame_points, initial_pos=np.array([0, 0, 1.309])):
        self.frame_points = frame_points
        self.pos = initial_pos
        self.Kp = 100
        self.Kd = 130
        self.threshold = 0.03
        self.prev_lengths = np.zeros(4)
        self.dt = 1.0/60.0
        
    def inverse_kinematics(self, pos=None):
        if pos is None:
            pos = self.pos
            
        lengths = []
        for i in range(4):
            vec = pos - self.frame_points[i]
            lengths.append(np.linalg.norm(vec))
        return np.array(lengths)
    
    def update_position(self, new_pos):
        self.pos = new_pos.copy()
    
    def compute_control(self, target_pos, current_ee_pos):
        self.update_position(current_ee_pos)
        cur_lengths = self.inverse_kinematics()
        target_lengths = self.inverse_kinematics(target_pos)
        
        length_errors = target_lengths - cur_lengths
        cable_velocities = (cur_lengths - self.prev_lengths) / self.dt
        
        control_signals = self.Kp * length_errors - self.Kd * cable_velocities
        self.prev_lengths = cur_lengths.copy()
        
        return -control_signals

class HeadlessCDPRSimulation:
    def __init__(self, xml_path, output_dir="trajectory_videos"):
        self.xml_path = xml_path
        self.output_dir = output_dir
        self.model = None
        self.data = None
        
        # CDPR parameters
        self.frame_points = np.array([
            [-0.535, -0.755, 1.309],
            [0.755, -0.525, 1.309],
            [0.535, 0.755, 1.309],
            [-0.755, 0.525, 1.309]
        ])
        
        self.controller = HeadlessCDPRController(self.frame_points)
        self.target_pos = np.array([0, 0, 1.309])
        
        # Video recording
        self.overview_frames = []
        self.ee_camera_frames = []
        self.trajectory_data = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def initialize(self):
        """Initialize MuJoCo simulation with OSMesa for headless rendering"""
        # Load model
        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data = mj.MjData(self.model)
        
        # Initialize offscreen rendering
        self._setup_offscreen_rendering()
        
        # Initialize simulation
        mj.mj_forward(self.model, self.data)
        
        print("Headless CDPR Simulation initialized successfully!")
        print("Using OSMesa software rendering")
        
    def _setup_offscreen_rendering(self):
        """Setup offscreen rendering context using OSMesa"""
        # Camera setup
        self.overview_cam = mj.MjvCamera()
        self.ee_cam = mj.MjvCamera()
        
        # Configure overview camera
        self.overview_cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.overview_cam.distance = 5.0
        self.overview_cam.azimuth = 45
        self.overview_cam.elevation = -30
        
        # Configure end-effector camera
        self.ee_cam.type = mj.mjtCamera.mjCAMERA_FIXED
        if mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera") != -1:
            self.ee_cam.fixedcamid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera")
        
        # Scene and options
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.opt = mj.MjvOption()
        mj.mjv_defaultOption(self.opt)
        
        # Offscreen buffer dimensions
        self.offwidth, self.offheight = 640, 480
        self.offviewport = mj.MjrRect(0, 0, self.offwidth, self.offheight)
        
        # Initialize OSMesa context
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        
    def capture_frame(self, camera, camera_name):
        """Capture a frame from specified camera"""
        try:
            # Update scene
            mj.mjv_updateScene(self.model, self.data, self.opt, None, camera, 
                              mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            
            # Render to offscreen buffer
            mj.mjr_render(self.offviewport, self.scene, self.context)
            
            # Read pixels
            rgb_buffer = np.zeros((self.offheight, self.offwidth, 3), dtype=np.uint8)
            depth_buffer = np.zeros((self.offheight, self.offwidth), dtype=np.float32)
            mj.mjr_readPixels(rgb_buffer, depth_buffer, self.offviewport, self.context)
            
            # Process image
            rgb_buffer = np.flipud(rgb_buffer)
            return rgb_buffer
            
        except Exception as e:
            print(f"Error capturing frame from {camera_name}: {e}")
            # Return a blank frame if rendering fails
            return np.zeros((self.offheight, self.offwidth, 3), dtype=np.uint8)
    
    # ... keep the rest of your methods the same ...

def main():
    # Set environment variable for OSMesa BEFORE any mujoco imports
    os.environ['MUJOCO_GL'] = 'osmesa'
    
    # Path to your XML file
    xml_path = "mujoco/cdpr.xml"
    
    # Check if XML file exists
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        print("Please ensure the cdpr.xml file exists")
        return
    
    # Create simulation
    sim = HeadlessCDPRSimulation(xml_path, output_dir="trajectory_results")
    
    try:
        sim.initialize()
        
        # Define trajectory waypoints
        trajectories = {
            "simple_test": [
                [0.3, 0.3, 1.2],
                [-0.3, -0.3, 1.2],
                [0.0, 0.0, 1.309]
            ],
            "square_trajectory": [
                [0.5, 0.5, 1.0],
                [0.5, -0.5, 1.0],
                [-0.5, -0.5, 1.0],
                [-0.5, 0.5, 1.0],
                [0.0, 0.0, 1.309]
            ]
        }
        
        # Run each trajectory
        for traj_name, waypoints in trajectories.items():
            print(f"\n{'='*50}")
            print(f"Running trajectory: {traj_name}")
            print(f"{'='*50}")
            
            success = sim.run_trajectory(waypoints, traj_name, max_steps_per_target=300)
            
            if success:
                print(f"✓ Trajectory '{traj_name}' completed successfully!")
            else:
                print(f"⚠ Trajectory '{traj_name}' had some issues")
            
            # Small pause between trajectories
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nTrajectory recording interrupted by user")
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sim.cleanup()

if __name__ == "__main__":
    main()