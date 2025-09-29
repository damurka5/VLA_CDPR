import mujoco as mj
import numpy as np
import time
import cv2
import os

import os
os.environ["MUJOCO_GL"] = "osmesa"
import mujoco as mj

class CDPR4:
    def __init__(self, approx=1, pos=np.array([0, 0, 0])):
        self.approx = approx
        self.pos = pos
        self.frame_points = np.array([
            [-0.535, -0.755, 1.309],
            [0.755, -0.525, 1.309],
            [0.535, 0.755, 1.309],
            [-0.755, 0.525, 1.309]
        ])
        self.prev_lengths = np.zeros(4)
    
    def inverse_kinematics(self, pos=None):
        if pos is None:
            pos = self.pos
        lengths = []
        for i in range(4):
            vec = pos - self.frame_points[i]
            lengths.append(np.linalg.norm(vec))
        return tuple(lengths)

def setup_offscreen_rendering(model, width=640, height=480):
    """Set up offscreen rendering using OSMesa"""
    # Create scene and camera
    scene = mj.MjvScene(model, maxgeom=10000)
    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)
    
    # Create offscreen context - this should work with OSMesa
    con = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    # Set offscreen buffer
    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, con)
    
    # Initialize buffers
    rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.zeros((height, width), dtype=np.float32)
    
    return scene, cam, opt, con, rgb_buffer, depth_buffer, (width, height)

# Load model
xml_path = 'cdpr.xml'
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# Get camera ID
camera_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera")
print(f"Camera ID: {camera_id}")

try:
    # Setup offscreen rendering
    scene, cam, opt, con, rgb_buffer, depth_buffer, (width, height) = setup_offscreen_rendering(model)
    print("Offscreen rendering setup successful!")
except Exception as e:
    print(f"Offscreen rendering failed: {e}")
    print("Falling back to sensor data only...")
    # You can still use sensor data even if rendering fails
    scene = cam = opt = con = None

# Initialize robot controller
initial_pos = np.array([0, 0, 1.309])
robot = CDPR4(approx=1, pos=initial_pos)

# Simulation parameters
simend = 10  # Shorter for testing
threshold = 0.03
received_goal = False
target_pos = np.array([0, 0, 1.309])

# Pre-defined target positions
target_sequence = [
    np.array([0.3, 0.3, 1.1]),
    np.array([-0.2, 0.2, 1.2]),
]
current_target_index = 0

# Create frames directory
os.makedirs('frames', exist_ok=True)

print("CDPR Simulation Started - Attempting to Capture Frames")
print("Press Ctrl+C to stop the simulation")

frame_count = 0
successful_frames = 0

try:
    while data.time < simend:
        # Get current end-effector position
        Ac = data.qpos[4:7]
        robot.pos = Ac.copy()
        
        # Automated target sequencing
        if not received_goal and current_target_index < len(target_sequence):
            target_pos = target_sequence[current_target_index]
            received_goal = True
            cur_L_1, cur_L_2, cur_L_3, cur_L_4 = robot.inverse_kinematics()
            robot.prev_lengths = np.array([cur_L_1, cur_L_2, cur_L_3, cur_L_4])
            print(f"Target {current_target_index + 1} set to: {target_pos}")
        
        # Control logic
        cur_L_1, cur_L_2, cur_L_3, cur_L_4 = robot.inverse_kinematics()
        cur_lengths = np.array([cur_L_1, cur_L_2, cur_L_3, cur_L_4])
        
        target_L_1, target_L_2, target_L_3, target_L_4 = robot.inverse_kinematics(target_pos)
        target_lengths = np.array([target_L_1, target_L_2, target_L_3, target_L_4])
        
        if received_goal:
            slider_positions = 0.9 - target_lengths
            for i in range(4):
                data.ctrl[i] = slider_positions[i]
            
            # Try to capture frame every 20 steps
            if frame_count % 20 == 0 and scene is not None:
                try:
                    # Set camera to use the end-effector camera
                    cam.type = mj.mjtCamera.mjCAMERA_FIXED
                    cam.fixedcamid = camera_id
                    
                    # Render the scene
                    viewport = mj.MjrRect(0, 0, width, height)
                    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
                    mj.mjr_render(viewport, scene, con)
                    
                    # Read pixels
                    mj.mjr_readPixels(rgb_buffer, depth_buffer, viewport, con)
                    
                    # Process image
                    image = np.flipud(rgb_buffer)
                    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Save image
                    cv2.imwrite(f'frames/frame_{frame_count:06d}.png', bgr_image)
                    successful_frames += 1
                    print(f"Successfully saved frame {frame_count}")
                    
                except Exception as e:
                    print(f"Failed to capture frame {frame_count}: {e}")
            
            if np.all(np.abs(target_lengths - cur_lengths) < threshold):
                print(f"Reached target position {current_target_index + 1}!")
                received_goal = False
                current_target_index += 1
                if current_target_index >= len(target_sequence):
                    print("All target positions reached!")
                    break
        
        # Step simulation
        mj.mj_step(model, data)
        frame_count += 1
        
except KeyboardInterrupt:
    print("\nSimulation stopped by user")
except Exception as e:
    print(f"Simulation error: {e}")

finally:
    if con is not None:
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW, con)
    
    print(f"Simulation finished. Successfully captured {successful_frames}/{frame_count} frames")
    print("Final end-effector position:", data.qpos[4:7])