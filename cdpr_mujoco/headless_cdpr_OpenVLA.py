import os
import mujoco as mj
import numpy as np
import cv2
import time
from datetime import datetime
import imageio
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# Try to import EGL for true headless rendering
try:
    from mujoco.egl import GLContext
    EGL_AVAILABLE = True
except ImportError:
    EGL_AVAILABLE = False
    print("EGL not available, falling back to software rendering")

class OpenVLAPlanner:
    def __init__(self, model_name="openvla/openvla-7b"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"OpenVLA using device: {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)
        
    def get_target_position(self, image, instruction="go near the red square"):
        """Get target position from OpenVLA model using overview camera image"""
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Process inputs and run inference
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        
        # Handle different return types from OpenVLA
        if isinstance(action, torch.Tensor):
            # If it's a PyTorch tensor, convert to numpy
            target_pos = action[:3].cpu().numpy()
        elif isinstance(action, np.ndarray):
            # If it's already a numpy array, just take first 3 elements
            target_pos = action[:3].copy()
        elif isinstance(action, (list, tuple)) and len(action) >= 3:
            # If it's a list or tuple, convert to numpy array
            target_pos = np.array(action[:3])
        else:
            # Return default position if extraction fails
            target_pos = np.array([0.0, 0.0, 1.0])
            print(f"Unexpected action type: {type(action)}, using default position")
        
        print(f"OpenVLA target position: {target_pos}")
        return target_pos

class CDPRController:
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
        
        slider_positions = 0.9 - target_lengths
        return slider_positions

class CDPRSimulation:
    def __init__(self, xml_path, output_dir="openvla_trajectories"):
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
        
        self.controller = CDPRController(self.frame_points)
        self.target_pos = np.array([0, 0, 1.309])
        
        # OpenVLA integration
        self.vla_planner = OpenVLAPlanner()
        self.instruction = "go near the red square"
        
        # Recording
        self.overview_frames = []
        self.trajectory_data = []
        
        os.makedirs(output_dir, exist_ok=True)
        
    def initialize(self):
        """Initialize MuJoCo simulation"""
        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data = mj.MjData(self.model)
        self.model.opt.timestep = self.controller.dt
        
        self._setup_offscreen_rendering()
        mj.mj_forward(self.model, self.data)
        
        print("CDPR Simulation with OpenVLA initialized!")
        
    def _setup_offscreen_rendering(self):
        """Setup offscreen rendering context"""
        self.overview_cam = mj.MjvCamera()
        self.ee_cam = mj.MjvCamera()

        # Configure overview camera
        self.overview_cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.overview_cam.distance = 3.0
        self.overview_cam.azimuth = 0
        self.overview_cam.elevation = -25

        # Configure end-effector camera
        self.ee_cam.type = mj.mjtCamera.mjCAMERA_FIXED
        self.ee_cam.fixedcamid = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera"
        )

        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.opt = mj.MjvOption()
        mj.mjv_defaultOption(self.opt)

        self.offwidth, self.offheight = 640, 480
        self.offviewport = mj.MjrRect(0, 0, self.offwidth, self.offheight)

        if EGL_AVAILABLE:
            self.gl_context = GLContext(max_width=self.offwidth, max_height=self.offheight)
            self.gl_context.make_current()
            self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        else:
            self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        
    def capture_frame(self, camera, camera_name):
        """Capture a frame from specified camera"""
        try:
            mj.mjv_updateScene(self.model, self.data, self.opt, None, camera, 
                              mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(self.offviewport, self.scene, self.context)
            
            rgb_buffer = np.zeros((self.offheight, self.offwidth, 3), dtype=np.uint8)
            depth_buffer = np.zeros((self.offheight, self.offwidth), dtype=np.float32)
            mj.mjr_readPixels(rgb_buffer, depth_buffer, self.offviewport, self.context)
            
            rgb_buffer = np.flipud(rgb_buffer)
            return rgb_buffer
            
        except Exception as e:
            print(f"Error capturing frame from {camera_name}: {e}")
            return np.zeros((self.offheight, self.offwidth, 3), dtype=np.uint8)
    
    def get_end_effector_position(self):
        """Get current end-effector position"""
        return self.data.qpos[4:7].copy()
    
    def set_target_position(self, target_pos):
        """Set new target position"""
        self.target_pos = np.clip(target_pos, -1.309, 1.309)
        cur_lengths = self.controller.inverse_kinematics()
        self.controller.prev_lengths = cur_lengths.copy()
        return True
    
    def check_success(self):
        """Check if end-effector reached target"""
        ee_pos = self.get_end_effector_position()
        error = np.linalg.norm(ee_pos - self.target_pos)
        return error < self.controller.threshold
    
    def run_simulation_step(self):
        """Run one simulation step"""
        ee_pos = self.get_end_effector_position()
        
        # Compute and apply control
        control_signals = self.controller.compute_control(self.target_pos, ee_pos)
        for j in range(4):
            self.data.ctrl[j] = control_signals[j]
        
        # Step simulation
        mj.mj_step(self.model, self.data)
        
        # Capture overview frame for recording
        overview_frame = self.capture_frame(self.overview_cam, "overview")
        self.overview_frames.append(overview_frame)
        
        # Record trajectory data
        ee_pos = self.get_end_effector_position()
        self.trajectory_data.append({
            'timestamp': self.data.time,
            'ee_position': ee_pos.copy(),
            'target_position': self.target_pos.copy()
        })
    
    def run_openvla_trajectory(self, max_steps=1000, vla_update_interval=100):
        """Run trajectory planning with OpenVLA"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_dir = os.path.join(self.output_dir, f"openvla_trajectory_{timestamp}")
        os.makedirs(trajectory_dir, exist_ok=True)
        
        print(f"Starting OpenVLA trajectory")
        print(f"Instruction: {self.instruction}")
        
        self.overview_frames = []
        self.trajectory_data = []
        
        for step in range(max_steps):
            # Use OpenVLA to plan at specified intervals
            if step % vla_update_interval == 0:
                overview_frame = self.capture_frame(self.overview_cam, "overview")
                new_target = self.vla_planner.get_target_position(overview_frame, self.instruction)
                self.set_target_position(new_target)
                print(f"Step {step}: New target = {new_target}")
            
            # Run simulation
            self.run_simulation_step()
            
            # Check for success
            if self.check_success():
                print(f"Target reached in {step} steps!")
                break
            
            # Progress reporting
            if step % 50 == 0:
                ee_pos = self.get_end_effector_position()
                error = np.linalg.norm(ee_pos - self.target_pos)
                print(f"Step {step}: Error={error:.4f}")
        
        # Save results
        self._save_results(trajectory_dir)
        return True
    
    def _save_results(self, trajectory_dir):
        """Save trajectory results"""
        # Save video
        if self.overview_frames:
            video_path = os.path.join(trajectory_dir, "trajectory_video.mp4")
            with imageio.get_writer(video_path, fps=20) as writer:
                for frame in self.overview_frames:
                    writer.append_data(frame)
            print(f"Video saved: {video_path}")
        
        # Save trajectory data
        if self.trajectory_data:
            data_path = os.path.join(trajectory_dir, "trajectory_data.npz")
            timestamps = np.array([data['timestamp'] for data in self.trajectory_data])
            ee_positions = np.array([data['ee_position'] for data in self.trajectory_data])
            target_positions = np.array([data['target_position'] for data in self.trajectory_data])
            
            np.savez(data_path, 
                    timestamps=timestamps,
                    ee_positions=ee_positions,
                    target_positions=target_positions)
            print(f"Data saved: {data_path}")
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'context'):
            try:
                mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW, self.context)
            except:
                pass
        if hasattr(self, 'gl_context') and self.gl_context:
            try:
                self.gl_context.free()
            except:
                pass

def main():
    xml_path = "cdpr.xml"
    
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return
    
    sim = CDPRSimulation(xml_path)
    
    try:
        sim.initialize()
        sim.run_openvla_trajectory(max_steps=800, vla_update_interval=100)
        print("✓ OpenVLA trajectory completed!")
            
    except KeyboardInterrupt:
        print("\nTrajectory interrupted by user")
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sim.cleanup()

if __name__ == "__main__":
    main()