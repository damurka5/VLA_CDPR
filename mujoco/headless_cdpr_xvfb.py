import os
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
    def __init__(self, xml_path, output_dir="trajectory_videos", render=True):
        self.xml_path = xml_path
        self.output_dir = output_dir
        self.render = render  # Control whether to render images
        self.model = None
        self.data = None
        self.scene = None
        self.context = None
        
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
        """Initialize MuJoCo simulation"""
        # Load model
        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data = mj.MjData(self.model)
        
        if self.render:
            # Initialize rendering only if needed
            self._setup_rendering()
        
        # Initialize simulation
        mj.mj_forward(self.model, self.data)
        
        print(f"Headless CDPR Simulation initialized successfully! Rendering: {self.render}")
        
    def _setup_rendering(self):
        """Setup rendering context"""
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
        ee_cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera")
        if ee_cam_id != -1:
            self.ee_cam.fixedcamid = ee_cam_id
        
        # Scene and options
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.opt = mj.MjvOption()
        mj.mjv_defaultOption(self.opt)
        
        # Initialize context
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        
        # Offscreen buffer dimensions
        self.offwidth, self.offheight = 640, 480
        self.offviewport = mj.MjrRect(0, 0, self.offwidth, self.offheight)
        
    def capture_frame(self, camera, camera_name):
        """Capture a frame from specified camera"""
        if not self.render or self.scene is None:
            return np.zeros((self.offheight, self.offwidth, 3), dtype=np.uint8)
            
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
            return np.zeros((self.offheight, self.offwidth, 3), dtype=np.uint8)
    
    def get_end_effector_position(self):
        """Get current end-effector position"""
        return self.data.qpos[4:7].copy()
    
    def set_target_position(self, target_pos):
        """Set new target position"""
        if all(-1.309 <= coord <= 1.309 for coord in target_pos):
            self.target_pos = np.array(target_pos)
            ee_pos = self.get_end_effector_position()
            cur_lengths = self.controller.inverse_kinematics(ee_pos)
            self.controller.prev_lengths = cur_lengths.copy()
            return True
        return False
    
    def check_success(self):
        """Check if end-effector reached target"""
        ee_pos = self.get_end_effector_position()
        cur_lengths = self.controller.inverse_kinematics(ee_pos)
        target_lengths = self.controller.inverse_kinematics(self.target_pos)
        length_errors = np.abs(target_lengths - cur_lengths)
        return np.all(length_errors < self.controller.threshold)
    
    def record_trajectory_step(self):
        """Record current state for trajectory data"""
        ee_pos = self.get_end_effector_position()
        slider_positions = [self.data.qpos[i] for i in range(4)]
        cable_lengths = self.controller.inverse_kinematics(ee_pos)
        
        self.trajectory_data.append({
            'timestamp': self.data.time,
            'ee_position': ee_pos.copy(),
            'target_position': self.target_pos.copy(),
            'slider_positions': slider_positions.copy(),
            'cable_lengths': cable_lengths.copy(),
            'control_signals': self.data.ctrl.copy() if len(self.data.ctrl) > 0 else [0, 0, 0, 0]
        })
    
    def run_simulation_step(self, capture_frame=True):
        """Run one simulation step"""
        ee_pos = self.get_end_effector_position()
        
        # Compute and apply control
        control_signals = self.controller.compute_control(self.target_pos, ee_pos)
        for j in range(min(4, len(self.data.ctrl))):
            self.data.ctrl[j] = control_signals[j]
        
        # Step simulation
        mj.mj_step(self.model, self.data)
        
        # Capture frames if requested and rendering is enabled
        if capture_frame and self.render:
            overview_frame = self.capture_frame(self.overview_cam, "overview")
            ee_frame = self.capture_frame(self.ee_cam, "ee_camera")
            
            self.overview_frames.append(overview_frame)
            self.ee_camera_frames.append(ee_frame)
        
        # Record trajectory data
        self.record_trajectory_step()
    
    def run_trajectory(self, target_positions, trajectory_name="trajectory", 
                      max_steps_per_target=600, capture_every_n=3):
        """Run a complete trajectory through multiple target positions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_dir = os.path.join(self.output_dir, f"{trajectory_name}_{timestamp}")
        os.makedirs(trajectory_dir, exist_ok=True)
        
        print(f"Starting trajectory: {trajectory_name}")
        
        # Reset recording
        if self.render:
            self.overview_frames = []
            self.ee_camera_frames = []
        self.trajectory_data = []
        
        total_steps = 0
        trajectory_success = True
        
        for i, target_pos in enumerate(target_positions):
            print(f"Moving to target {i+1}/{len(target_positions)}: {target_pos}")
            
            if not self.set_target_position(target_pos):
                print(f"Invalid target position: {target_pos}")
                trajectory_success = False
                break
            
            # Run until target reached or timeout
            steps_for_target = 0
            target_reached = False
            
            while steps_for_target < max_steps_per_target:
                # Run simulation step (capture frames periodically)
                capture_this_frame = (total_steps % capture_every_n == 0) and self.render
                self.run_simulation_step(capture_frame=capture_this_frame)
                
                total_steps += 1
                steps_for_target += 1
                
                # Check if target reached
                if self.check_success():
                    print(f"Target {i+1} reached in {steps_for_target} steps")
                    target_reached = True
                    break
            
            if not target_reached:
                print(f"Timeout reaching target {i+1} after {max_steps_per_target} steps")
                trajectory_success = False
                # Continue to next target anyway
        
        # Save results
        self.save_trajectory_results(trajectory_dir, trajectory_name)
        return trajectory_success
    
    def save_trajectory_results(self, trajectory_dir, trajectory_name):
        """Save all trajectory data"""
        print("Saving trajectory results...")
        
        # Save videos if we have frames and rendering is enabled
        if self.render and self.overview_frames:
            try:
                overview_video_path = os.path.join(trajectory_dir, "overview_video.mp4")
                self.save_video(self.overview_frames, overview_video_path, fps=20)
                print(f"Overview video saved: {overview_video_path}")
            except Exception as e:
                print(f"Error saving overview video: {e}")
        
        if self.render and self.ee_camera_frames:
            try:
                ee_video_path = os.path.join(trajectory_dir, "ee_camera_video.mp4")
                self.save_video(self.ee_camera_frames, ee_video_path, fps=20)
                print(f"End-effector video saved: {ee_video_path}")
            except Exception as e:
                print(f"Error saving EE camera video: {e}")
        
                # Save raw frames for inspection
        if self.render and self.ee_camera_frames:
            try:
                frames_dir = os.path.join(trajectory_dir, "ee_camera_frames")
                os.makedirs(frames_dir, exist_ok=True)
                
                for i, frame in enumerate(self.ee_camera_frames):
                    frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
                    imageio.imwrite(frame_path, frame)
                
                print(f"Saved {len(self.ee_camera_frames)} raw EE camera frames to {frames_dir}")
            except Exception as e:
                print(f"Error saving raw EE frames: {e}")

        if self.render and self.overview_frames:
            try:
                frames_dir = os.path.join(trajectory_dir, "overview_frames")
                os.makedirs(frames_dir, exist_ok=True)
                
                for i, frame in enumerate(self.overview_frames):
                    frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
                    imageio.imwrite(frame_path, frame)
                
                print(f"Saved {len(self.overview_frames)} raw overview frames to {frames_dir}")
            except Exception as e:
                print(f"Error saving raw overview frames: {e}")

        
        # Save trajectory data (always save this)
        try:
            trajectory_file = os.path.join(trajectory_dir, "trajectory_data.npz")
            self.save_trajectory_data(trajectory_file)
            print(f"Trajectory data saved: {trajectory_file}")
        except Exception as e:
            print(f"Error saving trajectory data: {e}")
        
        # Save summary
        try:
            self.save_summary(trajectory_dir, trajectory_name)
        except Exception as e:
            print(f"Error saving summary: {e}")
            
        
        
        print(f"Results saved to: {trajectory_dir}")
    
    def save_video(self, frames, filepath, fps=30):
        """Save frames as MP4 video"""
        if not frames:
            return
        
        with imageio.get_writer(filepath, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
    
    def save_trajectory_data(self, filepath):
        """Save trajectory data as numpy file"""
        if not self.trajectory_data:
            return
            
        trajectory_dict = {}
        
        arrays_to_save = ['timestamp', 'ee_position', 'target_position', 
                         'slider_positions', 'cable_lengths', 'control_signals']
        
        for key in arrays_to_save:
            if key in self.trajectory_data[0]:
                trajectory_dict[key] = np.array([data[key] for data in self.trajectory_data])
        
        np.savez(filepath, **trajectory_dict)
    
    def save_summary(self, trajectory_dir, trajectory_name):
        """Save a text summary of the trajectory"""
        summary_file = os.path.join(trajectory_dir, "summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"CDPR Trajectory Summary\n")
            f.write(f"=======================\n")
            f.write(f"Trajectory: {trajectory_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Rendering enabled: {self.render}\n")
            if self.render:
                f.write(f"Frames captured: {len(self.overview_frames)}\n")
            f.write(f"Simulation steps: {len(self.trajectory_data)}\n")
            f.write(f"Simulation time: {self.data.time:.2f} seconds\n")
            if self.trajectory_data:
                f.write(f"Final EE position: {self.trajectory_data[-1]['ee_position']}\n")
                f.write(f"Final target position: {self.trajectory_data[-1]['target_position']}\n")
    
    def cleanup(self):
        """Cleanup resources"""
        print("Simulation cleanup completed")

def main():
    # Path to your XML file
    xml_path = "cdpr.xml"
    
    # Check if XML file exists
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        print("Please ensure the cdpr.xml file exists")
        return
    
    # Create simulation with rendering enabled
    sim = HeadlessCDPRSimulation(xml_path, output_dir="trajectory_results", render=True)
    
    try:
        sim.initialize()
        
        # Define trajectory waypoints
        trajectories = {
            "simple_test": [
                [0.3, 0.3, 1.2],
                [-0.3, -0.3, 1.2],
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