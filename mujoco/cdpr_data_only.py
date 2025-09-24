import os
# Try to avoid OpenGL initialization
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/root/.mujoco/mujoco200'  # Adjust path as needed
os.environ['MUJOCO_PY_MJKEY_PATH'] = '/root/.mujoco/mjkey.txt'   # Adjust path as needed

try:
    from mujoco_py import load_model_from_path, MjSim, MjViewer
    MUJOCO_PY_AVAILABLE = True
except ImportError:
    MUJOCO_PY_AVAILABLE = False
    print("mujoco-py not available, trying alternative approach")

import numpy as np
import time
from datetime import datetime

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
        
        control_signals = self.Kp * length_errors - self.Kd * cable_velocities
        self.prev_lengths = cur_lengths.copy()
        
        return -control_signals

class CDPRSimulationMuJoCoPy:
    def __init__(self, xml_path, output_dir="trajectory_data"):
        self.xml_path = xml_path
        self.output_dir = output_dir
        
        if not MUJOCO_PY_AVAILABLE:
            raise ImportError("mujoco-py is not available")
        
        self.model = None
        self.sim = None
        
        # CDPR parameters
        self.frame_points = np.array([
            [-0.535, -0.755, 1.309],
            [0.755, -0.525, 1.309],
            [0.535, 0.755, 1.309],
            [-0.755, 0.525, 1.309]
        ])
        
        self.controller = CDPRController(self.frame_points)
        self.target_pos = np.array([0, 0, 1.309])
        self.trajectory_data = []
        
        os.makedirs(output_dir, exist_ok=True)
        
    def initialize(self):
        """Initialize MuJoCo simulation using mujoco-py"""
        try:
            self.model = load_model_from_path(self.xml_path)
            self.sim = MjSim(self.model)
            print("✓ MuJoCo simulation initialized successfully with mujoco-py")
            return True
        except Exception as e:
            print(f"✗ Error initializing simulation: {e}")
            return False
    
    def get_end_effector_position(self):
        """Get current end-effector position"""
        if self.sim is None:
            return np.array([0, 0, 1.309])
        return self.sim.data.qpos[4:7].copy()
    
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
        slider_positions = [self.sim.data.qpos[i] for i in range(4)]
        cable_lengths = self.controller.inverse_kinematics(ee_pos)
        
        self.trajectory_data.append({
            'timestamp': self.sim.data.time,
            'ee_position': ee_pos.copy(),
            'target_position': self.target_pos.copy(),
            'slider_positions': slider_positions.copy(),
            'cable_lengths': cable_lengths.copy(),
            'control_signals': self.sim.data.ctrl.copy() if len(self.sim.data.ctrl) > 0 else [0, 0, 0, 0]
        })
    
    def run_simulation_step(self):
        """Run one simulation step"""
        ee_pos = self.get_end_effector_position()
        
        # Compute and apply control
        control_signals = self.controller.compute_control(self.target_pos, ee_pos)
        for j in range(min(4, len(self.sim.data.ctrl))):
            self.sim.data.ctrl[j] = control_signals[j]
        
        # Step simulation
        self.sim.step()
        
        # Record trajectory data
        self.record_trajectory_step()
    
    def run_trajectory(self, target_positions, trajectory_name="trajectory", max_steps_per_target=600):
        """Run a complete trajectory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_dir = os.path.join(self.output_dir, f"{trajectory_name}_{timestamp}")
        os.makedirs(trajectory_dir, exist_ok=True)
        
        print(f"Starting trajectory: {trajectory_name}")
        self.trajectory_data = []
        
        for i, target_pos in enumerate(target_positions):
            print(f"Moving to target {i+1}/{len(target_positions)}: {target_pos}")
            
            if not self.set_target_position(target_pos):
                print(f"Invalid target position: {target_pos}")
                continue
            
            steps_for_target = 0
            while steps_for_target < max_steps_per_target:
                self.run_simulation_step()
                steps_for_target += 1
                
                if self.check_success():
                    print(f"Target {i+1} reached in {steps_for_target} steps")
                    break
            else:
                print(f"Timeout reaching target {i+1}")
        
        self.save_results(trajectory_dir, trajectory_name)
        return True
    
    def save_results(self, trajectory_dir, trajectory_name):
        """Save trajectory data"""
        trajectory_file = os.path.join(trajectory_dir, "trajectory_data.npz")
        
        trajectory_dict = {}
        if self.trajectory_data:
            arrays_to_save = ['timestamp', 'ee_position', 'target_position', 
                             'slider_positions', 'cable_lengths', 'control_signals']
            
            for key in arrays_to_save:
                if key in self.trajectory_data[0]:
                    trajectory_dict[key] = np.array([data[key] for data in self.trajectory_data])
        
        np.savez(trajectory_file, **trajectory_dict)
        print(f"✓ Trajectory data saved: {trajectory_file}")

def main():
    xml_path = "mujoco/cdpr.xml"
    
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return
    
    if not MUJOCO_PY_AVAILABLE:
        print("mujoco-py is not available. Please install it with: pip install mujoco-py")
        return
    
    sim = CDPRSimulationMuJoCoPy(xml_path)
    
    if sim.initialize():
        trajectories = {
            "simple_test": [
                [0.3, 0.3, 1.2],
                [-0.3, -0.3, 1.2],
                [0.0, 0.0, 1.309]
            ]
        }
        
        for traj_name, waypoints in trajectories.items():
            print(f"\nRunning trajectory: {traj_name}")
            sim.run_trajectory(waypoints, traj_name)

if __name__ == "__main__":
    main()