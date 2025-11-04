import mujoco as mj
import numpy as np
import time

class CDPR4:
    def __init__(self, approx=1, pos=np.array([0, 0, 0])):
        self.approx = approx
        self.pos = pos  # Current end-effector position in world frame
        self.frame_points = np.array([
            [-0.535, -0.755, 1.309],
            [0.755, -0.525, 1.309],
            [0.535, 0.755, 1.309],
            [-0.755, 0.525, 1.309]
        ])
        self.ee_points = np.array([
            [-0.05, -0.05, 0.05],
            [0.05, -0.05, 0.05],
            [0.05, 0.05, 0.05],
            [-0.05, 0.05, 0.05]
        ])
        # Store previous cable lengths for derivative term
        self.prev_lengths = np.zeros(4)
    
    def inverse_kinematics(self, pos=None):
        if pos is None:
            pos = self.pos
            
        lengths = []
        for i in range(4):
            vec = pos - self.frame_points[i]
            lengths.append(np.linalg.norm(vec))
        return tuple(lengths)

def controller(model, data):
    # Empty controller - we'll handle control in the main loop
    pass

# Load model
xml_path = 'cdpr.xml'  # Adjust path as needed
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# Initialize controller
mj.set_mjcb_control(controller)

# Initialize robot controller with correct initial position
initial_pos = np.array([0, 0, 1])
robot = CDPR4(approx=1, pos=initial_pos)

# Simulation parameters
simend = 7
dt = 1.0/60.0  # Simulation time step
model.opt.timestep = dt
threshold = 0.03
received_goal = False
target_pos = np.array([0, 0, 1.309])  # Initialize target position

# PD controller parameters
Kp = 100  # Proportional gain
Kd = 130   # Derivative gain

print("CDPR Simulation Started (Completely Headless Mode)")
print("Press Ctrl+C to stop the simulation")

# Pre-defined target positions for automated testing
target_sequence = [
    np.array([0.5, 0.5, 0.5]),
    # np.array([-0.3, 0.3, 1.2]),
    # np.array([0.2, -0.2, 0.8]),
    np.array([0, 0, 0.1])
]
current_target_index = 0

try:
    while data.time < simend:
        # Get current end-effector position from simulation and update robot's position
        Ac = data.qpos[4:7]  # Current end-effector position in world frame
        robot.pos = Ac.copy()  # Update the robot's current position
        
        # Automated target sequencing (remove this if you want manual input)
        if not received_goal and current_target_index < len(target_sequence):
            target_pos = target_sequence[current_target_index]
            received_goal = True
            # Reset previous lengths when setting a new goal
            cur_L_1, cur_L_2, cur_L_3, cur_L_4 = robot.inverse_kinematics()
            robot.prev_lengths = np.array([cur_L_1, cur_L_2, cur_L_3, cur_L_4])
            print(f"Target {current_target_index + 1} set to: {target_pos}")
        
        # Get current cable lengths (using robot's current position)
        cur_L_1, cur_L_2, cur_L_3, cur_L_4 = robot.inverse_kinematics()
        cur_lengths = np.array([cur_L_1, cur_L_2, cur_L_3, cur_L_4])
        
        # Get target cable lengths (using target position)
        target_L_1, target_L_2, target_L_3, target_L_4 = robot.inverse_kinematics(target_pos)
        target_lengths = np.array([target_L_1, target_L_2, target_L_3, target_L_4])
        
        # PD Control logic
        if received_goal:
            # Calculate cable length errors
            length_errors = target_lengths - cur_lengths
            
            # Convert cable length changes to slider position changes
            slider_positions = 0.9 - target_lengths

            for i in range(4):
                data.ctrl[i] = slider_positions[i]
            
            # Print status every 0.5 seconds
            if int(data.time * 2) % 2 == 0:  # Print every 0.5 simulation seconds
                print(f'Time: {data.time:.2f}s, Position: {Ac}, Errors: {length_errors} dt {model.opt.timestep}')
            
            # Check if we've reached the goal
            if np.all(np.linalg.norm(Ac-target_pos) < threshold):
                print(f"Reached target position {current_target_index + 1}!")
                received_goal = False
                current_target_index += 1
                # break
                
                # Stop if all targets are reached
                if current_target_index >= len(target_sequence):
                    print("All target positions reached! Simulation complete.")
                    break
        
        # Step simulation
        mj.mj_step(model, data)
        
        # Optional: Add small delay to prevent excessive CPU usage
        # time.sleep(0.001)
        
except KeyboardInterrupt:
    print("\nSimulation stopped by user")

finally:
    print("Simulation finished")
    print(f"Final end-effector position: {data.qpos[4:7]}")
    print(f"Total simulation time: {data.time:.2f} seconds")