import numpy as np
import random
import json
from math import sqrt, sin, cos, radians
from typing import List, Dict, Tuple

class CDPRKinematics:
    def __init__(self, cable_points: List[np.ndarray], initial_pos: np.ndarray):
        """
        Initialize CDPR system with cable attachment points in world frame
        
        Args:
            cable_points: List of 4 cable vanishing points (a_i) in world coordinates
            initial_pos: Initial end effector position (c) in world coordinates
        """
        self.cable_points = cable_points
        self.current_pos = initial_pos.copy()
        self.cable_lengths = self.calculate_cable_lengths()
        
    def calculate_cable_lengths(self) -> List[float]:
        """Calculate current cable lengths based on end effector position"""
        return [np.linalg.norm(self.current_pos - a_i) for a_i in self.cable_points]
    
    def move_effector(self, delta: np.ndarray) -> List[float]:
        """
        Move end effector by delta and return required cable length changes
        
        Args:
            delta: 3D displacement vector for end effector
            
        Returns:
            List of cable length changes (Δl_i)
        """
        new_pos = self.current_pos + delta
        new_lengths = [np.linalg.norm(new_pos - a_i) for a_i in self.cable_points]
        delta_lengths = [new - old for new, old in zip(new_lengths, self.cable_lengths)]
        
        # Update state
        self.current_pos = new_pos
        self.cable_lengths = new_lengths
        
        return delta_lengths

class SyntheticDatasetGenerator:
    def __init__(self, workspace_dimensions: Tuple[float, float, float]):
        """
        Initialize dataset generator with workspace dimensions (width, depth, height)
        """
        self.workspace = workspace_dimensions
        self.motion_verbs = ["move", "shift", "translate", "displace", "adjust"]
        self.directions = {
            "upward": np.array([0, 0, 1]),
            "downward": np.array([0, 0, -1]),
            "left": np.array([-1, 0, 0]),
            "right": np.array([1, 0, 0]),
            "forward": np.array([0, 1, 0]),
            "backward": np.array([0, -1, 0]),
            "up-left": np.array([-0.707, 0, 0.707]),
            "up-right": np.array([0.707, 0, 0.707]),
            "diagonal-forward": np.array([0.577, 0.577, 0.577])
        }
        self.units = ["cm", "mm", "meters"]
        
    def generate_cable_points(self) -> List[np.ndarray]:
        """Generate random cable vanishing points above the workspace"""
        width, depth, height = self.workspace
        return [
            np.array([random.uniform(0, width), random.uniform(0, depth), height * 1.2]),
            np.array([random.uniform(0, width), random.uniform(0, depth), height * 1.2]),
            np.array([random.uniform(0, width), random.uniform(0, depth), height * 1.2]),
            np.array([random.uniform(0, width), random.uniform(0, depth), height * 1.2])
        ]
    
    def generate_random_position(self) -> np.ndarray:
        """Generate random position within workspace"""
        width, depth, height = self.workspace
        return np.array([
            random.uniform(0.1 * width, 0.9 * width),
            random.uniform(0.1 * depth, 0.9 * depth),
            random.uniform(0.1 * height, 0.9 * height)
        ])
    
    def generate_motion_command(self) -> Tuple[str, np.ndarray]:
        """Generate a random natural language motion command"""
        verb = random.choice(self.motion_verbs)
        direction = random.choice(list(self.directions.keys()))
        distance = round(random.uniform(0.01, 0.2), 3)  # 1cm to 20cm
        unit = random.choice(self.units)
        
        # Convert to meters for internal use
        scale = 0.01 if unit == "cm" else 0.001 if unit == "mm" else 1.0
        delta = self.directions[direction] * distance * scale
        
        command = f"{verb} end effector {direction} by {distance} {unit}"
        return command, delta
    
    def generate_example(self) -> Dict:
        """Generate a single training example"""
        # Set up CDPR system
        cable_points = self.generate_cable_points()
        initial_pos = self.generate_random_position()
        cdpr = CDPRKinematics(cable_points, initial_pos)
        
        # Generate motion command
        command, delta = self.generate_motion_command()
        
        # Calculate cable adjustments
        delta_lengths = cdpr.move_effector(delta)
        delta_lengths_cm = [round(abs(dl) * 100, 1) for dl in delta_lengths]  # convert to cm
        
        # Create chain-of-thought explanation
        cot = (
            f"To {command}, we need to calculate the required cable length changes. "
            f"The current end effector position is at {initial_pos}. "
            f"The new position will be at {cdpr.current_pos}. "
            f"The cable length changes required are: "
            f"Cable 1: {'pull' if delta_lengths[0] > 0 else 'release'} {delta_lengths_cm[0]} cm, "
            f"Cable 2: {'pull' if delta_lengths[1] > 0 else 'release'} {delta_lengths_cm[1]} cm, "
            f"Cable 3: {'pull' if delta_lengths[2] > 0 else 'release'} {delta_lengths_cm[2]} cm, "
            f"Cable 4: {'pull' if delta_lengths[3] > 0 else 'release'} {delta_lengths_cm[3]} cm."
        )
        
        # Create action string
        actions = []
        for i, dl in enumerate(delta_lengths_cm, 1):
            action = "pull" if delta_lengths[i-1] > 0 else "release"
            actions.append(f"{action} cable {i} by {dl} cm")
        action_str = "; ".join(actions)
        
        return {
            "prompt": command,
            "completion": action_str,
            "chain_of_thought": cot,
            "initial_position": initial_pos.tolist(),
            "delta": delta.tolist(),
            "cable_points": [p.tolist() for p in cable_points],
            "cable_length_changes": delta_lengths_cm
        }
    
    def generate_dataset(self, num_examples: int = 1000) -> List[Dict]:
        """Generate a dataset of training examples"""
        dataset = []
        for _ in range(num_examples):
            try:
                dataset.append(self.generate_example())
            except:
                continue  # skip any problematic examples
        return dataset

if __name__ == "__main__":
    workspace = (1.51, 1.05, 1.309)  # 1.51m x 1.05m x 1.309m workspace
    
    # Create dataset generator
    generator = SyntheticDatasetGenerator(workspace)
    
    # Generate dataset
    dataset = generator.generate_dataset(num_examples=5000)
    
    # Save to JSON file
    with open("dataset/cdpr_motion_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} training examples.")