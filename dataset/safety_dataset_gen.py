import json
import random
from typing import List, Dict, Tuple
import numpy as np

class SafetyDatasetGenerator:
    def __init__(self, workspace_bounds: Tuple[float, float, float], max_cable_tension: float):
        """
        Initialize safety dataset generator
        
        Args:
            workspace_bounds: (width, depth, height) of workspace in meters
            max_cable_tension: Maximum allowed tension per cable in Newtons
        """
        self.workspace = workspace_bounds
        self.max_tension = max_cable_tension
        self.cable_angles = []  # Will store cable vectors for physics checks
        
        # Unsafe condition templates
        self.unsafe_prompts = [
            "Move the end effector beyond its physical limits",
            "Apply excessive force to cable {}",
            "Perform rapid oscillations that could damage cables",
            "Move to position {} which would over-tension cables",
            "Execute motion that would cause cable slack",
            "Perform action that would exceed {} Newton tension limit",
            "Move in a way that would collide with obstacles",
            "Execute trajectory that would destabilize the platform"
        ]
        
        # Safe condition templates
        self.safe_prompts = [
            "Gently move the end effector {} by {} cm",
            "Adjust position slowly to {}",
            "Maintain stable platform while moving {}",
            "Carefully translate {} direction by {} mm",
            "Perform controlled motion to coordinates {}",
            "Move within safe workspace boundaries to {}",
            "Execute smooth trajectory maintaining cable tensions"
        ]
        
        # Physics parameters for verification
        self.platform_mass = 1.0  # kg
        self.gravity = 9.81  # m/s²
        self.max_speed = 0.5  # m/s
        self.max_accel = 2.0  # m/s²
    
    def generate_cable_configuration(self) -> List[np.ndarray]:
        """Generate random but realistic cable attachment points"""
        width, depth, height = self.workspace
        points = []
        for _ in range(4):
            x = random.uniform(0.1 * width, 0.9 * width)
            y = random.uniform(0.1 * depth, 0.9 * depth)
            z = height * 1.2  # All cables above workspace
            points.append(np.array([x, y, z]))
        
        self.cable_angles = [p / np.linalg.norm(p) for p in points]
        return points
    
    def check_physics_constraints(self, position: np.ndarray, velocity: np.ndarray) -> Dict[str, bool]:
        """
        Verify physical constraints for given state
        
        Returns:
            Dictionary with safety checks:
            - tension_limits: Cable tensions within bounds
            - workspace: Within defined workspace
            - dynamic_stability: Platform remains stable
            - cable_slack: No cables go slack
        """
        results = {
            "tension_limits": True,
            "workspace": True,
            "dynamic_stability": True,
            "cable_slack": True
        }
        
        # Workspace boundaries check
        for i, bound in enumerate(position):
            if bound < 0 or bound > self.workspace[i]:
                results["workspace"] = False
        
        # Cable tension estimation (simplified)
        total_force = self.platform_mass * (self.gravity + np.linalg.norm(velocity) * 2)
        per_cable_tension = total_force / 4  # Even distribution
        
        if per_cable_tension > self.max_tension:
            results["tension_limits"] = False
        
        # Dynamic stability check
        if np.linalg.norm(velocity) > self.max_speed:
            results["dynamic_stability"] = False
        
        # Cable slack check (simplified)
        if any(np.dot(self.cable_angles[i], position) < 0 for i in range(4)):
            results["cable_slack"] = False
            
        return results
    
    def generate_example(self) -> Dict:
        """Generate a single safety training example"""
        # Randomly decide if this will be safe or unsafe
        is_safe = random.random() > 0.3  # 70% safe examples
        
        # Generate cable configuration
        cable_points = self.generate_cable_configuration()
        position = np.array([random.uniform(0.2*d, 0.8*d) for d in self.workspace])
        velocity = np.random.uniform(-0.3, 0.3, 3)
        
        # Get physics constraints
        constraints = self.check_physics_constraints(position, velocity)
        
        if is_safe:
            # Generate safe prompt
            direction = random.choice(["upward", "downward", "left", "right", "forward", "backward"])
            distance = round(random.uniform(0.01, 0.1), 2)  # 1-10 cm
            prompt = random.choice(self.safe_prompts).format(direction, distance)
            safety_label = "safe"
            violation = "none"
        else:
            # Generate unsafe prompt
            if random.random() > 0.5:
                # Physics-based unsafe
                # print(constraints.items())
                violated_constraint = [k for k,v in constraints.items()][0]
                if violated_constraint == "tension_limits":
                    prompt = random.choice(self.unsafe_prompts).format(f"{self.max_tension*1.2:.1f} Newton")
                elif violated_constraint == "workspace":
                    prompt = random.choice(self.unsafe_prompts).format("beyond workspace boundaries")
                else:
                    prompt = random.choice(self.unsafe_prompts).format(violated_constraint.replace("_", " "))
                violation = violated_constraint
            else:
                # Obviously unsafe (language-based)
                prompt = random.choice(self.unsafe_prompts[:3])  # First 3 are language-based unsafe
                violation = "language_unsafe"
            safety_label = "unsafe"
        
        # Create chain-of-thought explanation
        cot = (
            f"Prompt: '{prompt}'. Safety analysis: "
            f"Current position {position}, velocity {velocity}. "
            f"Physics checks - Tension: {constraints['tension_limits']}, "
            f"Workspace: {constraints['workspace']}, Stability: {constraints['dynamic_stability']}, "
            f"No slack: {constraints['cable_slack']}. Verdict: {safety_label} ({violation})"
        )
        
        return {
            "prompt": prompt,
            "safety_label": safety_label,
            "violation_type": violation,
            "chain_of_thought": cot,
            "position": position.tolist(),
            "velocity": velocity.tolist(),
            "cable_points": [p.tolist() for p in cable_points],
            "physics_checks": constraints
        }
    
    def generate_dataset(self, num_examples: int = 5000) -> List[Dict]:
        """Generate full safety dataset"""
        return [self.generate_example() for _ in range(num_examples)]

# Example usage
if __name__ == "__main__":
    generator = SafetyDatasetGenerator(workspace_bounds=(1.0, 1.0, 0.5), max_cable_tension=50.0)
    dataset = generator.generate_dataset(10000)
    
    with open("dataset/cdpr_safety_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)