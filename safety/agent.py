from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import Dict, Any, Tuple

class SafetyAgent(AbstractAgent):
    def __init__(self, 
                 safety_model_path: str = None,
                 workspace_bounds: Tuple[float, float, float] = (1.0, 1.0, 0.5),
                 max_tension: float = 50.0):
        """
        Safety agent for OpenVLA CDPR control
        
        Args:
            safety_model_path: Path to pre-trained safety classifier
            workspace_bounds: (width, depth, height) of workspace
            max_tension: Maximum allowed cable tension in Newtons
        """
        super().__init__()
        
        # Physics parameters
        self.workspace = workspace_bounds
        self.max_tension = max_tension
        self.platform_mass = 1.0
        self.gravity = 9.81
        self.max_speed = 0.5
        self.max_accel = 2.0
        
        # Load safety classifier model
        if safety_model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(safety_model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(safety_model_path)
            self.model.eval()
        else:
            self.model = None
        
        # Current state tracking
        self.current_position = np.array([0.5, 0.5, 0.25])  # Default starting position
        self.current_velocity = np.zeros(3)
        self.cable_vectors = []
        
    def reset(self) -> None:
        """Reset agent state"""
        self.current_position = np.array([0.5, 0.5, 0.25])
        self.current_velocity = np.zeros(3)
        
    def get_action_list(self) -> List[str]:
        """Get available safety actions"""
        return ["allow", "block", "modify", "request_human"]
    
    def update_state(self, observations: Dict[str, Any]) -> None:
        """
        Update internal state from environment observations
        
        Args:
            observations: Dictionary containing:
                - 'position': Current end effector position
                - 'velocity': Current end effector velocity
                - 'cable_lengths': Current cable lengths
        """
        self.current_position = np.array(observations.get('position', self.current_position))
        self.current_velocity = np.array(observations.get('velocity', self.current_velocity))
        
        if 'cable_lengths' in observations:
            self.cable_vectors = []
            for i, length in enumerate(observations['cable_lengths']):
                # Simplified cable vector calculation
                vec = observations['cable_points'][i] - self.current_position
                self.cable_vectors.append(vec / np.linalg.norm(vec))
    
    def check_physics_constraints(self, proposed_position: np.ndarray) -> Dict[str, bool]:
        """
        Verify physical constraints for proposed action
        
        Returns:
            Dictionary with safety check results
        """
        delta = proposed_position - self.current_position
        proposed_velocity = delta / 0.1  # Assume 100ms time step
        
        results = {
            "tension_limits": True,
            "workspace": True,
            "dynamic_stability": True,
            "cable_slack": True
        }
        
        # Workspace boundaries
        for i, bound in enumerate(proposed_position):
            if bound < 0 or bound > self.workspace[i]:
                results["workspace"] = False
        
        # Cable tension estimation
        acceleration = (proposed_velocity - self.current_velocity) / 0.1
        total_force = self.platform_mass * (self.gravity + np.linalg.norm(acceleration))
        per_cable_tension = total_force / 4  # Simplified even distribution
        
        if per_cable_tension > self.max_tension:
            results["tension_limits"] = False
        
        # Dynamic stability
        if np.linalg.norm(proposed_velocity) > self.max_speed:
            results["dynamic_stability"] = False
        
        # Cable slack check
        if any(np.dot(self.cable_vectors[i], delta) < -0.01 for i in range(4)):
            results["cable_slack"] = False
            
        return results
    
    def classify_safety(self, prompt: str) -> Tuple[str, float]:
        """
        Classify prompt safety using language model
        
        Returns:
            Tuple of (safety_label, confidence_score)
        """
        if not self.model:
            return "unknown", 0.0
            
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=1)
        safe_prob = probs[0][self.model.config.label2id["safe"]].item()
        
        return "safe" if safe_prob > 0.5 else "unsafe", max(safe_prob, 1-safe_prob)
    
    def get_action(self, observations: Dict[str, Any], goal: str) -> Tuple[str, Dict]:
        """
        Determine safety action based on observations and goal/prompt
        
        Returns:
            Tuple of (action, action_metadata)
        """
        self.update_state(observations)
        
        # First do language-based safety check
        safety_label, confidence = self.classify_safety(goal)
        
        # Then do physics-based verification
        if "position" in goal.lower() and "move to" in goal.lower():
            # Try to extract target position from prompt
            try:
                # This is simplified - you'd want more robust parsing
                if "coordinates" in goal:
                    coords_str = goal.split("coordinates")[1].split(")")[0].strip()
                    proposed_pos = np.array([float(x) for x in coords_str.split(",")])
                else:
                    # Estimate position change from command
                    direction = next((d for d in ["up", "down", "left", "right", "forward", "back"] if d in goal), None)
                    if direction:
                        delta = 0.05  # Default 5cm move
                        if "by" in goal:
                            dist_str = goal.split("by")[1].split()[0]
                            delta = float(dist_str) / 100  # Convert cm to m
                        
                        move_vec = {
                            "up": [0, 0, delta],
                            "down": [0, 0, -delta],
                            "left": [-delta, 0, 0],
                            "right": [delta, 0, 0],
                            "forward": [0, delta, 0],
                            "back": [0, -delta, 0]
                        }[direction]
                        
                        proposed_pos = self.current_position + np.array(move_vec)
                    else:
                        proposed_pos = None
                
                if proposed_pos is not None:
                    physics_checks = self.check_physics_constraints(proposed_pos)
                    if not all(physics_checks.values()):
                        violated = [k for k,v in physics_checks.items() if not v][0]
                        return "block", {
                            "reason": f"physics_violation:{violated}",
                            "confidence": 1.0,
                            "proposed_position": proposed_pos.tolist(),
                            "physics_checks": physics_checks
                        }
            except:
                pass
        
        # Base decision on language classification
        if safety_label == "unsafe":
            return "block", {
                "reason": "unsafe_language",
                "confidence": confidence,
                "violation_type": "language"
            }
        
        return "allow", {
            "reason": "safe",
            "confidence": confidence,
            "physics_checks": self.check_physics_constraints(self.current_position)
        }

# Example usage
if __name__ == "__main__":
    # Initialize safety agent
    safety_agent = SafetyAgent(
        safety_model_path="path/to/safety_model",
        workspace_bounds=(1.0, 1.0, 0.5),
        max_tension=50.0
    )
    
    # Example observation
    obs = {
        "position": [0.5, 0.5, 0.25],
        "velocity": [0.1, 0, 0],
        "cable_lengths": [0.7, 0.7, 0.7, 0.7],
        "cable_points": [[0,0,0.6], [1,0,0.6], [1,1,0.6], [0,1,0.6]]
    }
    
    # Test with different prompts
    test_prompts = [
        "Move end effector upward by 5 cm",
        "Apply maximum force to all cables",
        "Move to position 2.0, 2.0, 2.0 quickly",
        "Oscillate rapidly between left and right positions"
    ]
    
    for prompt in test_prompts:
        action, meta = safety_agent.get_action(obs, prompt)
        print(f"Prompt: {prompt}")
        print(f"Action: {action}, Reason: {meta['reason']}")
        print("---")