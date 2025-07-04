import json
from datasets import Dataset
import numpy as np
import random
from transformers import AutoTokenizer

class CDPRDatasetGenerator:
    def __init__(self, workspace_dimensions=(1.0, 1.0, 0.5)):
        self.workspace = workspace_dimensions
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Motion vocabulary
        self.motion_verbs = ["move", "shift", "translate", "displace", "adjust"]
        self.directions = {
            "upward": [0, 0, 1],
            "downward": [0, 0, -1],
            "left": [-1, 0, 0],
            "right": [1, 0, 0],
            "forward": [0, 1, 0],
            "backward": [0, -1, 0],
            "up-left": [-0.707, 0, 0.707],
            "up-right": [0.707, 0, 0.707],
            "diagonal-forward": [0.577, 0.577, 0.577]
        }
        self.units = ["cm", "mm", "meters"]

    def generate_example(self):
        """Generate a single training example in OpenVLA format"""
        # Generate random CDPR configuration
        cable_points = [np.random.uniform(0, dim, 3) for dim in self.workspace]
        initial_pos = np.array([random.uniform(0.1*d, 0.9*d) for d in self.workspace])
        
        # Generate motion command
        verb = random.choice(self.motion_verbs)
        direction = random.choice(list(self.directions.keys()))
        distance = round(random.uniform(0.01, 0.2), 3)
        unit = random.choice(self.units)
        
        # Create prompt
        prompt = (f"<s>[INST] <<SYS>>\n"
                 f"You are a control system for a cable-driven parallel robot. "
                 f"Translate motion commands into cable adjustments.\n"
                 f"<</SYS>>\n\n"
                 f"{verb} end effector {direction} by {distance} {unit} [/INST]")
        
        # Calculate cable adjustments
        delta = np.array(self.directions[direction]) * distance * (0.01 if unit == "cm" else 0.001 if unit == "mm" else 1)
        new_pos = initial_pos + delta
        delta_lengths = [np.linalg.norm(new_pos - a_i) - np.linalg.norm(initial_pos - a_i) for a_i in cable_points]
        
        # Format completion with chain-of-thought
        cot = (
            f"To {verb} the end effector {direction} by {distance} {unit}, calculate position change: "
            f"from {initial_pos} to {new_pos}. Cable length changes: "
        )
        actions = []
        for i, dl in enumerate(delta_lengths, 1):
            action = "pull" if dl > 0 else "release"
            cm = abs(round(dl*100, 1))
            actions.append(f"{action} cable {i} by {cm} cm")
            cot += f"Cable {i}: {cm} cm {'increase' if dl>0 else 'decrease'}. "
        
        completion = f"{cot} Action: {'; '.join(actions)} </s>"
        
        return {
            "text": prompt + completion,
            "prompt": prompt,
            "completion": completion
        }

    def generate_dataset(self, num_samples=5000):
        """Generate dataset and save in HuggingFace format"""
        data = [self.generate_example() for _ in range(num_samples)]
        
        # Convert to HuggingFace Dataset
        hf_dataset = Dataset.from_list(data)
        
        # Split dataset
        dataset = hf_dataset.train_test_split(test_size=0.1)
        
        return dataset

# Generate and save dataset
if __name__ == "__main__":
    generator = CDPRDatasetGenerator()
    dataset = generator.generate_dataset(10000)
    
    # Save to disk
    dataset.save_to_disk("cdpr_openvla_dataset")
    
    # Also save as JSON for inspection
    with open("cdpr_openvla_dataset.json", "w") as f:
        json.dump(dataset["train"][:100], f, indent=2)  # save sample for inspection