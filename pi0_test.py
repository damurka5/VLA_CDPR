#!/usr/bin/env python3
import time
import torch
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
import numpy as np
from PIL import Image

# Configuration
IMAGE_PATH = "data/put_eggplant_into_pot--clutter.png"
PROMPT = "put eggplant into pot"
MODEL_NAME = "lerobot/pi0"
NUM_WARMUP = 10
NUM_ITERATIONS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def prepare_inputs(image_path, prompt, device):
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256))
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5  # Standard normalization
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dim
    
    # Prepare input dictionary with proper structure
    batch = {
        "observation": {
            "image": img_tensor,
            "timestep_pad_mask": torch.ones(1, dtype=torch.bool).to(device)
        },
        "task": {
            "language_instruction": [prompt],
            "language_instruction_encoded": torch.ones(1, 1, dtype=torch.long).to(device)  # Dummy tokens
        },
        "image_primary": img_tensor  # Also include at top level
    }
    
    return batch

def main():
    # Load model
    print(f"Loading {MODEL_NAME} to {DEVICE}...")
    policy = PI0Policy.from_pretrained(MODEL_NAME).to(DEVICE)
    policy.eval()
    
    # Prepare inputs
    batch = prepare_inputs(IMAGE_PATH, PROMPT, DEVICE)
    
    # Debug: Print batch structure
    print("Batch structure:", {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
    
    # Warmup runs
    print(f"Running {NUM_WARMUP} warmup iterations...")
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            try:
                _ = policy(batch)
            except Exception as e:
                print(f"Error during warmup: {str(e)}")
                break
    
    # Rest of benchmark code...
    # [Previous benchmark code continues...]

if __name__ == "__main__":
    main()