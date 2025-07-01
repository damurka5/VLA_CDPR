#!/usr/bin/env python3
import time
import torch
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
import numpy as np
from PIL import Image
import inspect

# Configuration
IMAGE_PATH = "data/put_eggplant_into_pot--clutter.png"
PROMPT = "put eggplant into pot"
MODEL_NAME = "lerobot/pi0"
NUM_WARMUP = 10
NUM_ITERATIONS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def prepare_inputs(image_path, prompt, device, policy):
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256))
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5  # Standard normalization
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dim

    # Inspect model's prepare_images method to understand expected format
    prepare_images_src = inspect.getsource(policy.model.prepare_images)
    print("\nModel's prepare_images method expects:")
    print(prepare_images_src)

    # Based on common patterns, try this comprehensive input format
    batch = {
        "observation": {
            "image_primary": img_tensor,
            "image_wrist": img_tensor,  # Some models expect multiple views
            "timestep_pad_mask": torch.ones(1, dtype=torch.bool).to(device)
        },
        "task": {
            "language_instruction": [prompt],
            "language_instruction_encoded": torch.ones(1, 1, dtype=torch.long).to(device)
        },
        "image": img_tensor,  # Some models expect direct image key
        "image_primary": img_tensor,
        "pad_mask": torch.ones(1, dtype=torch.bool).to(device)
    }

    return batch

def main():
    # Load model
    print(f"Loading {MODEL_NAME} to {DEVICE}...")
    policy = PI0Policy.from_pretrained(MODEL_NAME).to(DEVICE)
    policy.eval()
    
    # Prepare inputs with model reference for inspection
    batch = prepare_inputs(IMAGE_PATH, PROMPT, DEVICE, policy)
    
    # Debug: Print simplified batch structure
    print("\nPrepared batch structure:")
    for k, v in batch.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sk, sv in v.items():
                print(f"  {sk}: {type(sv)} {sv.shape if hasattr(sv, 'shape') else ''}")
        else:
            print(f"{k}: {type(v)} {v.shape if hasattr(v, 'shape') else ''}")

    # Warmup runs with detailed error handling
    print(f"\nRunning {NUM_WARMUP} warmup iterations...")
    with torch.no_grad():
        for i in range(NUM_WARMUP):
            try:
                output = policy(batch)
                print(f"Warmup {i+1} successful!")
                break  # Exit if successful
            except Exception as e:
                print(f"Warmup {i+1} failed: {str(e)}")
                if i == NUM_WARMUP - 1:
                    print("\nAll warmup runs failed. Please check:")
                    print("1. The model's expected input format")
                    print("2. Image normalization parameters")
                    print("3. Required keys in the input batch")
                    return

    # Rest of benchmark code...
    # [Previous benchmark code continues...]

if __name__ == "__main__":
    main()