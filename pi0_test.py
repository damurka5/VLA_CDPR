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
    
    # Basic input structure that should work with PI0FlowMatching
    batch = {
        "image": img_tensor,
        "language_instruction": [prompt],
        "pad_mask": torch.ones(1, dtype=torch.bool).to(device)
    }
    
    return batch

def main():
    # Load model
    print(f"Loading {MODEL_NAME} to {DEVICE}...")
    policy = PI0Policy.from_pretrained(MODEL_NAME).to(DEVICE)
    policy.eval()
    
    # Print model architecture for debugging
    print("\nModel architecture:")
    print(policy.model)
    
    # Prepare inputs
    batch = prepare_inputs(IMAGE_PATH, PROMPT, DEVICE)
    
    # Debug: Print batch structure
    print("\nInput batch structure:")
    for k, v in batch.items():
        print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")

    # Warmup runs
    print(f"\nRunning {NUM_WARMUP} warmup iterations...")
    with torch.no_grad():
        for i in range(NUM_WARMUP):
            try:
                output = policy(batch)
                print(f"Warmup {i+1} successful! Output shape: {output.shape}")
                break
            except Exception as e:
                print(f"Warmup {i+1} failed: {str(e)}")
                if i == NUM_WARMUP - 1:
                    print("\nAll warmup runs failed. Please check:")
                    print("1. The exact input format expected by PI0FlowMatching")
                    print("2. Whether the model expects multiple image views")
                    print("3. If text needs proper tokenization")
                    return

    # Benchmark
    print(f"\nRunning {NUM_ITERATIONS} benchmark iterations...")
    timings = []
    for i in range(NUM_ITERATIONS):
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = policy(batch)
        
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
        
        if (i+1) % 10 == 0:
            print(f"Completed {i+1}/{NUM_ITERATIONS} iterations")

    # Stats calculation and output
    # [Previous stats code continues...]

if __name__ == "__main__":
    main()