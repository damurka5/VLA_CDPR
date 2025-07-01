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
    img = img.resize((256, 256))  # PI0 expects 256x256
    
    # Convert to tensor and normalize (example values - adjust per PI0's preprocessing)
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dim
    
    # Prepare proper input dictionary with image features
    batch = {
        "observation": {
            "image_primary": img_tensor,  # Changed from 'image' to 'image_primary'
            "timestep_pad_mask": torch.ones(1, dtype=torch.bool).to(device)
        },
        "task": {
            "language_instruction": [prompt],
            "language_instruction_encoded": torch.ones(1, 1, dtype=torch.long).to(device)  # Dummy tokens
        }
    }
    
    return batch

def main():
    # Load model
    print(f"Loading {MODEL_NAME} to {DEVICE}...")
    policy = PI0Policy.from_pretrained(MODEL_NAME).to(DEVICE)
    policy.eval()
    
    # Prepare inputs
    batch = prepare_inputs(IMAGE_PATH, PROMPT, DEVICE)
    
    # Warmup runs
    print(f"Running {NUM_WARMUP} warmup iterations...")
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = policy(batch)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running {NUM_ITERATIONS} benchmark iterations...")
    timings = []
    for _ in range(NUM_ITERATIONS):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = policy(batch)
        
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
    
    # Stats
    timings = np.array(timings)
    avg_time = np.mean(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)
    std_dev = np.std(timings)
    fps = 1 / avg_time
    
    print("\nBenchmark Results:")
    print(f"- Average inference time: {avg_time:.4f} ± {std_dev:.4f} s")
    print(f"- Best case: {min_time:.4f} s")
    print(f"- Worst case: {max_time:.4f} s")
    print(f"- Throughput: {fps:.2f} FPS")
    
    if DEVICE == "cuda":
        print(f"- Max GPU memory used: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")

if __name__ == "__main__":
    main()