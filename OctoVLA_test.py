import jax
import numpy as np
from PIL import Image
import time
import torch  # Only for device checking
from octo.model.octo_model import OctoModel
import tqdm  # For progress bar if needed

# Check device (note: Octo runs on CPU/GPU via JAX, not PyTorch)
print("JAX devices:", jax.devices())

def get_from_camera():
    """Example using local image file - replace with your camera capture"""
    img = Image.open("data/put_eggplant_into_pot--clutter.png")  # replace with your image path
    return np.array(img)

# Load Octo model
print("Loading Octo model...")
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

# Example usage
image = get_from_camera()
instruction = "put eggplant into pot"

# Prepare inputs - add batch and time dimensions
img = image[np.newaxis, np.newaxis, ...]  # shape becomes [1, 1, H, W, C]
observation = {
    "image_primary": img,
    "timestep_pad_mask": np.array([[True]])  # Indicates valid timesteps
}
task = model.create_tasks(texts=[instruction])

# Warm up
print("Warming up...")
for _ in range(10):
    action = model.sample_actions(
        observation, 
        task, 
        unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"], 
        rng=jax.random.PRNGKey(0)
    )

# Benchmark setup
num_iterations = 100
timings = []

# Benchmark loop
print(f"Running benchmark with {num_iterations} iterations...")
for _ in range(num_iterations):
    start_time = time.time()
    
    action = model.sample_actions(
        observation, 
        task, 
        unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"], 
        rng=jax.random.PRNGKey(0)
    )
    
    end_time = time.time()
    timings.append(end_time - start_time)

# Calculate statistics
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

# Print the predicted action (unnormalized)
print(f"\nPredicted action: {action}")