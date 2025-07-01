from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from PIL import Image
import time
import torch
import numpy as np

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_from_camera():
    """Example using local image file - replace with your camera capture"""
    return Image.open("data/put_eggplant_into_pot--clutter.png")  # replace with your image path

class RobotController:
    def act(self, action, **kwargs):
        print(f"Executing action: {action}")
        pass

# Load Octo model and processors
print("Loading Octo model...")
model = AutoModel.from_pretrained("rail-berkeley/octo-base", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("rail-berkeley/octo-base", trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained("rail-berkeley/octo-base", trust_remote_code=True)

# Example usage
image = get_from_camera()
instruction = "put eggplant into pot"

# Prepare inputs
images = image_processor(image, return_tensors="pt").pixel_values.to(device)
img_masks = torch.ones(1, 1, dtype=torch.bool).to(device)  # Assuming single image
lang_tokens = tokenizer(instruction, return_tensors="pt").input_ids.to(device)
lang_masks = torch.ones_like(lang_tokens, dtype=torch.bool).to(device)
state = torch.zeros(1, 0, model.config.proprio_dim).to(device)  # Empty state

# Warm up
print("Warming up...")
for _ in range(10):
    with torch.no_grad():
        _ = model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
torch.cuda.synchronize()  # Wait for all kernels to finish

# Benchmark setup
num_iterations = 100
timings = []

# Create CUDA events for precise timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Clear GPU cache
torch.cuda.empty_cache()

# Benchmark loop
print(f"Running benchmark with {num_iterations} iterations...")
for _ in range(num_iterations):
    torch.cuda.synchronize()
    start_event.record()
    
    with torch.no_grad():
        actions = model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
    
    end_event.record()
    torch.cuda.synchronize()
    timings.append(start_event.elapsed_time(end_event))  # milliseconds

# Convert to numpy array for calculations
timings_ms = torch.tensor(timings)
timings_s = timings_ms / 1000  # convert to seconds

# Calculate statistics
avg_time = timings_s.mean().item()
min_time = timings_s.min().item()
max_time = timings_s.max().item()
std_dev = timings_s.std().item()
fps = 1 / avg_time

# Memory usage
max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

print("\nBenchmark Results:")
print(f"- Average inference time: {avg_time:.4f} ± {std_dev:.4f} s")
print(f"- Best case: {min_time:.4f} s")
print(f"- Worst case: {max_time:.4f} s")
print(f"- Throughput: {fps:.2f} FPS")
print(f"- Max GPU memory used: {max_mem:.2f} MB")

# Print the predicted action
print(f"\nPredicted action: {actions}")