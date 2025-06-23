import torch
import time
from PIL import Image
from dataclasses import dataclass
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.normalize import Normalize, NormalizationMode
from transformers import AutoProcessor

# Configuration
CHECKPOINT_PATH = "/root/repo/smolvla_base"
IMAGE_PATH = "data/put_eggplant_into_pot--clutter.png"
PROMPT = "put eggplant into pot"

# Simple PolicyFeature replacement
@dataclass
class PolicyFeature:
    shape: list
    type: str

# Load model
policy = SmolVLAPolicy.from_pretrained(CHECKPOINT_PATH).to("cuda")
policy.eval()

# Initialize normalization if it's missing
# if not hasattr(policy, 'normalize_inputs'):
state_dim = policy.config.state_dim if hasattr(policy.config, 'state_dim') else 10

features = {
    'observation.image': PolicyFeature(shape=[3, 256, 256], type="image"),
    'observation.state': PolicyFeature(shape=[state_dim], type="state")
}

norm_map = {
    'observation.image': NormalizationMode.MEAN_STD,
    'observation.state': NormalizationMode.MEAN_STD
}

stats = {
    'observation.image': {
        'mean': torch.zeros(3, 1, 1),
        'std': torch.ones(3, 1, 1)
    },
    'observation.state': {
        'mean': torch.zeros(state_dim),
        'std': torch.ones(state_dim)
    }
}

policy.normalize_inputs = Normalize(features, norm_map, stats).to("cuda")

# Set up processor
processor = AutoProcessor.from_pretrained(policy.config.vlm_model_name)
policy.language_tokenizer = processor.tokenizer

# Load and prepare image
image = Image.open(IMAGE_PATH).convert("RGB")

# Create batch
dummy_batch = {
    "observation.image": torch.rand(1, 3, 256, 256, device="cuda"),  # placeholder
    "observation.state": torch.rand(1, state_dim, device="cuda"),
    "task": [PROMPT],
}

# Process image without explicit size parameter
processed = processor(
    images=image,
    return_tensors="pt",
    padding=True
)
processed_image = processed["pixel_values"].to("cuda")
dummy_batch["observation.image"] = processed_image

# Prepare inputs
normalized_batch = policy.normalize_inputs(dummy_batch)
images, img_masks = policy.prepare_images(normalized_batch)
state = policy.prepare_state(normalized_batch)
lang_tokens, lang_masks = policy.prepare_language(normalized_batch)

# Run inference
with torch.no_grad():
    actions = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)

print("Inference completed!")
print("Actions shape:", actions.shape)
print("First few action values:", actions[0, :5].cpu().numpy())

print("Warming up...")
for _ in range(10):
    with torch.no_grad():
        _ = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
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
        _ = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
    
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

# Print action info (original functionality)
print("\nInference completed!")
print("Actions shape:", actions.shape)
print("First few action values:", actions[0, :5].cpu().numpy())