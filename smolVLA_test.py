import torch
import time
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.normalize import Normalize
from transformers import AutoProcessor
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace with your actual checkpoint path
CHECKPOINT_PATH = "~/repo/smolvla_base"

# --- Image Loading ---
def get_from_camera():
    """Load real image (replace with camera capture)"""
    return Image.open("data/put_eggplant_into_pot--clutter.png")  # Your image path

# --- Model Setup ---
policy = SmolVLAPolicy.from_pretrained(CHECKPOINT_PATH).to(device)
policy.eval()
policy.language_tokenizer = AutoProcessor.from_pretrained(policy.config.vlm_model_name).tokenizer

state_dim = policy.model.config.max_state_dim

# Create valid normalization stats
norm_stats = {
    "observation.images.top": {
        "mean": torch.tensor([0.0, 0.0, 0.0]).cuda(),
        "std": torch.tensor([1.0, 1.0, 1.0]).cuda(),
    },
    "observation.state": {
        "mean": torch.zeros(state_dim).cuda(),
        "std": torch.ones(state_dim).cuda(),
    }
}
policy.normalize_inputs = Normalize(stats=norm_stats)

# --- Prepare REAL inputs ---
def load_image():
    return Image.open("data/put_eggplant_into_pot--clutter.png").convert("RGB")

real_batch = {
    "observation.images.top": [load_image()],  # List of PIL images
    "observation.state": [np.zeros(state_dim)],  # Dummy state
    "task": ["put eggplant into pot"]  # Your instruction
}

# --- Process inputs ---
normalized_batch = policy.normalize_inputs(real_batch)
images, img_masks = policy.prepare_images(normalized_batch)
state = policy.prepare_state(normalized_batch)
lang_tokens, lang_masks = policy.prepare_language(normalized_batch)

# Warmup
for _ in range(3):
    with torch.no_grad():
        _ = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)

# Benchmark
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
torch.cuda.synchronize()
end = time.time()

print(f"Avg inference time: {(end - start)/100:.6f} s")
print(f"Max GPU memory used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")