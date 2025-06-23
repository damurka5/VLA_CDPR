import torch
import time
from PIL import Image
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from transformers import AutoProcessor

# Configuration
CHECKPOINT_PATH = "/root/repo/smolvla_base"
IMAGE_PATH = "data/put_eggplant_into_pot--clutter.png"
PROMPT = "put eggplant into pot"

# Load model
policy = SmolVLAPolicy.from_pretrained(CHECKPOINT_PATH).to("cuda")
policy.eval()

# patch: The loaded policy is missing the language_tokenizer attribute.
policy.language_tokenizer = AutoProcessor.from_pretrained(policy.config.vlm_model_name).tokenizer

# Load and prepare image
image = Image.open(IMAGE_PATH).convert("RGB")

# Infer state_dim from the loaded normalization stats
state_dim = policy.normalize_inputs.buffer_observation_state.mean.shape[-1]

# Create batch
dummy_batch = {
    "observation.images.top": torch.rand(1, 3, 512, 512, device="cuda"),  # placeholder, will be replaced
    "observation.state": torch.rand(1, state_dim, device="cuda"),
    "task": [PROMPT],
}

# Preprocess image
processor = AutoProcessor.from_pretrained(policy.config.vlm_model_name)
processed_image = processor(images=image, return_tensors="pt")["pixel_values"].to("cuda")
dummy_batch["observation.images.top"] = processed_image

# --- Prepare inputs for the model ---
normalized_batch = policy.normalize_inputs(dummy_batch)
images, img_masks = policy.prepare_images(normalized_batch)
state = policy.prepare_state(normalized_batch)
lang_tokens, lang_masks = policy.prepare_language(normalized_batch)
# ---

# Run inference
with torch.no_grad():
    actions = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)

print("Inference completed!")
print("Actions shape:", actions.shape)
print("First few action values:", actions[0, :5].cpu().numpy())  # Print first few action values

# Benchmark
torch.cuda.reset_peak_memory_stats()
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
end = time.time()

print(f"\nBenchmark results:")
print(f"Avg inference time: {(end - start)/100:.6f} s")
print(f"Max GPU memory used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")