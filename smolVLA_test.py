import torch
import time
from PIL import Image
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.normalize import Normalize, NormalizationMode
from lerobot.common.policies.types import PolicyFeature
from transformers import AutoProcessor

# Configuration
CHECKPOINT_PATH = "/root/repo/smolvla_base"
IMAGE_PATH = "data/put_eggplant_into_pot--clutter.png"
PROMPT = "put eggplant into pot"

# Load model
policy = SmolVLAPolicy.from_pretrained(CHECKPOINT_PATH).to("cuda")
policy.eval()

# Initialize normalization if it's missing
# if not hasattr(policy, 'normalize_inputs'):
    # Get state_dim from the model config or observation
state_dim = policy.config.state_dim if hasattr(policy.config, 'state_dim') else 10  # default fallback

# Create proper PolicyFeature objects
features = {
    'observation.images.top': PolicyFeature(shape=[3, 512, 512], type="image"),
    'observation.state': PolicyFeature(shape=[state_dim], type="state")
}

# Create normalization mapping
norm_map = {
    'observation.images.top': NormalizationMode.MEAN_STD,
    'observation.state': NormalizationMode.MEAN_STD
}

# Create stats dictionary
stats = {
    'observation.images.top': {
        'mean': torch.zeros(3, 1, 1),  # Using 1x1 for height/width to be invariant
        'std': torch.ones(3, 1, 1)
    },
    'observation.state': {
        'mean': torch.zeros(state_dim),
        'std': torch.ones(state_dim)
    }
}

policy.normalize_inputs = Normalize(features, norm_map, stats).to("cuda")

# patch: The loaded policy is missing the language_tokenizer attribute.
policy.language_tokenizer = AutoProcessor.from_pretrained(policy.config.vlm_model_name).tokenizer

# Load and prepare image
image = Image.open(IMAGE_PATH).convert("RGB")

# Create batch
dummy_batch = {
    "observation.images.top": torch.rand(1, 3, 512, 512, device="cuda"),  # placeholder
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
print("First few action values:", actions[0, :5].cpu().numpy())