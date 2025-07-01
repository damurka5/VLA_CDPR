import jax
import numpy as np
from PIL import Image
import time
from octo.model.octo_model import OctoModel

# Verify JAX installation
print("JAX version:", jax.__version__)
print("Devices:", jax.devices())

def get_image():
    """Load your image as numpy array"""
    img = Image.open("example.jpg")  # replace with your image
    return np.array(img)

# Load model
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")

# Prepare inputs
image = get_image()[None, None, ...]  # Add batch and time dimensions
observation = {
    "image_primary": image,
    "timestep_pad_mask": np.array([[True]])
}
task = model.create_tasks(texts=["put eggplant into pot"])

# Run inference
action = model.sample_actions(
    observation,
    task,
    unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
    rng=jax.random.PRNGKey(0)
)

print("Predicted action:", action)