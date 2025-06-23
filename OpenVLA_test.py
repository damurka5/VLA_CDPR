from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import time
import torch

# Check if MPS (Apple Metal) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_from_camera():
    """Example using local image file - replace with your camera capture"""
    return Image.open("data/put_eggplant_into_pot--clutter.png")  # replace with your image path

class RobotController:
    def act(self, action, **kwargs):
        print(f"Executing action: {action}")
        pass

# Load Processor & VLA with modified settings for M1
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

try:
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,  # bfloat16 not fully supported on MPS
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    print("You may need to reduce model size or use CPU-only mode")
    raise

# Example usage
image = get_from_camera()
instruction = "put eggplant into pot"
prompt = f"In: What action should the robot take to {instruction}?\nOut:"

try:
    # Warm-up run (first inference is always slower)
    print("Running warm-up...")
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    _ = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    # Actual timed inference
    print("Running timed inference...")
    start_time = time.time()
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"Action: {action}")
except Exception as e:
    print(f"Error during inference: {e}")