from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import time
import torch
import llava

# Configuration matching eval_real_franka.py
action_head = 'droid_diffusion'  # or your preferred action head
policy_config = {
    "model_path": "lesjie/Llava-Pythia-400M",  # or path to your trained LoRA weights
    "model_base": "lesjie/Llava-Pythia-400M",  # base model path
    "enable_lora": False,  # set to True if using LoRA weights
    "conv_mode": "pythia",
    "action_head": action_head,
}

# Device setup
device = 'mps'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from transformers import AutoConfig, AutoModelForCausalLM
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

model = LlavaLlamaForCausalLM.from_pretrained("liuhaotian/llava-llama-2-13b-chat-lightning-preview")


class llava_pythia_act_policy:
    def __init__(self, policy_config):
        self.processor = AutoProcessor.from_pretrained(
            policy_config["model_base"],
            trust_remote_code=True
        )
        
        self.model = model.from_pretrained(  # Use specific model class
            policy_config["model_base"],
            torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)
        
        self.action_head = policy_config["action_head"]
        self.conv_mode = policy_config["conv_mode"]
        
    def predict_action(self, image, instruction):
        prompt = f"USER: <image>\nWhat action should the robot take to {instruction}?\nASSISTANT:"
        inputs = self.processor(
            text=prompt, 
            images=image, 
            return_tensors="pt"
        ).to(device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return generated_text.split("ASSISTANT:")[-1].strip()

def eval_bc(policy, env, config, save_episode=False, num_rollouts=1, raw_lang=""):
    """Evaluation function with timing"""
    image = Image.open("data/put_eggplant_into_pot--clutter.png")  # replace with your image
    
    # Warm-up run
    print("Running warm-up...")
    _ = policy.predict_action(image, raw_lang)
    
    # Timed inference
    print("Running timed inference...")
    start_time = time.time()
    action = policy.predict_action(image, raw_lang)
    end_time = time.time()
    
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"Action: {action}")

# Initialize policy
policy = llava_pythia_act_policy(policy_config)

# Mock environment (replace with your actual robot env)
class MockEnv:
    def __init__(self):
        pass

# Run evaluation
raw_lang = 'put the tennis ball on the right side into the tennis bucket'
eval_bc(policy, MockEnv(), policy_config, raw_lang=raw_lang)
# import transformers

# print(transformers.__version__)