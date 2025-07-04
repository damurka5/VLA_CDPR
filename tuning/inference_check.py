from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
import torch

def load_finetuned_model():
    model_path = "./openvla_cdpr_finetuned/final_model"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    return model, tokenizer

def generate_response(prompt, model, tokenizer):
    # Format prompt for OpenVLA/Llama2 chat
    formatted_prompt = (
        f"<s>[INST] <<SYS>>\n"
        f"You are a control system for a cable-driven parallel robot. "
        f"Translate motion commands into cable adjustments.\n"
        f"<</SYS>>\n\n"
        f"{prompt} [/INST]"
    )
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True
    )
    
    # Generate response
    result = pipe(formatted_prompt)
    return result[0]['generated_text']

# Example usage
if __name__ == "__main__":
    model, tokenizer = load_finetuned_model()
    
    test_prompts = [
        "move end effector upward by 5 cm",
        "shift the platform left by 3 cm",
        "adjust position diagonally-forward by 10 cm"
    ]
    
    for prompt in test_prompts:
        print(f"Prompt: {prompt}")
        response = generate_response(prompt, model, tokenizer)
        print(f"Response: {response}\n")