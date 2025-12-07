#!/usr/bin/env python3
# test_adapter_simple.py
import sys
sys.path.append("/root/repo/openvla-oft")

import torch
from safetensors.torch import load_file
from prismatic.vla import VLA

def test_adapter_load():
    adapter_path = "/root/oft_cdpr_ckpts/cdpr_finetune_20251203-133649/vla_cdpr_adapter"
    action_head_path = "/root/oft_cdpr_ckpts/cdpr_finetune_20251203-133649/action_head_cdpr.pt"
    
    print("Testing adapter loading...")
    print(f"Adapter path: {adapter_path}")
    print(f"Action head path: {action_head_path}")
    
    # List adapter directory
    if os.path.isdir(adapter_path):
        print(f"\nAdapter directory contents:")
        for f in os.listdir(adapter_path):
            print(f"  - {f}")
    
    # Try to load base model
    print("\nLoading base model...")
    try:
        vla = VLA.from_pretrained(
            model_id="moojink/openvla-7b-oft-finetuned-libero-spatial",
            use_l1_regression=True,
            use_proprio=True,
            num_images_in_input=2,
        )
        print("✅ Base model loaded")
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        return
    
    # Try to load action head
    print("\nLoading action head...")
    try:
        if os.path.exists(action_head_path):
            action_head = torch.load(action_head_path, map_location="cpu")
            print(f"✅ Action head loaded: {type(action_head)}")
            if isinstance(action_head, torch.nn.Module):
                print(f"   - Parameters: {sum(p.numel() for p in action_head.parameters()):,}")
        else:
            print(f"❌ Action head not found at {action_head_path}")
    except Exception as e:
        print(f"❌ Error loading action head: {e}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    import os
    test_adapter_load()