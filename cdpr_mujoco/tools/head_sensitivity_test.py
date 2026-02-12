#!/usr/bin/env python3
"""
Test action head sensitivity on random inputs.

Usage:
  python tools/head_sensitivity_test.py \
    --action-head-path /root/repo/VLA_CDPR/.../action_head_cdpr.pt \
    --base-ckpt moojink/openvla-7b-oft-finetuned-libero-spatial
"""

import argparse
import numpy as np
import torch
import sys

OPENVLA_OFT_ROOT = "/root/repo/openvla-oft"
if OPENVLA_OFT_ROOT not in sys.path:
    sys.path.append(OPENVLA_OFT_ROOT)

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_vla
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-head-path", required=True)
    ap.add_argument("--base-ckpt", default="moojink/openvla-7b-oft-finetuned-libero-spatial")
    args = ap.parse_args()

    cfg = GenerateConfig(
        pretrained_checkpoint=args.base_ckpt,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=None,
    )

    vla = get_vla(cfg).eval()
    llm_dim = vla.llm_dim
    print("llm_dim =", llm_dim)

    head = get_action_head(cfg, llm_dim=llm_dim)
    head_mod = getattr(head, "model", head)

    ckpt = torch.load(args.action_head_path, map_location="cpu")
    head_keys = list(head_mod.state_dict().keys())
    ckpt_keys = list(ckpt.keys())

    def strip_model_prefix(sd):
        out = {}
        for k, v in sd.items():
            out[k[6:]] = v if k.startswith("model.") else v
        return out

    def add_model_prefix(sd):
        return {("model." + k if not k.startswith("model.") else k): v for k, v in sd.items()}

    head_expects_model_prefix = any(k.startswith("model.") for k in head_keys)
    ckpt_has_model_prefix = any(k.startswith("model.") for k in ckpt_keys)

    if head_expects_model_prefix and not ckpt_has_model_prefix:
        ckpt = add_model_prefix(ckpt)
    elif (not head_expects_model_prefix) and ckpt_has_model_prefix:
        ckpt = strip_model_prefix(ckpt)

    head_mod.load_state_dict(ckpt, strict=True)

    # float32 test
    head = head.to("cuda", dtype=torch.float32).eval()
    head_mod = getattr(head, "model", head)

    # The head input dim is 20480 in your probes; use that directly
    D = 20480
    x1 = torch.randn(8, D, device="cuda", dtype=torch.float32)
    x2 = torch.randn(8, D, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        y1 = head_mod(x1).cpu().numpy()
        y2 = head_mod(x2).cpu().numpy()

    max_abs = float(np.max(np.abs(y1 - y2)))
    mean_abs = float(np.mean(np.abs(y1 - y2)))
    print("Random input sensitivity:")
    print("  max_abs(y1-y2) =", max_abs)
    print("  mean_abs(y1-y2) =", mean_abs)
    print("  y1[0] =", y1[0])
    print("  y2[0] =", y2[0])

if __name__ == "__main__":
    main()
