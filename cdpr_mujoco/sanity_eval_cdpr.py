#!/usr/bin/env python3
# Load a saved checkpoint and run a single forward pass on a held-out step to
# confirm reasonable action magnitudes.

import pickle, glob, json
from pathlib import Path
import numpy as np

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

CKPT_DIR = Path("./oft_cdpr_ckpts")  # adjust

def main():
    # Use your new unnorm key
    cfg = GenerateConfig(
        pretrained_checkpoint = str(CKPT_DIR / "epoch_0002.pt"),  # latest
        use_l1_regression = True,
        use_diffusion = False,
        use_film = False,
        num_images_in_input = 2,
        use_proprio = True,
        load_in_8bit = False,
        load_in_4bit = False,
        center_crop = True,
        num_open_loop_steps = NUM_ACTIONS_CHUNK,
        unnorm_key = "cdpr_synth",
    )
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

    # Create a minimal "observation" sample: PNGs + small state + language.
    # You can dump one from your exporter or read raw frames here.
    with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as f:
        obs = pickle.load(f)
    # Replace obs fields with your images/state/lang to sanity-check if you prefer.

    acts = get_vla_action(cfg, vla, processor, obs, obs["task_description"], action_head, proprio_projector)
    print("Δ-actions:", acts)

if __name__ == "__main__":
    main()
