#!/usr/bin/env python3
import os, sys, time
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

import mujoco as mj
import torch
from safetensors.torch import load_file
import json

# ---- repo local import of your sim ----
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
from headless_cdpr_egl import HeadlessCDPRSimulation  # your class

# After sys.path.append for openvla_path / libero_path and BEFORE you construct the model:
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

# ---- CDPR monkey-patch: disable RLDS action unnormalization ----
import types

try:
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
except ImportError:
    OpenVLAForActionPrediction = None

try:
    from prismatic.models.vlas.openvla import OpenVLA
except ImportError:
    OpenVLA = None

def cdpr_unnormalize_actions(self, normalized_actions, unnorm_key=None):
    """
    Override for *_unnormalize_actions.

    For CDPR continuous 5-DoF actions, we DO NOT want RLDS-based un-normalization
    (which assumes 7-D actions and causes shape mismatches). The CDPR action head
    already outputs metric deltas, so just return them as-is.
    """
    return normalized_actions

# Patch class-level methods (covers models constructed AFTER this)
if OpenVLAForActionPrediction is not None:
    OpenVLAForActionPrediction._unnormalize_actions = cdpr_unnormalize_actions
    print("🔧 Patched OpenVLAForActionPrediction._unnormalize_actions (class-level).")

if OpenVLA is not None:
    OpenVLA._unnormalize_actions = cdpr_unnormalize_actions
    print("🔧 Patched OpenVLA._unnormalize_actions (class-level).")

# Add OpenVLA-OFT to path
openvla_path = "/root/repo/openvla-oft"
if openvla_path not in sys.path:
    sys.path.append(openvla_path)

# Add LIBERO to path
libero_path = "/root/repo/LIBERO"
if libero_path not in sys.path:
    sys.path.append(libero_path)

print("🔍 Importing OpenVLA-OFT modules...")
try:
    from experiments.robot.libero.run_libero_eval import GenerateConfig
    from experiments.robot.openvla_utils import (
        get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action,
        _load_dataset_stats,
    )
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK
    print("✅ Successfully imported OpenVLA-OFT modules")
except ImportError as e:
    print(f"❌ Error importing OpenVLA-OFT modules: {e}")
    print("Trying alternative import approach...")
    import importlib.util

    # Import GenerateConfig
    spec = importlib.util.spec_from_file_location(
        "GenerateConfig",
        "/root/repo/openvla-oft/experiments/robot/libero/run_libero_eval.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    GenerateConfig = module.GenerateConfig

    # Import openvla_utils
    spec = importlib.util.spec_from_file_location(
        "openvla_utils",
        "/root/repo/openvla-oft/experiments/robot/openvla_utils.py",
    )
    utils_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_module)
    get_action_head = utils_module.get_action_head
    get_processor = utils_module.get_processor
    get_proprio_projector = utils_module.get_proprio_projector
    get_vla = utils_module.get_vla
    get_vla_action = utils_module.get_vla_action

    from prismatic.vla.constants import NUM_ACTIONS_CHUNK
    print("✅ Successfully imported with alternative approach")

from peft import PeftModel


# ---------- Observation and action helpers (5-DoF) ----------

def make_observation(sim, task_text, gripper_range=(0.0, 0.06)):
    """
    Build observation for OpenVLA-OFT (CDPR, 5-DoF).

    State is 5-D absolute:
      [x, y, z, yaw, grip]

    where:
      - x,y,z in world meters (like in your dataset)
      - yaw in radians
      - grip is physical opening mapped to [0,1]
    """
    full_rgb = sim.capture_frame(sim.overview_cam, "overview")
    wrist_rgb = sim.capture_frame(sim.ee_cam, "ee_camera")

    ee = sim.get_end_effector_position().astype(np.float32)
    yaw = float(sim.get_yaw()) if hasattr(sim, "get_yaw") else 0.0
    grip_phys = float(getattr(sim, "get_gripper_opening", lambda: 0.03)())

    g_lo, g_hi = gripper_range
    grip_norm = (grip_phys - g_lo) / (g_hi - g_lo + 1e-9)
    grip_norm = float(np.clip(grip_norm, 0.0, 1.0))

    state = np.array(
        [ee[0], ee[1], ee[2], yaw, grip_norm],
        dtype=np.float32,
    )

    obs = {
        "full_image": np.ascontiguousarray(full_rgb),
        "wrist_image": np.ascontiguousarray(wrist_rgb),
        "state": state,
        "task_description": task_text,
    }
    return obs, state.size


def map_action_to_cdpr(
    act5,
    xyz_bounds,
    gripper_range,
    current_xyz,
    current_yaw,
    current_grip_phys,
    max_xyz_step=0.05,
    max_yaw_step=np.pi * 0.1,
    max_grip_step=0.02,
):
    """
    Map 5D action [Δx, Δy, Δz, Δyaw, Δgrip_phys] to CDPR target state.

    We assume get_vla_action already returns *unnormalized* deltas in
    the same units as your training data (meters, radians, grip units).
    """
    a = np.array(act5, dtype=float).flatten()
    if a.size < 5:
        a = np.pad(a, (0, 5 - a.size))

    dx, dy, dz, dyaw, dgrip = a

    # Clamp deltas for safety
    dx = float(np.clip(dx, -max_xyz_step, max_xyz_step))
    dy = float(np.clip(dy, -max_xyz_step, max_xyz_step))
    dz = float(np.clip(dz, -max_xyz_step, max_xyz_step))
    dyaw = float(np.clip(dyaw, -max_yaw_step, max_yaw_step))
    dgrip = float(np.clip(dgrip, -max_grip_step, max_grip_step))

    # Compute targets
    target_xyz = current_xyz + np.array([dx, dy, dz], dtype=float)
    target_yaw = current_yaw + dyaw
    target_grip_phys = current_grip_phys + dgrip

    # Clamp XYZ to bounds
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = xyz_bounds
    target_xyz[0] = np.clip(target_xyz[0], xlo, xhi)
    target_xyz[1] = np.clip(target_xyz[1], ylo, yhi)
    target_xyz[2] = np.clip(target_xyz[2], zlo, zhi)

    # Wrap yaw into [-pi, pi]
    target_yaw = ((target_yaw + np.pi) % (2 * np.pi)) - np.pi

    # Clamp gripper to physical range
    g_lo, g_hi = gripper_range
    target_grip_phys = float(np.clip(target_grip_phys, g_lo, g_hi))

    return target_xyz, target_yaw, target_grip_phys


# ---------- Main runner ----------

def main():
    import argparse

    ap = argparse.ArgumentParser("OpenVLA-OFT CDPR Runner")
    ap.add_argument("--xml", required=True, help="Wrapper MJCF")

    # Accept both --base-ckpt and --ckpt
    ap.add_argument(
        "--base-ckpt",
        "--ckpt",
        dest="base_ckpt",
        default="moojink/openvla-7b-oft-finetuned-libero-spatial",
        help="Base VLA checkpoint (HF repo or local path)",
    )

    ap.add_argument(
        "--adapter-path",
        default="/root/oft_cdpr_ckpts/cdpr_finetune_20251203-133649/vla_cdpr_adapter",
    )
    ap.add_argument(
        "--action-head-path",
        default="/root/oft_cdpr_ckpts/cdpr_finetune_20251203-133649/action_head_cdpr.pt",
    )
    ap.add_argument("--center-crop", action="store_true", default=True)
    ap.add_argument("--no-center-crop", dest="center_crop", action="store_false")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--chunk-length", type=int, default=None, help="Override NUM_ACTIONS_CHUNK")
    ap.add_argument("--instr", default="Pick up the ketchup bottle.")
    ap.add_argument("--x-bound", default="-0.8,0.8")
    ap.add_argument("--y-bound", default="-0.8,0.8")
    ap.add_argument("--z-bound", default="0.10,1.20")
    ap.add_argument("--grip-range", default="0.0,0.06")
    ap.add_argument("--no-adapter", action="store_true", help="Skip adapter loading")
    args = ap.parse_args()

    def parse_pair(s):
        lo, hi = map(float, s.split(","))
        return (lo, hi)

    xyz_bounds = (parse_pair(args.x_bound), parse_pair(args.y_bound), parse_pair(args.z_bound))
    gr_range = parse_pair(args.grip_range)

    print("=" * 80)
    print("🤖 OpenVLA-OFT CDPR Runner (5-DoF)")
    print("=" * 80)
    print(f"Instruction: {args.instr}")
    print(f"Steps: {args.steps}")
    print(f"XYZ bounds: {xyz_bounds}")
    print(f"Gripper range: {gr_range}")
    print(f"Adapter: {'Disabled' if args.no_adapter else 'Enabled'}")
    if not args.no_adapter:
        print(f"Adapter path: {args.adapter_path}")
        print(f"Action head: {args.action_head_path}")
    print(f"Base checkpoint: {args.base_ckpt}")
    print("=" * 80)

    # ---- Build config ----
    print("\n📝 Building config...")
    cfg = GenerateConfig(
        pretrained_checkpoint=args.base_ckpt,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=args.center_crop,
        num_open_loop_steps=args.chunk_length if args.chunk_length else NUM_ACTIONS_CHUNK,
        unnorm_key="bc_z",
    )
    cfg.cdpr_dataset_stats_path = (
        "/root/oft_cdpr_ckpts/"
        "openvla-7b-oft-finetuned-libero-spatial+cdpr_local+b1+lr-0.0001+lora-r32+dropout-0.0/"
        "dataset_statistics.json"
    )

    # 👇 Add this line
    cfg.cdpr_action_head_path = args.action_head_path


    # ---- Init sim ----
    print("\n🎮 Initializing simulation...")
    sim = HeadlessCDPRSimulation(args.xml, output_dir="oft_cdpr_runs")
    sim.initialize()
    if hasattr(sim, "hold_current_pose"):
        sim.hold_current_pose(warm_steps=0)

    ee_start = sim.get_end_effector_position().copy()
    yaw_start = sim.get_yaw() if hasattr(sim, "get_yaw") else 0.0
    grip_start = getattr(sim, "get_gripper_opening", lambda: 0.03)()
    print(f"📍 Start: EE={ee_start.round(3)}, Yaw={yaw_start:.3f}, Grip={grip_start:.3f}")

    # ---- Load base VLA model ----
    print("\n🤖 Loading VLA base model...")
    vla_base = get_vla(cfg)
    vla_base.eval()

    device = next(vla_base.parameters()).device
    print(f"✅ Base model on {device}")

    # ---- Load CDPR adapter via PEFT ----
    if not args.no_adapter and os.path.isdir(args.adapter_path):
        print("\n🔧 Loading CDPR adapter (PEFT)...")
        try:
            vla = PeftModel.from_pretrained(vla_base, args.adapter_path)
            vla.eval()
            device = next(vla.parameters()).device
            print("✅ Adapter loaded into VLA model")
        except Exception as e:
            print(f"❌ Error loading adapter: {e}")
            print("⚠️  Falling back to base model without CDPR adapter")
            vla = vla_base
    else:
        print("⚠️  Adapter disabled or path not found; using base model")
        vla = vla_base
        
    device = next(vla.parameters()).device
    # ---- Instance-level monkey-patch for _unnormalize_actions ----
    def cdpr_unnormalize_actions_inst(self, normalized_actions, unnorm_key=None):
        return normalized_actions

    # target is the underlying base model if PEFT-wrapped, otherwise vla itself
    target = getattr(vla, "base_model", vla)

    # patch on the base model
    target._unnormalize_actions = types.MethodType(cdpr_unnormalize_actions_inst, target)
    print(f"🔧 Patched instance _unnormalize_actions on {type(target).__name__}")


    import torch

    # We know OpenVLA runs its policy in bfloat16 (your errors also reference BFloat16),
    # so we standardize all CDPR add-ons to this dtype.
    half_dtype = torch.bfloat16

    # --- Ensure vision backbone knows about multi-image input ---
    if hasattr(vla, "vision_backbone"):
        vb = vla.vision_backbone
        old_nimg = getattr(vb, "num_images_in_input", None)
        try:
            vb.num_images_in_input = cfg.num_images_in_input
            print(f"👀 Vision backbone num_images_in_input: {old_nimg} -> {vb.num_images_in_input}")
        except Exception as e:
            print(f"⚠️ Could not set vision_backbone.num_images_in_input: {e}")


        # --- Ensure vision backbone knows we are using multi-image input (full + wrist) ---
    if hasattr(vla, "vision_backbone"):
        vb = vla.vision_backbone
        old_nimg = getattr(vb, "num_images_in_input", None)
        try:
            vb.num_images_in_input = cfg.num_images_in_input
            print(
                f"👀 Vision backbone num_images_in_input: {old_nimg} -> {vb.num_images_in_input}"
            )
        except Exception as e:
            print(f"⚠️ Could not set vision_backbone.num_images_in_input: {e}")
        
    cdpr_stats_root = (
        "/root/oft_cdpr_ckpts/"
        "openvla-7b-oft-finetuned-libero-spatial+cdpr_local+b1+lr-0.0001+lora-r32+dropout-0.0"
    )
    print(f"\n📊 Loading CDPR dataset statistics from: {cdpr_stats_root}")
    _load_dataset_stats(vla, cdpr_stats_root)
    
    print("cdpr_local entry keys:", vla.norm_stats["cdpr_local"].keys())
    
    print("🔑 Available norm_stats keys:", getattr(vla, "norm_stats", {}).keys())
        
    device = next(vla.parameters()).device
    # ---- Load action head ----
    print("\n🎯 Loading action head...")

    if not args.no_adapter and os.path.exists(args.action_head_path):
        action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
        print("✅ Loaded fine-tuned CDPR action head via get_action_head")
    else:
        print("⚠️ Using base action head (no adapter or missing path)")
        action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

    action_head = action_head.to(device=device, dtype=half_dtype)
    action_head.eval()
    
    print(f"🎯 Action head dtype set to {action_head.model.fc1.weight.dtype}")

    # ---- Load processor and proprio projector ----
    print("\n⚙️ Loading processor & proprio projector...")
    processor = get_processor(cfg)

    obs, proprio_dim = make_observation(sim, args.instr, gripper_range=gr_range)
    proprio_projector = get_proprio_projector(
        cfg, llm_dim=vla.llm_dim, proprio_dim=proprio_dim
    )
    proprio_projector = proprio_projector.to(device=device, dtype=half_dtype)
    print(f"⚙️ Proprio projector dtype set to {next(proprio_projector.parameters()).dtype}")


    print(f"\n📊 System ready:")
    print(f"   - Action chunk length: {cfg.num_open_loop_steps}")
    print(f"   - Proprio dimension: {proprio_dim}")
    print(f"   - Images: {obs['full_image'].shape} (full), {obs['wrist_image'].shape} (wrist)")

    # Save debug images
    Image.fromarray(obs["full_image"]).save("debug_full.png")
    Image.fromarray(obs["wrist_image"]).save("debug_wrist.png")
    print("💾 Saved debug images: debug_full.png, debug_wrist.png")

    # ---- Action generation helper ----
    def get_action_chunk():
        """Get a chunk of 5-DoF actions from the policy."""
        o, _ = make_observation(sim, args.instr, gripper_range=gr_range)
        try:
            acts = get_vla_action(
                cfg, vla, processor, o, o["task_description"], action_head, proprio_projector
            )
            if acts and len(acts) > 0:
                acts_np = np.array(acts, dtype=float)
                print(f"🎯 Generated {len(acts)} actions (shape {acts_np.shape})")
                print(f"   First: {acts_np[0].round(3)}")
                print(f"   Mean: {acts_np.mean(axis=0).round(3)}")
            return acts
        except Exception as e:
            print(f"❌ Error generating actions: {e}")
            # Return small safe 5D actions
            return [
                np.zeros(5, dtype=float) for _ in range(cfg.num_open_loop_steps)
            ]

    # ---- Initial test ----
    print("\n🧪 Testing action generation...")
    chunk = get_action_chunk()
    if not chunk:
        print("❌ No actions generated, exiting")
        return

    # ---- Rollout ----
    print("\n🚀 Starting rollout...")
    current_chunk = chunk
    chunk_idx = 0

    current_xyz = ee_start.copy()
    current_yaw = yaw_start
    current_grip_phys = grip_start

    for step in range(args.steps):
        # Get new chunk if needed
        if chunk_idx >= len(current_chunk):
            print(f"\n🔄 Replanning at step {step}...")
            current_chunk = get_action_chunk()
            chunk_idx = 0

        # Get current action (5D)
        action_5d = current_chunk[chunk_idx]
        chunk_idx += 1

        # Map to CDPR commands
        target_xyz, target_yaw, target_grip_phys = map_action_to_cdpr(
            action_5d,
            xyz_bounds,
            gr_range,
            current_xyz,
            current_yaw,
            current_grip_phys,
        )

        # Update current state
        current_xyz = target_xyz.copy()
        current_yaw = target_yaw
        current_grip_phys = target_grip_phys

        # Apply to simulation
        sim.set_target_position(target_xyz)
        if hasattr(sim, "set_yaw"):
            sim.set_yaw(target_yaw)
        if hasattr(sim, "set_gripper"):
            sim.set_gripper(target_grip_phys)

        sim.run_simulation_step(capture_frame=True)

        # Print progress
        if step % 10 == 0 or step < 5:
            ee_pos = sim.get_end_effector_position()[:3]
            print(
                f"  Step {step:3d}: EE=({ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}) "
                f"Grip={target_grip_phys:.3f}"
            )

    # ---- Save results ----
    print("\n💾 Saving results...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(sim.output_dir) / f"run_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    ee_end = sim.get_end_effector_position().copy()
    distance = np.linalg.norm(ee_end - ee_start)
    print(f"\n📊 Results:")
    print(f"   Start: {ee_start.round(3)}")
    print(f"   End: {ee_end.round(3)}")
    print(f"   Distance: {distance:.3f} meters")

    sim.save_trajectory_results(str(outdir), "openvla_cdpr")
    sim.cleanup()

    print(f"\n✅ Run completed!")
    print(f"📁 Results saved to: {outdir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
