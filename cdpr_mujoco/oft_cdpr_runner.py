#!/usr/bin/env python3
import os, sys, time
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

import mujoco as mj
import torch

# ---- Apply cumsum patch ----
_orig_cumsum = torch.cumsum


def cumsum_bool_safe(input, dim, *args, **kwargs):
    if isinstance(input, torch.Tensor) and input.dtype == torch.bool:
        input = input.to(torch.int64)
    return _orig_cumsum(input, dim, *args, **kwargs)


torch.cumsum = cumsum_bool_safe
print("🔧 Applied cumsum patch")

from safetensors.torch import load_file
import json
import yaml

# ---- repo local import of your sim ----
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))

# ✅ Add VLA_CDPR root so `import cdpr_mujoco` works in other modules
vla_cdpr_root = "/root/repo/VLA_CDPR"
if vla_cdpr_root not in sys.path:
    sys.path.append(vla_cdpr_root)

from headless_cdpr_egl import HeadlessCDPRSimulation  # your class

# Add OpenVLA-OFT to path BEFORE imports
openvla_path = "/root/repo/openvla-oft"
if openvla_path not in sys.path:
    sys.path.append(openvla_path)

# Add LIBERO to path
libero_path = "/root/repo/LIBERO"
if libero_path not in sys.path:
    sys.path.append(libero_path)

# Add CDPR_Dataset to path
cdpr_dataset_root = "/root/repo/CDPR_Dataset"
if cdpr_dataset_root not in sys.path:
    sys.path.append(cdpr_dataset_root)

# Now import OpenVLA-OFT modules
try:
    from experiments.robot.libero.run_libero_eval import GenerateConfig
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_processor,
        get_proprio_projector,
        get_vla,
        get_vla_action,
        _load_dataset_stats,
    )
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
    print("✅ Successfully imported OpenVLA-OFT modules")
except ImportError as e:
    print(f"❌ Error importing OpenVLA-OFT modules: {e}")
    sys.exit(1)

# Import dataset helper utilities
try:
    from cdpr_dataset.generate_cdpr_dataset import build_wrapper_if_needed
    from cdpr_dataset.synthetic_tasks import task_language, place_objects_non_overlapping
    print("✅ Imported CDPR dataset helpers (build_wrapper_if_needed, task_language, place_objects_non_overlapping)")
except ImportError as e:
    print(f"⚠️ Could not import CDPR dataset helpers: {e}")
    build_wrapper_if_needed = None
    task_language = None
    place_objects_non_overlapping = None

from peft import PeftModel
import types


# ======================================================================
#  CDPR-specific observation helper
# ======================================================================


def make_observation(sim, task_text, gripper_range=(0.0, 0.06)):
    """
    Build observation for OpenVLA-OFT (CDPR, 5-DoF).

    Returns:
      obs: dict compatible with OpenVLA-OFT get_vla_action()
      proprio_dim: int (dimension of 'state')
    """
    # Capture images
    full_rgb = sim.capture_frame(sim.overview_cam, "overview")
    wrist_rgb = sim.capture_frame(sim.ee_cam, "ee_camera")

    # Get state: [x, y, z, yaw, grip_norm]
    ee = sim.get_end_effector_position().astype(np.float32)
    yaw = float(sim.get_yaw()) if hasattr(sim, "get_yaw") else 0.0
    grip_phys = float(getattr(sim, "get_gripper_opening", lambda: 0.03)())

    # Normalize gripper to [0, 1]
    g_lo, g_hi = gripper_range
    grip_norm = (grip_phys - g_lo) / (g_hi - g_lo + 1e-9)
    grip_norm = np.clip(grip_norm, 0.0, 1.0)

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


# ======================================================================
#  CDPR-specific action mapping (5D → physical commands)
# ======================================================================


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
    Map 5D action [dx, dy, dz, dyaw, dgrip] to CDPR target state.
    """
    a = np.array(act5, dtype=float).flatten()
    if a.size < 5:
        a = np.pad(a, (0, 5 - a.size))

    dx, dy, dz, dyaw, dgrip = a

    # Clamp deltas
    dx = np.clip(dx, -max_xyz_step, max_xyz_step)
    dy = np.clip(dy, -max_xyz_step, max_xyz_step)
    dz = np.clip(dz, -max_xyz_step, max_xyz_step)
    dyaw = np.clip(dyaw, -max_yaw_step, max_yaw_step)
    dgrip = np.clip(dgrip, -max_grip_step, max_grip_step)

    # Compute targets
    target_xyz = current_xyz + np.array([dx, dy, dz], dtype=float)
    target_yaw = current_yaw + dyaw
    target_grip_phys = current_grip_phys + dgrip

    # Clamp to bounds
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = xyz_bounds
    target_xyz[0] = np.clip(target_xyz[0], xlo, xhi)
    target_xyz[1] = np.clip(target_xyz[1], ylo, yhi)
    target_xyz[2] = np.clip(target_xyz[2], zlo, zhi)

    # Wrap yaw
    target_yaw = ((target_yaw + np.pi) % (2 * np.pi)) - np.pi

    # Clamp gripper
    g_lo, g_hi = gripper_range
    target_grip_phys = np.clip(target_grip_phys, g_lo, g_hi)

    return target_xyz, target_yaw, target_grip_phys


# ======================================================================
#  Main runner
# ======================================================================


def main():
    import argparse

    ap = argparse.ArgumentParser("OpenVLA-OFT CDPR Runner")

    # If you pass --xml explicitly, we use that and ignore catalog scene/task logic.
    ap.add_argument(
        "--xml",
        required=False,
        help="Wrapper MJCF (if not using dataset catalog scene/object).",
    )

    ap.add_argument(
        "--base-ckpt",
        default="moojink/openvla-7b-oft-finetuned-libero-spatial",
        help="Base VLA checkpoint",
    )
    ap.add_argument(
        "--adapter-path",
        default="/root/oft_cdpr_ckpts/cdpr_finetune_20251203-133649/vla_cdpr_adapter",
    )
    ap.add_argument(
        "--action-head-path",
        default="/root/oft_cdpr_ckpts/cdpr_finetune_20251203-133649/action_head_cdpr.pt",
    )

    # We will auto-fill instr from dataset task if not provided.
    ap.add_argument(
        "--instr",
        type=str,
        default=None,
        help="Language instruction. "
        "If omitted and no --xml is provided, we use cdpr_dataset.task_language(task_name, object).",
    )

    ap.add_argument("--center-crop", action="store_true", default=True)
    ap.add_argument("--no-center-crop", dest="center_crop", action="store_false")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--chunk-length", type=int, default=None)

    # CDPR workspace bounds
    ap.add_argument("--x-bound", default="-0.8,0.8")
    ap.add_argument("--y-bound", default="-0.8,0.8")
    ap.add_argument("--z-bound", default="0.10,1.20")
    ap.add_argument("--grip-range", default="0.0,0.06")
    ap.add_argument("--no-adapter", action="store_true")

    # ---- Dataset-style configuration (matching your generator command) ----
    ap.add_argument(
        "--catalog",
        type=str,
        default="/root/repo/CDPR_Dataset/cdpr_dataset/datasets/cdpr_scene_catalog.yaml",
        help="Path to scene/object YAML (same as used in generate_cdpr_dataset). "
             "Ignored if --xml is provided.",
    )
    ap.add_argument(
        "--scene",
        type=str,
        default="desk",
        help="Scene name from the catalog. Default matches your catalog: 'desk'.",
    )
    ap.add_argument(
        "--object",
        type=str,
        default="milk",
        help="Main object name from catalog. Default: 'milk'.",
    )
    ap.add_argument(
        "--task-name",
        type=str,
        default="put_into_bowl",
        help="Task name used in dataset generator. Default: 'put_into_bowl'.",
    )

    args = ap.parse_args()

    def parse_pair(s):
        lo, hi = map(float, s.split(","))
        return (lo, hi)

    xyz_bounds = (parse_pair(args.x_bound), parse_pair(args.y_bound), parse_pair(args.z_bound))
    gr_range = parse_pair(args.grip_range)

    print("=" * 80)
    print("🤖 OpenVLA-OFT CDPR Runner (5-DoF)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Resolve XML and instruction: either user-specified XML, or
    # dataset-style scene+task (matching generate_cdpr_dataset.py).
    # ------------------------------------------------------------------
    def _require(cond, msg):
        if not cond:
            raise SystemExit(msg)

    dataset_scene_name = None
    dataset_object_name = None

    if args.xml is not None:
        # Explicit XML → behave like your original runner
        xml_path = args.xml
        print(f"\n🧱 Using explicit wrapper XML: {xml_path}")

        if args.instr is not None:
            instr = args.instr
        else:
            instr = "Pick up the ketchup bottle."
        print(f"🗣  Instruction: '{instr}' (explicit/legacy mode)")
    else:
        # Dataset-style mode: reuse catalog + build_wrapper_if_needed + task_language
        _require(build_wrapper_if_needed is not None, "CDPR dataset helpers not available (build_wrapper_if_needed).")
        _require(task_language is not None, "CDPR dataset helpers not available (task_language).")

        catalog_path = args.catalog
        print(f"\n📚 Loading catalog for dataset-style scene: {catalog_path}")
        with open(catalog_path, "r") as f:
            cfg_yaml = yaml.safe_load(f)

        defaults = cfg_yaml.get("defaults", {})
        scenes_cfg = cfg_yaml.get("scenes", [])

        scene_entry = None
        for entry in scenes_cfg:
            if isinstance(entry, dict):
                if entry.get("name") == args.scene:
                    scene_entry = entry
                    break
            else:
                if str(entry) == args.scene:
                    scene_entry = {"name": str(entry), "objects": []}
                    break

        _require(scene_entry is not None, f"Scene '{args.scene}' not found in catalog.")

        scene_name = scene_entry["name"]
        object_names = scene_entry.get("objects", [])
        dataset_scene_name = scene_name

        if args.object is not None:
            dataset_object_name = args.object
            if object_names and dataset_object_name not in object_names:
                print(
                    f"⚠️ Warning: object '{dataset_object_name}' not in catalog scene objects {object_names}. "
                    f"Using it anyway."
                )
        else:
            dataset_object_name = object_names[0] if object_names else "target_object"

        scene_z = defaults.get("scene_z", -0.85)
        ee_start = defaults.get("ee_start", (0.0, 0.0, 0.25))
        table_z = defaults.get("table_z", 0.15)
        settle_t = defaults.get("settle_time", 1.0)

        print(
            f"\n🏗  Building (or reusing) dataset-style wrapper for scene='{scene_name}', "
            f"objects={object_names}"
        )
        wrapper_xml = build_wrapper_if_needed(
            scene_name,
            object_names,
            scene_z=scene_z,
            ee_start=ee_start,
            table_z=table_z,
            settle_time=settle_t,
        )
        xml_path = str(wrapper_xml)
        print(f"🧱 Using wrapper XML from dataset builder: {xml_path}")

        # Decide instruction
        if args.instr is not None:
            instr = args.instr
            print(f"\n🗣  Using user-provided instruction: '{instr}'")
        else:
            task_name = args.task_name or "put_into_bowl"
            obj_for_lang = dataset_object_name or "object"
            instr = task_language(task_name, obj_for_lang)
            print(
                f"\n🗣  Using dataset-style language for task='{task_name}', object='{obj_for_lang}':\n"
                f"    '{instr}'"
            )

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
        # We patch _unnormalize_actions below to be identity for CDPR.
        unnorm_key=None,
    )

    # ---- Init sim ----
    print("\n🎮 Initializing simulation...")
    sim = HeadlessCDPRSimulation(xml_path=xml_path, output_dir="oft_cdpr_runs")
    sim.initialize()

    # Optional: apply central placement like in generate_cdpr_dataset.run_episode()
    if args.xml is None and place_objects_non_overlapping is not None:
        try:
            real_obj = getattr(sim, "get_object_body_name", lambda: None)()
            if real_obj is None:
                real_obj = dataset_object_name
            if real_obj is not None:
                xy_bounds = ((-0.12, 0.12), (-0.12, 0.12), 0.10)
                print(f"📦 Applying dataset-style central placement for object '{real_obj}'")
                place_objects_non_overlapping(sim, [real_obj], xy_bounds, min_gap=0.015)
            else:
                print("⚠️ Could not determine object body name for dataset-style placement.")
        except Exception as e:
            print(f"⚠️ Dataset-style placement failed: {e}")

    ee_start = sim.get_end_effector_position().copy()
    yaw_start = sim.get_yaw() if hasattr(sim, "get_yaw") else 0.0
    grip_start = getattr(sim, "get_gripper_opening", lambda: 0.03)()
    print(f"📍 Start: EE={ee_start.round(3)}, Yaw={yaw_start:.3f}, Grip={grip_start:.3f}")

    # ---- Load base VLA model ----
    print("\n🤖 Loading VLA base model...")
    vla_base = get_vla(cfg)
    vla_base.eval()

    # Optional: instance-level patch for unnormalize_actions (identity)
    def cdpr_unnormalize_actions_inst(self, normalized_actions, unnorm_key=None):
        # For CDPR, we treat actions as already in the right scale
        return normalized_actions

    vla_base._unnormalize_actions = types.MethodType(cdpr_unnormalize_actions_inst, vla_base)
    print("🔧 Patched _unnormalize_actions on base model (identity)")

    # ---- Load adapter ----
    if not args.no_adapter and os.path.isdir(args.adapter_path):
        print("\n🔧 Loading CDPR adapter (PEFT)...")
        try:
            vla = PeftModel.from_pretrained(vla_base, args.adapter_path)
            vla.eval()
            print("✅ Adapter loaded")
        except Exception as e:
            print(f"❌ Error loading adapter: {e}")
            print("⚠️  Falling back to base model")
            vla = vla_base
    else:
        vla = vla_base

    device = next(vla.parameters()).device
    print(f"✅ Model on {device}")

    # Set vision backbone for multi-image input
    if hasattr(vla, "vision_backbone"):
        try:
            vla.vision_backbone.num_images_in_input = cfg.num_images_in_input
            print(f"👀 Set vision_backbone.num_images_in_input to {cfg.num_images_in_input}")
        except Exception as e:
            print(f"⚠️ Could not set num_images_in_input: {e}")

    # ---- Load dataset statistics ----
    cdpr_stats_root = os.path.join(
        "/root/oft_cdpr_ckpts",
        "openvla-7b-oft-finetuned-libero-spatial+cdpr_local+b1+lr-0.0001+lora-r32+dropout-0.0",
    )
    print(f"\n📊 Loading CDPR dataset statistics from: {cdpr_stats_root}")

    stats_path = os.path.join(cdpr_stats_root, "dataset_statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        # Store in model for compatibility – if you later want to use unnorm_key="cdpr_local"
        if not hasattr(vla, "norm_stats"):
            vla.norm_stats = {}
        vla.norm_stats["cdpr_local"] = stats
        print("✅ Loaded CDPR dataset statistics (as 'cdpr_local')")
    else:
        print("⚠️ Dataset statistics not found")

    # ---- Load action head ----
    print("\n🎯 Loading action head...")
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

    if os.path.exists(args.action_head_path):
        try:
            action_head_state_dict = torch.load(args.action_head_path, map_location="cpu")

            # Adjust keys if needed (strip 'model.' prefix)
            if any(k.startswith("model.") for k in action_head_state_dict.keys()):
                new_state_dict = {}
                for k, v in action_head_state_dict.items():
                    if k.startswith("model."):
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
                action_head_state_dict = new_state_dict

            missing, unexpected = action_head.load_state_dict(action_head_state_dict, strict=False)
            if missing:
                print(f"⚠️ Missing keys in action head: {missing}")
            if unexpected:
                print(f"⚠️ Unexpected keys in action head: {unexpected}")
            print("✅ Loaded fine-tuned CDPR action head")
        except Exception as e:
            print(f"❌ Error loading action head weights: {e}")
    else:
        print("⚠️ Action head weights not found, using initialized weights")

    action_head = action_head.to(device=device, dtype=torch.bfloat16)
    action_head.eval()

    # ---- Load processor & proprio projector ----
    print("\n⚙️ Loading processor & proprio projector...")
    processor = get_processor(cfg)

    # Get proprio dimension from observation
    obs, proprio_dim = make_observation(sim, instr, gripper_range=gr_range)

    proprio_projector = get_proprio_projector(
        cfg, llm_dim=vla.llm_dim, proprio_dim=proprio_dim
    )
    proprio_projector = proprio_projector.to(device=device, dtype=torch.bfloat16)
    proprio_projector.eval()

    print(f"\n📊 System ready:")
    print(f"   - Action chunk length: {cfg.num_open_loop_steps}")
    print(f"   - Proprio dimension: {proprio_dim}")
    print(f"   - ACTION_DIM: {ACTION_DIM}")

    # ---- Test action generation using built-in get_vla_action ----
    print("\n🧪 Testing action generation...")
    try:
        # NOTE: this get_vla_action is from experiments.robot.openvla_utils
        # It internally calls vla.predict_action(), builds labels & action masks correctly,
        # and returns a numpy array of shape (chunk, ACTION_DIM).
        test_actions = get_vla_action(
            cfg,
            vla,
            processor,
            obs,
            instr,
            action_head,
            proprio_projector,
        )

        test_actions = np.asarray(test_actions, dtype=np.float32)
        print(f"✅ Generated {len(test_actions)} actions")
        print(f"   Shape: {test_actions.shape}")
        print(f"   First action: {test_actions[0]}")
        print(f"   Mean: {test_actions.mean(axis=0)}")
        print(f"   Std: {test_actions.std(axis=0)}")

        if np.allclose(test_actions, 0, atol=1e-6):
            print("⚠️ WARNING: Actions are all zeros!")
            print("   Testing action head with random input...")
            with torch.no_grad():
                random_features = torch.randn(
                    1, vla.llm_dim, device=device, dtype=torch.bfloat16
                )
                random_actions = action_head(random_features)
                print(f"   Random test output shape: {random_actions.shape}")
                print(f"   Random test values (first 5): {random_actions[0, :5]}")
    except Exception as e:
        print(f"❌ Error in test action generation: {e}")
        import traceback

        traceback.print_exc()
        return

    # ---- Rollout ----
    print("\n🚀 Starting rollout...")

    current_xyz = ee_start.copy()
    current_yaw = yaw_start
    current_grip_phys = grip_start

    # Use initial chunk from test_actions
    current_chunk = test_actions
    chunk_idx = 0

    for step in range(args.steps):
        # Get new chunk if needed
        if chunk_idx >= len(current_chunk):
            print(f"\n🔄 Replanning at step {step}...")
            obs, _ = make_observation(sim, instr, gripper_range=gr_range)
            try:
                current_chunk = get_vla_action(
                    cfg,
                    vla,
                    processor,
                    obs,
                    instr,
                    action_head,
                    proprio_projector,
                )
                current_chunk = np.asarray(current_chunk, dtype=np.float32)
                chunk_idx = 0
                print(f"   Generated {len(current_chunk)} new actions")
            except Exception as e:
                print(f"❌ Error replanning: {e}")
                # Use small safe actions
                current_chunk = np.zeros((cfg.num_open_loop_steps, 5), dtype=float)
                chunk_idx = 0

        # Get current action (5D for CDPR)
        action_5d = current_chunk[chunk_idx]
        chunk_idx += 1

        if step % 5 == 0:
            print(f"Step {step}: Raw action: {action_5d}")

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

        if step % 10 == 0 or step < 5:
            ee_pos = sim.get_end_effector_position()[:3]
            print(
                f"  Step {step:3d}: EE=({ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}) "
                f"Grip={target_grip_phys:.3f}"
            )

    # ---- Cleanup ----
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
