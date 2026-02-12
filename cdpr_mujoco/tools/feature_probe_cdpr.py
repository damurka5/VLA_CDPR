#!/usr/bin/env python3
"""
tools/feature_probe_cdpr.py

Collects feature probes from OpenVLA-OFT on your CDPR MuJoCo sim:
- vision pre-projector features
- vision post-projector features
- action head input vector (policy embedding)

Saves:
  out_dir/
    probe_000/
      meta.json
      overview.png
      wrist.png
      tensors.npz
    probe_001/
      ...

Usage example:
  python tools/feature_probe_cdpr.py \
    --base-ckpt moojink/openvla-7b-oft-finetuned-libero-spatial \
    --adapter-path /root/repo/VLA_CDPR/oft_cdpr_ckpts/cdpr_finetune_step10000_20260210-080822_sbs700/vla_cdpr_adapter \
    --action-head-path /root/repo/VLA_CDPR/oft_cdpr_ckpts/cdpr_finetune_step10000_20260210-080822_sbs700/action_head_cdpr.pt \
    --catalog /root/repo/CDPR-Dataset/cdpr_dataset/datasets/cdpr_scene_catalog.yaml \
    --scene desk --object milk --task-name put_into_bowl \
    --num-probes 10 --vary placement --out-dir probes_cdpr
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch

# ---- Apply cumsum patch (as in your runner) ----
_orig_cumsum = torch.cumsum
def cumsum_bool_safe(input, dim, *args, **kwargs):
    if isinstance(input, torch.Tensor) and input.dtype == torch.bool:
        input = input.to(torch.int64)
    return _orig_cumsum(input, dim, *args, **kwargs)
torch.cumsum = cumsum_bool_safe
print("🔧 Applied cumsum patch")

import yaml
from peft import PeftModel

# ============================================================
# Paths (match your environment)
# ============================================================
VLA_CDPR_ROOT = "/root/repo/VLA_CDPR"
OPENVLA_OFT_ROOT = "/root/repo/openvla-oft"
LIBERO_ROOT = "/root/repo/LIBERO"
CDPR_DATASET_ROOT = "/root/repo/CDPR-Dataset"

for p in [VLA_CDPR_ROOT, OPENVLA_OFT_ROOT, LIBERO_ROOT, CDPR_DATASET_ROOT]:
    if p not in sys.path:
        sys.path.append(p)

os.environ["PYTHONPATH"] = VLA_CDPR_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

# ============================================================
# Local sim import (your class)
# ============================================================
from cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation

# ============================================================
# OpenVLA-OFT imports
# ============================================================
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

# ============================================================
# Optional dataset helpers (placement + instruction)
# ============================================================
try:
    from cdpr_dataset.generate_cdpr_dataset import build_wrapper_if_needed
    from cdpr_dataset.synthetic_tasks import task_language, place_objects_non_overlapping
    HAVE_DATASET_HELPERS = True
    print("✅ Imported CDPR dataset helpers")
except Exception as e:
    HAVE_DATASET_HELPERS = False
    build_wrapper_if_needed = None
    task_language = None
    place_objects_non_overlapping = None
    print(f"⚠️ CDPR dataset helpers not available: {e}")

# ============================================================
# Observation helper (copied from your runner)
# ============================================================
def make_observation(sim, task_text, gripper_range=(0.0, 0.06)):
    full_rgb = sim.capture_frame(sim.overview_cam, "overview")
    wrist_rgb = sim.capture_frame(sim.ee_cam, "ee_camera")

    ee = sim.get_end_effector_position().astype(np.float32)
    yaw = float(sim.get_yaw()) if hasattr(sim, "get_yaw") else 0.0
    grip_phys = float(getattr(sim, "get_gripper_opening", lambda: 0.03)())

    g_lo, g_hi = gripper_range
    grip_norm = (grip_phys - g_lo) / (g_hi - g_lo + 1e-9)
    grip_norm = np.clip(grip_norm, 0.0, 1.0)

    state = np.array([ee[0], ee[1], ee[2], yaw, grip_norm], dtype=np.float32)

    obs = {
        "full_image": np.ascontiguousarray(full_rgb),
        "wrist_image": np.ascontiguousarray(wrist_rgb),
        "state": state,
        "task_description": task_text,
    }
    return obs, state.size, full_rgb, wrist_rgb

# ============================================================
# Hook utilities
# ============================================================
def list_named_modules(model, max_lines=250):
    rows = []
    for name, mod in model.named_modules():
        rows.append((name, type(mod).__name__))
    rows = sorted(rows, key=lambda x: x[0])
    print(f"\n🔎 named_modules() total: {len(rows)} (showing up to {max_lines})")
    for i, (n, t) in enumerate(rows[:max_lines]):
        print(f"  {i:03d}  {n:60s}  {t}")
    if len(rows) > max_lines:
        print(f"  ... ({len(rows)-max_lines} more)")

def find_modules_by_name(model, patterns):
    hits = defaultdict(list)
    for name, mod in model.named_modules():
        for p in patterns:
            if re.search(p, name):
                hits[p].append((name, mod))
    return hits

class HookBank:
    def __init__(self):
        self.handles = []
        self.data = defaultdict(list)

    def clear(self):
        self.data.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    @staticmethod
    def _to_tensor_list(x):
        if torch.is_tensor(x):
            return [x]
        if isinstance(x, (list, tuple)):
            return [t for t in x if torch.is_tensor(t)]
        if isinstance(x, dict):
            return [v for v in x.values() if torch.is_tensor(v)]
        return []

    def add_forward_hook(self, module, tag, take="output"):
        def _hook(mod, inp, out):
            x = out if take == "output" else inp
            tensors = self._to_tensor_list(x)
            if len(tensors) == 0:
                return
            # store FIRST tensor only (most useful)
            t0 = tensors[0].detach().to("cpu", torch.float32)
            self.data[tag].append(t0)
        self.handles.append(module.register_forward_hook(_hook))

    def add_input_hook(self, module, tag):
        def _hook(mod, inp, out):
            if not inp or (not torch.is_tensor(inp[0])):
                return
            t0 = inp[0].detach().to("cpu", torch.float32)
            self.data[tag].append(t0)
        self.handles.append(module.register_forward_hook(_hook))

# ============================================================
# Robust module selection for vision backbone & projector
# ============================================================
def resolve_vision_modules(vla):
    """
    Returns:
      vision_backbone_module, projector_module, report_dict
    Tries attributes first, then name patterns.
    """
    report = {"attr_candidates": {}, "pattern_candidates": {}}

    # 1) Attribute-based (fast + usually correct)
    attr_backbone_names = ["vision_backbone", "vision_tower", "visual_encoder", "vision_encoder"]
    attr_proj_names = ["vision_projector", "mm_projector", "projector", "multi_modal_projector", "visual_projector"]

    backbone = None
    projector = None

    for an in attr_backbone_names:
        if hasattr(vla, an):
            backbone = getattr(vla, an)
            report["attr_candidates"]["backbone"] = an
            break

    for an in attr_proj_names:
        if hasattr(vla, an):
            projector = getattr(vla, an)
            report["attr_candidates"]["projector"] = an
            break

    # 2) Fallback: name-based search inside named_modules
    patterns_backbone = [r"vision_backbone$", r"vision_tower$", r"visual_encoder$", r"vision_encoder$"]
    patterns_proj = [r"(mm_|multi_modal_)?projector$", r"vision_projector$", r"visual_projector$"]

    hits = find_modules_by_name(vla, patterns_backbone + patterns_proj)
    report["pattern_candidates"] = {k: [n for (n, _) in v] for k, v in hits.items()}

    if backbone is None:
        # Choose first matching module name if exists
        for p in patterns_backbone:
            if p in hits and len(hits[p]) > 0:
                backbone = hits[p][0][1]
                report["pattern_selected_backbone"] = hits[p][0][0]
                break

    if projector is None:
        for p in patterns_proj:
            if p in hits and len(hits[p]) > 0:
                projector = hits[p][0][1]
                report["pattern_selected_projector"] = hits[p][0][0]
                break

    return backbone, projector, report

# ============================================================
# Save helpers
# ============================================================
def save_png(arr_uint8, path):
    Image.fromarray(arr_uint8).save(path)

def dump_probe(out_dir, meta, tensors_dict, overview_rgb, wrist_rgb):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    save_png(overview_rgb, out_dir / "overview.png")
    save_png(wrist_rgb, out_dir / "wrist.png")

    # Convert tensors to numpy
    np_pack = {}
    for k, v in tensors_dict.items():
        if torch.is_tensor(v):
            np_pack[k] = v.numpy()
        else:
            np_pack[k] = v
    np.savez_compressed(out_dir / "tensors.npz", **np_pack)

# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="moojink/openvla-7b-oft-finetuned-libero-spatial")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--no-adapter", action="store_true", default=False)

    ap.add_argument("--action-head-path", required=True)

    # dataset-style
    ap.add_argument("--catalog", default="/root/repo/CDPR-Dataset/cdpr_dataset/datasets/cdpr_scene_catalog.yaml")
    ap.add_argument("--scene", default="desk")
    ap.add_argument("--object", default="milk")
    ap.add_argument("--task-name", default="put_into_bowl")
    ap.add_argument("--instr", default=None)

    # probe control
    ap.add_argument("--num-probes", type=int, default=10)
    ap.add_argument("--vary", choices=["placement", "instruction", "both", "none"], default="placement")

    # sim + stats
    ap.add_argument("--grip-range", default="0.0,0.06")
    ap.add_argument("--center-crop", action="store_true", default=True)
    ap.add_argument("--no-center-crop", dest="center_crop", action="store_false")

    ap.add_argument("--chunk-length", type=int, default=None)

    ap.add_argument("--out-dir", default="probes_cdpr")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    def parse_pair(s):
        lo, hi = map(float, s.split(","))
        return (lo, hi)

    gr_range = parse_pair(args.grip_range)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not HAVE_DATASET_HELPERS:
        raise SystemExit("CDPR dataset helpers not importable. Ensure CDPR-Dataset is on sys.path.")

    # ------------------------------------------------------------
    # Build wrapper XML using catalog (same logic as your runner)
    # ------------------------------------------------------------
    with open(args.catalog, "r") as f:
        cfg_yaml = yaml.safe_load(f)

    defaults = cfg_yaml.get("defaults", {})
    scenes_cfg = cfg_yaml.get("scenes", [])

    scene_entry = None
    for entry in scenes_cfg:
        if isinstance(entry, dict) and entry.get("name") == args.scene:
            scene_entry = entry
            break
        if not isinstance(entry, dict) and str(entry) == args.scene:
            scene_entry = {"name": str(entry), "objects": []}
            break
    if scene_entry is None:
        raise SystemExit(f"Scene '{args.scene}' not found in catalog.")

    scene_name = scene_entry["name"]
    object_names = scene_entry.get("objects", [])
    main_obj = args.object if args.object else (object_names[0] if object_names else "target_object")

    scene_z = defaults.get("scene_z", -0.85)
    ee_start = defaults.get("ee_start", (0.0, 0.0, 0.45))
    table_z = defaults.get("table_z", 0.15)
    settle_t = defaults.get("settle_time", 1.0)

    wrapper_xml = build_wrapper_if_needed(
        scene_name,
        object_names,
        scene_z=scene_z,
        ee_start=ee_start,
        table_z=table_z,
        settle_time=settle_t,
    )
    xml_path = str(wrapper_xml)
    print(f"🧱 Wrapper XML: {xml_path}")

    # Instruction
    if args.instr is not None:
        base_instr = args.instr
    else:
        base_instr = task_language(args.task_name, main_obj)
    print(f"🗣 Base instruction: {base_instr}")

    # ------------------------------------------------------------
    # Build OpenVLA cfg
    # ------------------------------------------------------------
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
        unnorm_key="cdpr_local",
    )

    # ------------------------------------------------------------
    # Init sim
    # ------------------------------------------------------------
    sim = HeadlessCDPRSimulation(xml_path=xml_path, output_dir=str(out_root / "sim_frames"))
    sim.initialize()

    # ------------------------------------------------------------
    # Load base model + optional adapter
    # ------------------------------------------------------------
    print("\n🤖 Loading VLA base...")
    vla_base = get_vla(cfg)
    vla_base.eval()

    if (not args.no_adapter) and args.adapter_path and os.path.isdir(args.adapter_path):
        print(f"🔧 Loading PEFT adapter from: {args.adapter_path}")
        vla = PeftModel.from_pretrained(vla_base, args.adapter_path)
        vla.eval()
    else:
        vla = vla_base

    device = next(vla.parameters()).device
    print(f"✅ VLA device: {device}")

    # Ensure vision backbone knows multi-image count
    if hasattr(vla, "vision_backbone"):
        try:
            vla.vision_backbone.num_images_in_input = cfg.num_images_in_input
        except Exception as e:
            print(f"⚠️ Could not set num_images_in_input: {e}")

    # ------------------------------------------------------------
    # Load action head
    # ------------------------------------------------------------
    if not os.path.exists(args.action_head_path):
        raise SystemExit(f"Action head not found: {args.action_head_path}")

    print("\n🎯 Loading action head...")
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    head_mod = getattr(action_head, "model", action_head)

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
    action_head = action_head.to(device=device, dtype=torch.bfloat16).eval()
    head_mod = getattr(action_head, "model", action_head)

    print(f"✅ Action head loaded. ACTION_DIM constant: {ACTION_DIM}")

    # ------------------------------------------------------------
    # Processor + proprio projector (NOTE: this is randomly init unless you load it!)
    # ------------------------------------------------------------
    print("\n⚙️ Loading processor + proprio projector...")
    processor = get_processor(cfg)

    # Use one obs to infer proprio_dim
    obs0, proprio_dim, _, _ = make_observation(sim, base_instr, gripper_range=gr_range)

    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=proprio_dim)
    proprio_projector = proprio_projector.to(device=device, dtype=torch.bfloat16).eval()

    print(f"   proprio_dim={proprio_dim}")
    print("   ⚠️ proprio_projector is randomly initialized unless you load a trained checkpoint.")

    # ------------------------------------------------------------
    # Resolve vision backbone + projector modules
    # ------------------------------------------------------------
    print("\n🔎 Resolving vision modules...")
    vision_backbone, projector, report = resolve_vision_modules(vla)

    print("Module resolution report:")
    print(json.dumps(report, indent=2))

    if vision_backbone is None or projector is None:
        print("\n❌ Could not auto-resolve vision backbone/projector.")
        print("Helpful debug: dumping named modules (trimmed).")
        list_named_modules(vla, max_lines=220)
        raise SystemExit("Please locate the correct module names and update resolve_vision_modules().")

    print(f"✅ vision_backbone type: {type(vision_backbone).__name__}")
    print(f"✅ projector type:       {type(projector).__name__}")
    print("   (Projector is assumed pretrained in base OpenVLA; you did not train it.)")

    # ------------------------------------------------------------
    # Setup hooks
    # ------------------------------------------------------------
    hooks = HookBank()
    hooks.add_forward_hook(vision_backbone, tag="vision_pre", take="output")
    hooks.add_forward_hook(projector, tag="vision_post", take="output")
    hooks.add_input_hook(head_mod, tag="action_head_in")

    # ------------------------------------------------------------
    # Probe loop
    # ------------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    instr_pool = [
        base_instr,
        f"Pick up the {main_obj}.",
        f"Move the {main_obj} to the left.",
        f"Move the {main_obj} to the right.",
        f"Put the {main_obj} into the bowl.",
        f"Place the {main_obj} on the table.",
    ]

    for i in range(args.num_probes):
        probe_dir = out_root / f"probe_{i:03d}"
        print(f"\n================ PROBE {i:03d} ================")

        # decide instruction
        if args.vary in ["instruction", "both"]:
            instr = instr_pool[i % len(instr_pool)]
        else:
            instr = base_instr

        # placement randomization
        placement_ok = False
        placement_note = None
        if args.vary in ["placement", "both"]:
            try:
                real_obj = getattr(sim, "get_object_body_name", lambda: None)()
                if real_obj is None:
                    real_obj = main_obj
                xy_bounds = ((-0.12, 0.12), (-0.12, 0.12), 0.10)
                # call placement helper (should randomize internally or depend on sim RNG)
                place_objects_non_overlapping(sim, [real_obj], xy_bounds, min_gap=0.015)
                placement_ok = True
                placement_note = f"randomized via place_objects_non_overlapping({real_obj})"
            except Exception as e:
                placement_note = f"placement failed: {e}"

        # build obs
        hooks.clear()
        obs, _, full_rgb, wrist_rgb = make_observation(sim, instr, gripper_range=gr_range)

        # run one action query (this triggers the hooks)
        t0 = time.time()
        with torch.no_grad():
            acts = get_vla_action(cfg, vla, processor, obs, instr, action_head, proprio_projector)
        dt = time.time() - t0
        acts = np.asarray(acts, dtype=np.float32)

        # fetch last captured tensors
        def last_or_none(tag):
            return hooks.data[tag][-1] if (tag in hooks.data and len(hooks.data[tag]) > 0) else None

        t_vision_pre = last_or_none("vision_pre")
        t_vision_post = last_or_none("vision_post")
        t_head_in = last_or_none("action_head_in")

        # sanity prints
        print(f"Instruction: {instr}")
        print(f"Placement varied: {placement_ok} ({placement_note})")
        print(f"get_vla_action dt={dt:.3f}s  acts.shape={acts.shape}  acts[0]={acts[0] if len(acts)>0 else None}")
        print(f"vision_pre:  {None if t_vision_pre is None else tuple(t_vision_pre.shape)}")
        print(f"vision_post: {None if t_vision_post is None else tuple(t_vision_post.shape)}")
        print(f"head_in:     {None if t_head_in is None else tuple(t_head_in.shape)}")

        meta = {
            "i": i,
            "instr": instr,
            "placement_varied": placement_ok,
            "placement_note": placement_note,
            "action_dim_constant_ACTION_DIM": int(ACTION_DIM),
            "acts_shape": list(acts.shape),
            "dt_sec": dt,
        }

        tensors = {
            "acts": acts,
        }
        if t_vision_pre is not None:
            tensors["vision_pre"] = t_vision_pre
        if t_vision_post is not None:
            tensors["vision_post"] = t_vision_post
        if t_head_in is not None:
            tensors["action_head_in"] = t_head_in

        dump_probe(probe_dir, meta, tensors, full_rgb, wrist_rgb)

    hooks.remove()
    sim.cleanup()

    print(f"\n✅ Done. Probes saved under: {out_root.resolve()}")

if __name__ == "__main__":
    main()
