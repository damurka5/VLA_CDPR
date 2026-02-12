"""
  python tools/feature_probe_cdpr_v2_llm.py \
    --base-ckpt moojink/openvla-7b-oft-finetuned-libero-spatial \
    --adapter-path /root/repo/VLA_CDPR/oft_cdpr_ckpts/cdpr_finetune_step10000_20260210-080822_sbs700/vla_cdpr_adapter \
    --action-head-path /root/repo/VLA_CDPR/oft_cdpr_ckpts/cdpr_finetune_step10000_20260210-080822_sbs700/action_head_cdpr.pt \
    --catalog /root/repo/CDPR-Dataset/cdpr_dataset/datasets/cdpr_scene_catalog.yaml \
    --scene desk --object apple --task-name put_into_bowl \
    --num-probes 10 --vary both --out-dir probes_cdpr_v3
"""

#!/usr/bin/env python3
import os, sys, re, json, time, argparse
from pathlib import Path
from collections import defaultdict
import hashlib

import numpy as np
from PIL import Image
import torch

# ---- Apply cumsum patch ----
_orig_cumsum = torch.cumsum
def cumsum_bool_safe(input, dim, *args, **kwargs):
    if isinstance(input, torch.Tensor) and input.dtype == torch.bool:
        input = input.to(torch.int64)
    return _orig_cumsum(input, dim, *args, **kwargs)
torch.cumsum = cumsum_bool_safe

import yaml
from peft import PeftModel

VLA_CDPR_ROOT = "/root/repo/VLA_CDPR"
OPENVLA_OFT_ROOT = "/root/repo/openvla-oft"
LIBERO_ROOT = "/root/repo/LIBERO"
CDPR_DATASET_ROOT = "/root/repo/CDPR-Dataset"
for p in [VLA_CDPR_ROOT, OPENVLA_OFT_ROOT, LIBERO_ROOT, CDPR_DATASET_ROOT]:
    if p not in sys.path:
        sys.path.append(p)
os.environ["PYTHONPATH"] = VLA_CDPR_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

from cdpr_mujoco.headless_cdpr_egl import HeadlessCDPRSimulation

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

from cdpr_dataset.generate_cdpr_dataset import build_wrapper_if_needed
from cdpr_dataset.synthetic_tasks import task_language, place_objects_non_overlapping

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def sha1_tensor(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    return sha1_bytes(t.numpy().tobytes())

def sha1_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return sha1_bytes(a.tobytes())

def short(x, n=12):
    return x[:n]

def summarize_tensor(name, t):
    t_cpu = t.detach().float().cpu()
    return {
        "name": name,
        "shape": list(t_cpu.shape),
        "mean": float(t_cpu.mean().item()),
        "std": float(t_cpu.std(unbiased=False).item()),
        "min": float(t_cpu.min().item()),
        "max": float(t_cpu.max().item()),
        "sha1": sha1_tensor(t_cpu),
    }

def summarize_np(name, a):
    a = np.asarray(a)
    return {
        "name": name,
        "shape": list(a.shape),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "sha1": sha1_array(a),
    }

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

def save_png(arr, path):
    Image.fromarray(arr).save(path)
    
def find_last_llm_block(vla):
    # try to locate layers and pick last
    layers = None
    llm = getattr(vla, "llm", None) or getattr(vla, "language_model", None)
    # common: vla.llm.model.layers
    for path in ["llm.model.layers", "model.layers", "transformer.h", "llm.transformer.h"]:
        cur = vla
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False; break
            cur = getattr(cur, part)
        if ok:
            layers = cur
            break
    if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
        return layers[-1]
    return None

def find_final_norm(vla):
    for name, mod in vla.named_modules():
        if re.search(r"(llm\.)?(model\.)?norm$", name):
            return name, mod
    return None, None

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
    def _first_tensor(x):
        if torch.is_tensor(x):
            return x
        if isinstance(x, (list, tuple)):
            for t in x:
                if torch.is_tensor(t):
                    return t
        if isinstance(x, dict):
            for v in x.values():
                if torch.is_tensor(v):
                    return v
        return None

    def add_fwd(self, module, tag, take="output"):
        def _hook(mod, inp, out):
            x = out if take == "output" else inp
            t = self._first_tensor(x)
            if t is None:
                return
            self.data[tag].append(t.detach().to("cpu", torch.float32))
        self.handles.append(module.register_forward_hook(_hook))

    def add_input0(self, module, tag):
        def _hook(mod, inp, out):
            if not inp or not torch.is_tensor(inp[0]):
                return
            self.data[tag].append(inp[0].detach().to("cpu", torch.float32))
        self.handles.append(module.register_forward_hook(_hook))

def find_first_llm_block(vla):
    """
    Robust-ish: find first transformer block module.
    Works across common HF LLaMA-style and similar naming.
    """
    # common candidates
    candidates = []
    for name, mod in vla.named_modules():
        # LLaMA-ish: model.layers.0
        if re.search(r"(model\.)?layers\.0$", name):
            candidates.append((name, mod))
        # Some use "transformer.h.0"
        if re.search(r"(transformer\.)?h\.0$", name):
            candidates.append((name, mod))
        # Some use "llm.model.layers.0"
        if re.search(r"llm\.(model\.)?layers\.0$", name):
            candidates.append((name, mod))
    if candidates:
        # pick shortest name (usually the real one)
        candidates = sorted(candidates, key=lambda x: len(x[0]))
        return candidates[0][0], candidates[0][1]

    # fallback: find any module name ending in ".0" with "layer" in name
    for name, mod in vla.named_modules():
        if name.endswith(".0") and ("layer" in name or "block" in name or "h" in name):
            return name, mod
    return None, None

def resolve_vision_modules(vla):
    backbone = getattr(vla, "vision_backbone", None) or getattr(vla, "vision_tower", None)
    projector = getattr(vla, "vision_projector", None) or getattr(vla, "mm_projector", None) or getattr(vla, "projector", None)

    if backbone is None or projector is None:
        # fallback by pattern
        bb_name = None
        pr_name = None
        for name, mod in vla.named_modules():
            if bb_name is None and re.search(r"vision_backbone$|vision_tower$|visual_encoder$|vision_encoder$", name):
                bb_name = name; backbone = mod
            if pr_name is None and re.search(r"(mm_|multi_modal_)?projector$|vision_projector$|visual_projector$", name):
                pr_name = name; projector = mod
        return backbone, projector, {"backbone_name": bb_name, "projector_name": pr_name}
    return backbone, projector, {"backbone_name": type(backbone).__name__, "projector_name": type(projector).__name__}

def dump_probe(out_dir, meta, tensors, overview_rgb, wrist_rgb):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    save_png(overview_rgb, out_dir / "overview.png")
    save_png(wrist_rgb, out_dir / "wrist.png")

    np_pack = {}
    for k, v in tensors.items():
        if torch.is_tensor(v):
            np_pack[k] = v.numpy()
        else:
            np_pack[k] = v
    np.savez_compressed(out_dir / "tensors.npz", **np_pack)

def infer_concat_tokens(action_in, llm_seq):
    # action_in: [B, 20480] ; llm_seq: [B, T, 4096]
    B, T, D = llm_seq.shape
    action_in = action_in.view(B, -1, D)  # [B, 5, 4096]
    # for each of the 5 chunks, find the token index with max cosine
    idxs = []
    for j in range(action_in.shape[1]):
        a = torch.nn.functional.normalize(action_in[:, j, :], dim=-1)      # [B, D]
        s = torch.nn.functional.normalize(llm_seq, dim=-1)                  # [B, T, D]
        # cosine per token
        cos = (s * a[:, None, :]).sum(-1)                                   # [B, T]
        idx = cos.argmax(dim=1)                                             # [B]
        idxs.append(idx.cpu().tolist())
    return idxs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="moojink/openvla-7b-oft-finetuned-libero-spatial")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--no-adapter", action="store_true", default=False)
    ap.add_argument("--action-head-path", required=True)

    ap.add_argument("--catalog", default="/root/repo/CDPR-Dataset/cdpr_dataset/datasets/cdpr_scene_catalog.yaml")
    ap.add_argument("--scene", default="desk")
    ap.add_argument("--object", default="milk")
    ap.add_argument("--task-name", default="put_into_bowl")
    ap.add_argument("--instr", default=None)

    ap.add_argument("--num-probes", type=int, default=10)
    ap.add_argument("--vary", choices=["placement", "instruction", "both", "none"], default="placement")
    ap.add_argument("--out-dir", default="probes_cdpr_v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grip-range", default="0.0,0.06")
    ap.add_argument("--chunk-length", type=int, default=None)
    ap.add_argument("--center-crop", action="store_true", default=True)
    ap.add_argument("--no-center-crop", dest="center_crop", action="store_false")
    args = ap.parse_args()

    def parse_pair(s):
        lo, hi = map(float, s.split(","))
        return (lo, hi)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    gr_range = parse_pair(args.grip_range)

    # build wrapper xml
    cfg_yaml = yaml.safe_load(Path(args.catalog).read_text())
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
        raise SystemExit(f"scene '{args.scene}' not found")

    scene_name = scene_entry["name"]
    object_names = scene_entry.get("objects", [])
    main_obj = args.object if args.object else (object_names[0] if object_names else "target_object")

    wrapper_xml = build_wrapper_if_needed(
        scene_name,
        object_names,
        scene_z=defaults.get("scene_z", -0.85),
        ee_start=defaults.get("ee_start", (0.0, 0.0, 0.45)),
        table_z=defaults.get("table_z", 0.15),
        settle_time=defaults.get("settle_time", 1.0),
    )
    xml_path = str(wrapper_xml)
    base_instr = args.instr if args.instr is not None else task_language(args.task_name, main_obj)

    # openvla cfg
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

    # sim
    sim = HeadlessCDPRSimulation(xml_path=xml_path, output_dir=str(out_root / "sim_frames"))
    sim.initialize()

    # model
    vla_base = get_vla(cfg).eval()
    if (not args.no_adapter) and args.adapter_path and os.path.isdir(args.adapter_path):
        vla = PeftModel.from_pretrained(vla_base, args.adapter_path).eval()
    else:
        vla = vla_base

    import inspect
    print("vla type:", type(vla))
    print("vla.predict_action defined in:", inspect.getfile(vla.predict_action))
    print("vla.predict_action signature:", inspect.signature(vla.predict_action))


    device = next(vla.parameters()).device
    print("VLA device:", device, "ACTION_DIM:", ACTION_DIM)

    if hasattr(vla, "vision_backbone"):
        try:
            vla.vision_backbone.num_images_in_input = cfg.num_images_in_input
        except Exception:
            pass

    # -------------------------
    # Action head: load weights FIRST, then move to device/dtype, then run tests.
    # -------------------------
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    # action_head = action_head.to(device=device, dtype=torch.float32).eval()
    head_mod = getattr(action_head, "model", action_head)

    ckpt = torch.load(args.action_head_path, map_location="cpu")

    def sha1_state_dict(sd):
        h = hashlib.sha1()
        for k in sorted(sd.keys()):
            v = sd[k]
            if torch.is_tensor(v):
                v = v.detach().cpu().contiguous()
                h.update(k.encode("utf-8"))
                h.update(v.numpy().tobytes())
        return h.hexdigest()
    
    base_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    base_sd = getattr(base_head, "model", base_head).state_dict()

    loaded_sd = getattr(action_head, "model", action_head).state_dict()

    print("base head sha1:  ", sha1_state_dict(base_sd))
    print("loaded head sha1:", sha1_state_dict(loaded_sd))

    # Also: max abs diff
    mx = 0.0
    for k in base_sd:
        if k in loaded_sd:
            d = (base_sd[k].float() - loaded_sd[k].float()).abs().max().item()
            mx = max(mx, d)
    print("max|base-loaded|:", mx)

    # print("Runtime action_head sd sha1:", sha1_sd(getattr(action_head, "model", action_head).state_dict()))


    # prefix fix helpers
    def strip_model_prefix(sd):
        out = {}
        for k, v in sd.items():
            if k.startswith("model."):
                out[k[6:]] = v
            else:
                out[k] = v
        return out

    def add_model_prefix(sd):
        return {("model." + k if not k.startswith("model.") else k): v for k, v in sd.items()}

    head_keys = list(head_mod.state_dict().keys())
    ckpt_keys = list(ckpt.keys())
    head_expects_model_prefix = any(k.startswith("model.") for k in head_keys)
    ckpt_has_model_prefix = any(k.startswith("model.") for k in ckpt_keys)

    if head_expects_model_prefix and not ckpt_has_model_prefix:
        ckpt = add_model_prefix(ckpt)
    elif (not head_expects_model_prefix) and ckpt_has_model_prefix:
        ckpt = strip_model_prefix(ckpt)

    # Load now (still on CPU is fine)
    head_mod.load_state_dict(ckpt, strict=True)

    # Move the WHOLE head to the same device/dtype as VLA
    action_head = action_head.to(device=device, dtype=torch.bfloat16).eval()

    # Re-resolve head_mod after to(), in case references changed
    head_mod = getattr(action_head, "model", action_head)

    # Optional but very useful: assert all head params are on the right device
    def assert_all_on_device(m, device_):
        bad = [(n, p.device) for n, p in m.named_parameters() if p.device != device_]
        if bad:
            print("⚠️ parameters not on expected device (showing up to 20):")
            for n, d in bad[:20]:
                print("  ", n, d)
            raise RuntimeError(f"{len(bad)} parameters not on {device_}")

    assert_all_on_device(action_head, device)

    # -------------------------
    # Constant-function checks (run on the exact callable module we will use)
    # -------------------------
    torch.manual_seed(0)

    # The callable we test: if head has .model use it, else use head itself
    mlp = getattr(action_head, "model", action_head)
    mlp = mlp.to(device=device, dtype=torch.bfloat16).eval()  # ensure callable is on device

    # Create tensors using param device/dtype (prevents cpu/cuda mismatch)
    p0 = next(mlp.parameters())
    dev, dt = p0.device, p0.dtype

    B = 1
    input_dim = getattr(action_head, "input_dim", 4096)

    x1 = torch.randn(B * NUM_ACTIONS_CHUNK, ACTION_DIM * input_dim, device=dev, dtype=dt)
    x2 = torch.randn(B * NUM_ACTIONS_CHUNK, ACTION_DIM * input_dim, device=dev, dtype=dt) * 10.0 + 100.0

    with torch.no_grad():
        y1 = mlp(x1)
        y2 = mlp(x2)

    # print("mlp device/dtype:", dev, dt)
    # print("mlp y1 sha1:", sha1_tensor(y1.float().cpu()))
    # print("mlp y2 sha1:", sha1_tensor(y2.float().cpu()))
    # print("abs diff mean:", float((y1.float() - y2.float()).abs().mean().cpu()))
    # print("abs diff max :", float((y1.float() - y2.float()).abs().max().cpu()))

    # Param stats / bias checks
    def param_stats(m):
        s = []
        for n, p in m.named_parameters():
            pf = p.detach().float()
            s.append((n, float(pf.abs().mean()), float(pf.abs().max()), float(pf.norm())))
        return s

    # Activation saturation / dead path check
    acts = {}
    hs = []

    called = {"n": 0}
    def hook(m, inp, out):
        called["n"] += 1
        out_t = out if torch.is_tensor(out) else out[0]
        print("[DBG] action_head.forward called; out mean:", float(out_t.float().mean().cpu()), flush=True)

    h = getattr(action_head, "model", action_head).register_forward_hook(hook)

    

    # -------------------------
    # Wrap action_head.predict_action to capture wrapper input/output (not just MLP)
    # -------------------------
    orig_predict = action_head.predict_action
    last_pred = {}

    def predict_wrap(actions_hidden_states):
        # capture wrapper input (B, 40, 4096)-ish
        # actions_hidden_states = actions_hidden_states.to(dtype=torch.float32)
        if torch.is_tensor(actions_hidden_states):
            last_pred["ahs"] = actions_hidden_states.detach().to("cpu", torch.float32)
        y = orig_predict(actions_hidden_states)
        if torch.is_tensor(y):
            last_pred["y"] = y.detach().to("cpu", torch.float32)
        return y

    action_head.predict_action = predict_wrap

    # -------------------------
    # Processor debug wrap
    # -------------------------
    processor = get_processor(cfg)

    _orig_proc_call = processor.__call__
    proc_debug_last = {}

    def _proc_call_debug(*p_args, **p_kwargs):
        out = _orig_proc_call(*p_args, **p_kwargs)
        try:
            d = dict(out)
        except Exception:
            d = out
        snap = {}
        for k, v in d.items():
            if torch.is_tensor(v):
                snap[k] = v.detach().cpu()
        proc_debug_last.clear()
        proc_debug_last.update(snap)
        return out

    processor.__call__ = _proc_call_debug

    # -------------------------
    # Proprio projector (ensure dtype matches what VLA expects)
    # -------------------------
    obs0, proprio_dim, _, _ = make_observation(sim, base_instr, gripper_range=gr_range)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=proprio_dim)
    proprio_projector = proprio_projector.to(device=device, dtype=torch.bfloat16).eval()
    # print("⚠️ proprio_projector is random unless you load trained weights!")

    # -------------------------
    # Resolve modules + hooks
    # -------------------------
    vision_backbone, projector, vis_report = resolve_vision_modules(vla)
    llm0_name, llm0 = find_first_llm_block(vla)

    # print("vision module report:", vis_report)
    # print("llm0:", llm0_name, type(llm0).__name__ if llm0 is not None else None)

    if vision_backbone is None or projector is None:
        raise SystemExit("Could not resolve vision backbone/projector; print named_modules and adjust patterns.")
    if llm0 is None:
        raise SystemExit("Could not find first LLM block; need to adjust find_first_llm_block patterns.")

    hooks = HookBank()
    hooks.add_fwd(vision_backbone, "vision_pre", take="output")
    hooks.add_fwd(projector, "vision_post", take="output")

    hooks.add_input0(llm0, "llm_block0_in")
    hooks.add_fwd(llm0, "llm_block0_out", take="output")

    # Capture MLP input/output
    hooks.add_input0(head_mod, "action_head_in")
    hooks.add_fwd(head_mod, "action_head_out", take="output")

    llm_last = find_last_llm_block(vla)
    norm_name, norm_mod = find_final_norm(vla)
    if llm_last is not None:
        hooks.add_fwd(llm_last, "llm_last_out", take="output")
    if norm_mod is not None:
        hooks.add_fwd(norm_mod, "llm_norm_out", take="output")

    # -------------------------
    # Probe loop
    # -------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    instr_pool = [
        base_instr,
        f"Pick up the {main_obj}",
        f"Move the {main_obj} left",
        f"Move the {main_obj} right",
        f"Put the {main_obj} into the bowl",
    ]

    prev_ah_in = None
    prev = None
    prev_ahs = None
    prev_wrapper_y = None

    for i in range(args.num_probes):
        hooks.clear()

        if args.vary in ["instruction", "both"]:
            instr = instr_pool[i % len(instr_pool)]
        else:
            instr = base_instr

        placement_note = None
        if args.vary in ["placement", "both"]:
            try:
                real_obj = getattr(sim, "get_object_body_name", lambda: None)() or main_obj
                xy_bounds = ((-0.12, 0.12), (-0.12, 0.12), 0.10)
                place_objects_non_overlapping(sim, [real_obj], xy_bounds, min_gap=0.03)
                placement_note = f"placed {real_obj}"
            except Exception as e:
                placement_note = f"placement failed: {e}"

        obs, _, full_rgb, wrist_rgb = make_observation(sim, instr, gripper_range=gr_range)

        t0 = time.time()
        with torch.no_grad():
            acts_list = get_vla_action(cfg, vla, processor, obs, instr, action_head, proprio_projector)

            print("action_head object id:", id(action_head))

            # Processor-produced tensors (actual model inputs)
            proc_summ = {}
            for k in ["input_ids", "attention_mask", "pixel_values", "images", "image", "vision_pixel_values"]:
                if k in proc_debug_last:
                    proc_summ[k] = summarize_tensor(k, proc_debug_last[k])

            decoded_tail = None
            if "input_ids" in proc_debug_last:
                ids = proc_debug_last["input_ids"][0].tolist()
                tail = ids[-80:]
                try:
                    decoded_tail = processor.tokenizer.decode(tail, skip_special_tokens=False)
                except Exception:
                    decoded_tail = None

        dt = time.time() - t0
        acts = np.asarray(acts_list, dtype=np.float32)

        def last(tag):
            return hooks.data[tag][-1] if tag in hooks.data and hooks.data[tag] else None

        tensors = {"acts": acts}
        for tag in [
            "vision_pre",
            "vision_post",
            "llm_block0_in",
            "llm_block0_out",
            "action_head_in",
            "action_head_out",
            "llm_last_out",
            "llm_norm_out",
        ]:
            t = last(tag)
            if t is not None:
                tensors[tag] = t

        # Wrapper-level captures (from predict_wrap)
        if "ahs" in last_pred:
            tensors["action_head_wrapper_in"] = last_pred["ahs"]
        if "y" in last_pred:
            tensors["action_head_wrapper_out"] = last_pred["y"]

        # in your probe loop, after you capture action_head_wrapper_in (1,40,4096)
        ahs = tensors["action_head_wrapper_in"].float().cpu()  # (1,40,4096)
        v = ahs.reshape(-1)
        
        if prev is not None:
            diff = (v - prev).abs()
            print("mean|diff|:", float(diff.mean()), "max|diff|:", float(diff.max()))
            cos = torch.nn.functional.cosine_similarity(v, prev, dim=0)
            print("cos_sim:", float(cos))
        prev = v

        # Compute diffs across probes (mean/max)
        meta = {}
        if "action_head_in" in tensors and torch.is_tensor(tensors["action_head_in"]):
            ah_in = tensors["action_head_in"]
            if prev_ah_in is not None:
                d = (ah_in - prev_ah_in).abs()
                meta["ah_in_absdiff_mean"] = float(d.mean())
                meta["ah_in_absdiff_max"] = float(d.max())
            prev_ah_in = ah_in

        if "action_head_wrapper_in" in tensors and torch.is_tensor(tensors["action_head_wrapper_in"]):
            ahs = tensors["action_head_wrapper_in"]
            if prev_ahs is not None:
                d = (ahs - prev_ahs).abs()
                meta["ahs_absdiff_mean"] = float(d.mean())
                meta["ahs_absdiff_max"] = float(d.max())
            prev_ahs = ahs

        if "action_head_wrapper_out" in tensors and torch.is_tensor(tensors["action_head_wrapper_out"]):
            wy = tensors["action_head_wrapper_out"]
            if prev_wrapper_y is not None:
                d = (wy - prev_wrapper_y).abs()
                meta["wrapper_y_absdiff_mean"] = float(d.mean())
                meta["wrapper_y_absdiff_max"] = float(d.max())
            prev_wrapper_y = wy

        # Token-match inference (best-effort)
        ah_in = tensors.get("action_head_in", None)
        llm_out = tensors.get("llm_norm_out", None)
        if llm_out is None:
            llm_out = tensors.get("llm_last_out", None)
        if llm_out is None:
            llm_out = tensors.get("llm_block0_out", None)

        concat_idxs = None
        if (
            torch.is_tensor(ah_in)
            and torch.is_tensor(llm_out)
            and llm_out.ndim == 3
            and ah_in.ndim == 2
            and ah_in.shape[1] % llm_out.shape[-1] == 0
        ):
            try:
                concat_idxs = infer_concat_tokens(ah_in, llm_out)
            except Exception:
                concat_idxs = None

        raw_full_sha1 = sha1_array(full_rgb)
        raw_wrist_sha1 = sha1_array(wrist_rgb)
        instr_sha1 = sha1_bytes(instr.encode("utf-8"))

        meta.update(
            {
                "i": i,
                "instr": instr,
                "placement_note": placement_note,
                "dt_sec": dt,
                "acts_shape": list(acts.shape),
                "instr_sha1": instr_sha1,
                "raw_full_sha1": raw_full_sha1,
                "raw_wrist_sha1": raw_wrist_sha1,
                "proc": proc_summ,
                "decoded_tail": decoded_tail,
                "action_in_token_match_idxs": concat_idxs,
                "obs_task_description": obs.get("task_description", None),
                "instr_arg": instr,
                "state_sha1": sha1_array(obs["state"]),
                "state": obs["state"].tolist(),
            }
        )

        # sha1 for key tensors
        for k in [
            "action_head_in",
            "action_head_out",
            "action_head_wrapper_in",
            "action_head_wrapper_out",
            "llm_block0_out",
            "vision_post",
        ]:
            if k in tensors and torch.is_tensor(tensors[k]):
                meta[k + "_sha1"] = sha1_tensor(tensors[k])

        meta["acts_sha1"] = sha1_array(acts)
        meta["shapes"] = {k: (list(v.shape) if hasattr(v, "shape") else None) for k, v in tensors.items() if k != "acts"}

        out_dir = out_root / f"probe_{i:03d}"
        dump_probe(out_dir, meta, tensors, full_rgb, wrist_rgb)

        # print(
        #     f"[{i:03d}] dt={dt:.3f}s instr='{instr}' shapes:",
        #     {k: tuple(tensors[k].shape) for k in tensors if k != "acts"},
        # )

    hooks.remove()
    sim.cleanup()
    print("✅ done, saved to:", out_root.resolve())

if __name__ == "__main__":
    main()