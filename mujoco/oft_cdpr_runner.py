#!/usr/bin/env python3
import os, sys, time
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import numpy as np
import mujoco as mj

# ---- repo local import of your sim ----
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
from headless_cdpr_egl import HeadlessCDPRSimulation  # your class

import sys, os
# Try to locate LIBERO if not installed as a package
if "libero" not in sys.modules:
    candidate = "/root/repo/LIBERO"
    if os.path.isdir(candidate):
        sys.path.append(candidate)


# ---- openvla-oft imports (run in openvla-oft env) ----
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import (
    get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # chunk length

import itertools
import copy

# ---------- auto-orient helpers ----------

def guess_goal_xy(sim, override=None):
    if override is not None:
        return np.array(override[:2], dtype=float)

    # Prefer placed LIBERO objects (prefixed in your scene_switcher as p0_, p1_, ...)
    names = []
    for bid in range(sim.model.nbody):
        nm = mj.mj_id2name(sim.model, mj.mjtObj.mjOBJ_BODY, bid)
        names.append(nm if nm is not None else "")
    cand_ids = [i for i,n in enumerate(names)
                if n.startswith(("p0_", "p1_", "p2_", "p3_", "p4_"))]

    if cand_ids:
        return sim.data.xpos[cand_ids[0], :2].copy()

    # Fallback: your red block ("target_object")
    try:
        return sim.get_target_position()[:2].copy()
    except:
        # Fallback to current EE XY if nothing else
        return sim.get_end_effector_position()[:2].copy()

def snapshot(sim):
    # capture full sim state + your controller target
    return {
        "qpos": sim.data.qpos.copy(),
        "qvel": sim.data.qvel.copy(),
        "ctrl": sim.data.ctrl.copy(),
        "target_pos": sim.target_pos.copy() if hasattr(sim, "target_pos") else None,
    }

def restore(sim, snap):
    sim.data.qpos[:] = snap["qpos"]
    sim.data.qvel[:] = snap["qvel"]
    sim.data.ctrl[:] = snap["ctrl"]
    if snap["target_pos"] is not None and hasattr(sim, "target_pos"):
        sim.target_pos = snap["target_pos"].copy()
    mj.mj_forward(sim.model, sim.data)  # recompute derived state

def apply_mapping(vec3, perm, sign):
    v = np.array(vec3, dtype=float)
    v = v[list(perm)]
    return v * np.array(sign, dtype=float)

def probe_mapping(sim, acts, perm, sign, xyz_bounds, k_xyz=0.35, steps=25, goal_xy_override=None):
    snap0 = snapshot(sim)
    xyz_t = sim.get_end_effector_position().copy()
    goal_xy = guess_goal_xy(sim, goal_xy_override)

    (xlo,xhi), (ylo,yhi), (zlo,zhi) = xyz_bounds
    d0 = np.linalg.norm(xyz_t[:2] - goal_xy)

    for i in range(min(steps, len(acts))):
        a = acts[i]
        dxyz_raw = np.array([np.clip(a[0], -1, 1),
                             np.clip(a[1], -1, 1),
                             np.clip(a[2], -1, 1)], dtype=float) * k_xyz
        dxyz = apply_mapping(dxyz_raw, perm, sign)

        xyz_t = xyz_t + dxyz
        xyz_t[0] = np.clip(xyz_t[0], xlo, xhi)
        xyz_t[1] = np.clip(xyz_t[1], ylo, yhi)
        xyz_t[2] = np.clip(xyz_t[2], zlo, zhi)

        sim.set_target_position(xyz_t)
        # a couple of inner steps so plant follows the new target
        for _ in range(2):
            sim.run_simulation_step(capture_frame=False)

    dT = np.linalg.norm(sim.get_end_effector_position()[:2] - goal_xy)
    restore(sim, snap0)
    return d0 - dT  # positive means "moved closer"

def auto_orient_actions(sim, request_chunk_fn, xyz_bounds):
    import itertools
    acts = np.array(request_chunk_fn(), dtype=float)
    if acts.ndim != 2 or acts.shape[1] < 7:
        return ((0,1,2), (1,1,1))  # fallback

    perms = list(itertools.permutations([0,1,2], 3))
    signs = [(sx,sy,sz) for sx in (1,-1) for sy in (1,-1) for sz in (1,-1)]

    best = ((0,1,2),(1,1,1)); best_gain = -1e9
    for p in perms:
        for s in signs:
            gain = probe_mapping(sim, acts, p, s, xyz_bounds, k_xyz=0.35, steps=20)
            if gain > best_gain:
                best, best_gain = (p,s), gain

    print(f"[auto-orient] picked perm={best[0]} sign={best[1]} (gain={best_gain:.3f})")
    return best

# ---------- mapping & utils ----------

def clamp(v, lo, hi):
    return float(max(lo, min(hi, v)))

def norm01(v, lo, hi): 
    return (np.clip(v, lo, hi) - lo) / (hi - lo + 1e-9)

def norm11_from_bounds(v, lo, hi):
    return 2.0 * norm01(v, lo, hi) - 1.0

def _maybe_unnorm_1d(val, lo, hi):
    # if it looks normalized, map from [-1,1] -> [lo,hi]
    if -1.2 <= val <= 1.2:
        return lo + 0.5*(val+1.0)*(hi-lo)
    return clamp(val, lo, hi)

def map_7d_to_cdpr5(act7, xyz_bounds, gripper_range):
    a = np.array(act7, dtype=float).flatten()
    if a.size < 7:
        a = np.pad(a, (0, 7-a.size))

    (xlo, xhi), (ylo, yhi), (zlo, zhi) = xyz_bounds

    x = _maybe_unnorm_1d(a[0], xlo, xhi)
    y = _maybe_unnorm_1d(a[1], ylo, yhi)
    z = _maybe_unnorm_1d(a[2], zlo, zhi)

    yaw_raw = float(a[3])
    yaw = yaw_raw if abs(yaw_raw) > 1.2 else yaw_raw * np.pi
    yaw = (yaw + np.pi) % (2*np.pi) - np.pi

    g = float(a[6])
    if -1.2 <= g <= 1.2:
        g = 0.5*(g+1.0)  # [0,1]
    g = clamp(g, 0.0, 1.0)
    gr_lo, gr_hi = gripper_range
    grip = gr_lo + g*(gr_hi-gr_lo)

    return np.array([x, y, z]), yaw, grip


# def map_7d_to_cdpr5(act7, xyz_bounds, gripper_range):
#     """
#     act7: [x,y,z,yaw,pitch,roll,gripper]
#     returns: xyz(3), yaw(float), grip_opening_m(float)
#     """
#     a = np.array(act7, dtype=float).flatten()
#     if a.size < 7:  # pad if needed
#         a = np.pad(a, (0, 7 - a.size))

#     # assume OpenVLA-OFT emits in WORLD units already; if normalized, add a remap here
#     (xlo, xhi), (ylo, yhi), (zlo, zhi) = xyz_bounds
#     x = clamp(a[0], xlo, xhi)
#     y = clamp(a[1], ylo, yhi)
#     z = clamp(a[2], zlo, zhi)

#     # yaw: allow either radians or [-1,1] normalized; if looks small, scale by pi
#     yaw_raw = float(a[3])
#     yaw = yaw_raw if abs(yaw_raw) > 1.2 else yaw_raw * np.pi
#     # wrap to [-pi, pi]
#     yaw = (yaw + np.pi) % (2*np.pi) - np.pi

#     # gripper: assume normalized [0..1] or [-1..1] → meters
#     g = float(a[6])
#     if -1.2 <= g <= 1.2:
#         g = 0.5 * (g + 1.0)  # map [-1,1]→[0,1]
#     g = clamp(g, 0.0, 1.0)
#     gr_lo, gr_hi = gripper_range
#     grip = gr_lo + g * (gr_hi - gr_lo)

#     return np.array([x, y, z]), yaw, grip

def map_7d_delta_to_cdpr5(act7, xyz_bounds, gripper_range,
                          k_xyz=0.25, k_yaw=np.pi*0.20, k_grip=0.5,
                          max_step_xyz=0.08, max_step_yaw=np.deg2rad(20), max_step_grip=0.02):
    """
    Interpret act7 as *deltas* in normalized [-1,1] space:
      dx,dy,dz in [-1,1]  -> meters via k_xyz (then clipped to max_step_xyz)
      dyaw in [-1,1]      -> radians via k_yaw (clipped)
      grip in [-1,1] or [0,1] -> delta open amount (meters)
    Returns (delta_xyz, delta_yaw, delta_grip).
    """
    a = np.array(act7, dtype=float).flatten()
    if a.size < 7:
        a = np.pad(a, (0, 7 - a.size))
    # normalize plausible inputs
    dx = float(np.clip(a[0], -1, 1)) * k_xyz
    dy = float(np.clip(a[1], -1, 1)) * k_xyz
    dz = float(np.clip(a[2], -1, 1)) * k_xyz
    dyaw = float(np.clip(a[3], -1, 1)) * k_yaw

    # gripper: allow [-1,1] → delta-fraction of range
    g = float(a[6])
    if -1.2 <= g <= 1.2:
        g = 0.5 * (g)  # scale down; feel free to tune
    dgrip = g * (gripper_range[1] - gripper_range[0]) * k_grip

    # per-step clamps (safety/comfort)
    dx = float(np.clip(dx, -max_step_xyz, max_step_xyz))
    dy = float(np.clip(dy, -max_step_xyz, max_step_xyz))
    dz = float(np.clip(dz, -max_step_xyz, max_step_xyz))
    dyaw = float(np.clip(dyaw, -max_step_yaw, max_step_yaw))
    dgrip = float(np.clip(dgrip, -max_step_grip, max_step_grip))

    return np.array([dx, dy, dz]), dyaw, dgrip


# def make_observation(sim, task_text: str):
#     """
#     Build OFT-style observation dict.
#     Images must be np.uint8 HxWx3 (RGB). Proprio is 8-D for the pretrained projector:
#       [ee_x, ee_y, ee_z, yaw, pitch=0, roll=0, dummy=0, gripper]
#     """
#     # ---- images as np.uint8 arrays (RGB) ----
#     full_rgb  = sim.capture_frame(sim.overview_cam, "overview")   # already HxWx3 uint8
#     wrist_rgb = sim.capture_frame(sim.ee_cam, "ee_camera")
#     # ensure dtype and contiguous memory
#     full_rgb  = np.ascontiguousarray(full_rgb, dtype=np.uint8)
#     wrist_rgb = np.ascontiguousarray(wrist_rgb, dtype=np.uint8)

#     # ---- proprio (8-D expected by OFT checkpoints) ----
#     ee   = sim.get_end_effector_position().astype(np.float32)  # (3,)
#     yaw  = float(getattr(sim, "get_yaw", lambda: 0.0)())
#     grip = float(getattr(sim, "get_gripper_opening", lambda: 0.03)())
#     proprio = np.array([ee[0], ee[1], ee[2], yaw, 0.0, 0.0, 0.0, grip], dtype=np.float32)

#     obs = {
#         "full_image":  full_rgb,   # <- NumPy, not PIL
#         "wrist_image": wrist_rgb,  # <- NumPy, not PIL
#         "state": proprio,
#         "task_description": task_text,
#     }
#     return obs, proprio.size

def make_observation(sim, task_text, xyz_bounds=((-0.755, 0.755),(-0.755, 0.755),(0, 1.309)), gripper_range=(0, 1), normalize_proprio=True):
    full_rgb  = sim.capture_frame(sim.overview_cam, "overview")
    wrist_rgb = sim.capture_frame(sim.ee_cam, "ee_camera")
    ee   = sim.get_end_effector_position().astype(np.float32)
    yaw  = float(sim.get_yaw()) if hasattr(sim, "get_yaw") else 0.0
    grip = float(getattr(sim, "get_gripper_opening", lambda: 0.03)())

    if normalize_proprio:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = xyz_bounds
        ee_n = np.array([
            norm11_from_bounds(ee[0], xlo, xhi),
            norm11_from_bounds(ee[1], ylo, yhi),
            norm11_from_bounds(ee[2], zlo, zhi),
            np.clip(yaw/np.pi, -1, 1),   # yaw in [-pi,pi] → [-1,1]
            0.0, 0.0, 0.0,
            norm11_from_bounds(grip, gripper_range[0], gripper_range[1]),
        ], dtype=np.float32)
        proprio = ee_n
    else:
        proprio = np.array([ee[0], ee[1], ee[2], yaw, 0,0,0, grip], dtype=np.float32)

    obs = {"full_image": full_rgb, "wrist_image": wrist_rgb, "state": proprio,
           "task_description": task_text}
    return obs, proprio.size


# ---------- runner ----------

def main():
    import argparse
    ap = argparse.ArgumentParser("OpenVLA-OFT → CDPR runner")
    ap.add_argument("--xml", required=True, help="Wrapper MJCF (scene+cdpr+objects).")
    ap.add_argument("--ckpt", default="moojink/openvla-7b-oft-finetuned-libero-spatial",
                    help="OpenVLA-OFT checkpoint id")
    # ap.add_argument("--center_crop", type=bool, default=True)
    ap.add_argument("--center_crop", dest="center_crop", action="store_true")
    ap.add_argument("--no_center_crop", dest="center_crop", action="store_false")
    ap.set_defaults(center_crop=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--chunk_replan", type=int, default=NUM_ACTIONS_CHUNK,
                    help="Replan each chunk: request a new action chunk every N steps.")
    ap.add_argument("--instr", default="Pick up the orange juice, then hover.")
    # workspace & gripper bounds (adjust to your setup)
    ap.add_argument("--x_bound", default="-0.8,0.8")
    ap.add_argument("--y_bound", default="-0.8,0.8")
    ap.add_argument("--z_bound", default="0.10,1.20")
    ap.add_argument("--grip_range", default="0.0,0.06")
    args = ap.parse_args()

    def parse_pair(s):
        lo, hi = map(float, s.split(","))
        return (lo, hi)
    xyz_bounds = (parse_pair(args.x_bound), parse_pair(args.y_bound), parse_pair(args.z_bound))
    gr_range = parse_pair(args.grip_range)

    # ---- build OFT config ----
    cfg = GenerateConfig(
        pretrained_checkpoint=args.ckpt,
        use_l1_regression=True,   # continuous actions head
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,    # full_image + wrist_image
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=bool(args.center_crop),
        num_open_loop_steps=NUM_ACTIONS_CHUNK,  # policy emits a chunk of this length
        unnorm_key="libero_spatial_no_noops",   # choose per checkpoint family
    )

    # ---- init sim ----
    sim = HeadlessCDPRSimulation(args.xml, output_dir="oft_runs")
    sim.initialize()
    # Ensure we hold the current pose at t=0 (your earlier helper)
    if hasattr(sim, "hold_current_pose"):
        sim.hold_current_pose(warm_steps=0)

    # ---- load policy + heads ----
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

    # Build one observation to determine proprio_dim dynamically
    obs, proprio_dim = make_observation(sim, args.instr)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=proprio_dim)

    # helper to get a fresh chunk when needed
    def request_chunk():
        o, _ = make_observation(sim, args.instr)
        # print("full:", o["full_image"].shape, o["full_image"].dtype)
        # print("wrist:", o["wrist_image"].shape, o["wrist_image"].dtype)
        # print("proprio:", o["state"].shape, o["state"].dtype)

        acts = get_vla_action(cfg, vla, processor, o, o["task_description"], action_head, proprio_projector)
        # acts is a list/array of 7D actions per tick (length NUM_ACTIONS_CHUNK)
        return acts

    # first chunk
    chunk = request_chunk()
    c_idx = 0
    
    # --- quick sanity snapshot ---
    # obs_dbg, _ = make_observation(sim, args.instr, xyz_bounds, gr_range, normalize_proprio=True)
    # Image.fromarray(obs_dbg["full_image"]).save("diag_full.png")
    # Image.fromarray(obs_dbg["wrist_image"]).save("diag_wrist.png")
    # print("[diag] saved diag_full.png / diag_wrist.png")

    # acts = get_vla_action(cfg, vla, processor, obs_dbg, obs_dbg["task_description"], action_head,
    #                     get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=_))
    # acts = np.array(acts, dtype=float)
    # print("[diag] first 5 actions:\n", acts[:5])
    # print("[diag] action mean/std:", acts.mean(axis=0), acts.std(axis=0))

    # ee_before = sim.get_end_effector_position().copy()
    # for k in range(min(30, len(acts))):
    #     xyz, yaw, grip = map_7d_to_cdpr5(acts[k], xyz_bounds, gr_range)
    #     sim.set_target_position(xyz)
    #     if hasattr(sim, "set_yaw"): sim.set_yaw(yaw)
    #     if hasattr(sim, "set_gripper"): sim.set_gripper(grip)
    #     sim.run_simulation_step(capture_frame=False)
    # ee_after = sim.get_end_effector_position().copy()
    # print("[diag] Δee after 30 action steps:", (ee_after - ee_before))

    xyz_target = sim.get_end_effector_position().copy()
    yaw_target = sim.get_yaw() if hasattr(sim, "get_yaw") else 0.0
    grip_now   = getattr(sim, "get_gripper_opening", lambda: 0.03)()
    
    # ---- roll out ----
    for t in range(args.steps):
        if c_idx >= len(chunk):
            chunk = request_chunk()
            c_idx = 0
        a7 = chunk[c_idx]; c_idx += 1

        dxyz, dyaw, dgrip = map_7d_delta_to_cdpr5(a7, xyz_bounds, gr_range)

        # integrate + clamp to workspace
        xyz_target = xyz_target + dxyz
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = xyz_bounds
        xyz_target[0] = np.clip(xyz_target[0], xlo, xhi)
        xyz_target[1] = np.clip(xyz_target[1], ylo, yhi)
        xyz_target[2] = np.clip(xyz_target[2], zlo, zhi)

        yaw_target = ((yaw_target + dyaw + np.pi) % (2*np.pi)) - np.pi
        grip_now = float(np.clip(grip_now + dgrip, gr_range[0], gr_range[1]))

        sim.set_target_position(xyz_target)
        # sim.set_target_position(np.array([0.0, -0.3, 0.3]))
        # sim.set_yaw(1.0)
        # for _ in range(240):
        #     sim.run_simulation_step(capture_frame=False)
        if hasattr(sim, "set_yaw"): sim.set_yaw(yaw_target)
        if hasattr(sim, "set_gripper"): sim.set_gripper(grip_now)

        sim.run_simulation_step(capture_frame=True)
    
    # save via your built-ins
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(sim.output_dir) / f"oft_run_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    sim.save_trajectory_results(str(outdir), "oft_cdpr")
    sim.cleanup()

if __name__ == "__main__":
    main()
