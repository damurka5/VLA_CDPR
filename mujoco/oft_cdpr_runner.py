#!/usr/bin/env python3
import os, sys, time
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

# ---- repo local import of your sim ----
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
from headless_cdpr_egl import HeadlessCDPRSimulation  # your class

# ---- openvla-oft imports (run in openvla-oft env) ----
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import (
    get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # chunk length

# ---------- mapping & utils ----------

def clamp(v, lo, hi):
    return float(max(lo, min(hi, v)))

def map_7d_to_cdpr5(act7, xyz_bounds, gripper_range):
    """
    act7: [x,y,z,yaw,pitch,roll,gripper]
    returns: xyz(3), yaw(float), grip_opening_m(float)
    """
    a = np.array(act7, dtype=float).flatten()
    if a.size < 7:  # pad if needed
        a = np.pad(a, (0, 7 - a.size))

    # assume OpenVLA-OFT emits in WORLD units already; if normalized, add a remap here
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = xyz_bounds
    x = clamp(a[0], xlo, xhi)
    y = clamp(a[1], ylo, yhi)
    z = clamp(a[2], zlo, zhi)

    # yaw: allow either radians or [-1,1] normalized; if looks small, scale by pi
    yaw_raw = float(a[3])
    yaw = yaw_raw if abs(yaw_raw) > 1.2 else yaw_raw * np.pi
    # wrap to [-pi, pi]
    yaw = (yaw + np.pi) % (2*np.pi) - np.pi

    # gripper: assume normalized [0..1] or [-1..1] → meters
    g = float(a[6])
    if -1.2 <= g <= 1.2:
        g = 0.5 * (g + 1.0)  # map [-1,1]→[0,1]
    g = clamp(g, 0.0, 1.0)
    gr_lo, gr_hi = gripper_range
    grip = gr_lo + g * (gr_hi - gr_lo)

    return np.array([x, y, z]), yaw, grip

def make_observation(sim, task_text: str):
    """
    Build OFT-style observation dict.
    Images must be np.uint8 HxWx3 (RGB). Proprio is 8-D for the pretrained projector:
      [ee_x, ee_y, ee_z, yaw, pitch=0, roll=0, dummy=0, gripper]
    """
    # ---- images as np.uint8 arrays (RGB) ----
    full_rgb  = sim.capture_frame(sim.overview_cam, "overview")   # already HxWx3 uint8
    wrist_rgb = sim.capture_frame(sim.ee_cam, "ee_camera")
    # ensure dtype and contiguous memory
    full_rgb  = np.ascontiguousarray(full_rgb, dtype=np.uint8)
    wrist_rgb = np.ascontiguousarray(wrist_rgb, dtype=np.uint8)

    # ---- proprio (8-D expected by OFT checkpoints) ----
    ee   = sim.get_end_effector_position().astype(np.float32)  # (3,)
    yaw  = float(getattr(sim, "get_yaw", lambda: 0.0)())
    grip = float(getattr(sim, "get_gripper_opening", lambda: 0.03)())
    proprio = np.array([ee[0], ee[1], ee[2], yaw, 0.0, 0.0, 0.0, grip], dtype=np.float32)

    obs = {
        "full_image":  full_rgb,   # <- NumPy, not PIL
        "wrist_image": wrist_rgb,  # <- NumPy, not PIL
        "state": proprio,
        "task_description": task_text,
    }
    return obs, proprio.size



# ---------- runner ----------

def main():
    import argparse
    ap = argparse.ArgumentParser("OpenVLA-OFT → CDPR runner")
    ap.add_argument("--xml", required=True, help="Wrapper MJCF (scene+cdpr+objects).")
    ap.add_argument("--ckpt", default="moojink/openvla-7b-oft-finetuned-libero-spatial",
                    help="OpenVLA-OFT checkpoint id")
    ap.add_argument("--center_crop", type=bool, default=True)
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

    # ---- roll out ----
    for t in range(args.steps):
        if c_idx >= len(chunk):
            chunk = request_chunk()
            c_idx = 0

        a7 = chunk[c_idx]
        c_idx += 1

        # map 7D → [xyz,yaw,grip]
        xyz, yaw, grip = map_7d_to_cdpr5(a7, xyz_bounds, gr_range)

        # apply to sim
        sim.set_target_position(xyz)
        if hasattr(sim, "set_yaw"): sim.set_yaw(yaw)
        if hasattr(sim, "set_gripper"): sim.set_gripper(grip)
        sim.run_simulation_step(capture_frame=True)

    # save via your built-ins
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(sim.output_dir) / f"oft_run_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    sim.save_trajectory_results(str(outdir), "oft_cdpr")
    sim.cleanup()

if __name__ == "__main__":
    main()
