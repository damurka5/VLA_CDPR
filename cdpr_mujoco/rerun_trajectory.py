#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path

from headless_cdpr_egl import HeadlessCDPRSimulation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="MJCF xml used for the episode")
    ap.add_argument("--npz", required=True, help="downsampled trajectory_data.npz")
    ap.add_argument("--use", choices=["ee_position", "target_position"], default="target_position")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=None, help="limit number of replay steps")
    ap.add_argument("--warmup-goto", action="store_true", help="goto first pose before replay")
    args = ap.parse_args()

    logs = np.load(args.npz, allow_pickle=True)
    if args.use not in logs:
        raise SystemExit(f"Key '{args.use}' not in npz. Keys={logs.files}")

    pos = np.asarray(logs[args.use], dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] < 3:
        raise SystemExit(f"{args.use} must be (T,3+). Got {pos.shape}")
    pos = pos[:, :3]

    # build deltas at dataset rate
    dpos = pos[1:] - pos[:-1]

    if args.steps is not None:
        dpos = dpos[:args.steps]
        pos = pos[:args.steps+1]

    # sim init
    sim = HeadlessCDPRSimulation(xml_path=args.xml, output_dir="replay_out")
    sim.initialize()

    sim_dt = float(sim.controller.dt)
    hold = int(round((1.0 / args.fps) / sim_dt))
    hold = max(1, hold)
    print(f"sim_dt={sim_dt:.6f}, replay_fps={args.fps}, hold_steps={hold}")

    # optional: move close to first pose
    if args.warmup_goto:
        ok, n = sim.goto(pos[0], max_steps=2000, tol=0.02, capture_every_n=999999)
        print(f"goto(start) ok={ok} steps={n}")

    # replay by applying deltas
    cur = sim.get_end_effector_position()[:3].copy()
    errs = []

    for t in range(len(dpos)):
        cur = cur + dpos[t]
        sim.set_target_position(cur)

        for _ in range(hold):
            sim.run_simulation_step(capture_frame=False)

        ee = sim.get_end_effector_position()[:3]
        # Compare to expected next pose in the logged sequence (pos[t+1])
        err = np.linalg.norm(ee - pos[t+1])
        errs.append(err)

        if t % 20 == 0:
            print(f"t={t:4d} ee={ee.round(3)} expected={pos[t+1].round(3)} err={err:.4f}")

    errs = np.array(errs)
    print("\nReplay error vs logged next pose:")
    print(f"  mean: {errs.mean():.6f} m")
    print(f"  max:  {errs.max():.6f} m")
    print(f"  p95:  {np.quantile(errs, 0.95):.6f} m")

    sim.cleanup()

if __name__ == "__main__":
    main()
