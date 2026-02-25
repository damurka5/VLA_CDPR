#!/usr/bin/env python3
"""
Replay a CDPR trajectory from a trajectory_data.npz in MuJoCo, and optionally write
overview / wrist videos from the replay.

This script can also replay GRIPPER commands if they are present in the npz.

Example:
  python replay_cdpr_npz_with_video.py --xml wrapper.xml --npz trajectory_data.npz \
      --use target_position --replay-fps 10 --record-video --video-fps 10 \
      --replay-gripper auto --warmup-goto --out replay_out
"""

import argparse
from pathlib import Path
import numpy as np
import cv2

from headless_cdpr_egl import HeadlessCDPRSimulation


def pick_log_indices_for_video(timestamps, video_fps, num_frames=None, start_time=None):
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.ndim != 1 or len(ts) < 2:
        raise ValueError("timestamps must be 1D and length>=2")

    t0 = ts[0] if start_time is None else float(start_time)

    if num_frames is None:
        t_end = ts[-1]
        num_frames = int(np.floor((t_end - t0) * float(video_fps))) + 1
        num_frames = max(1, num_frames)

    frame_times = t0 + np.arange(int(num_frames), dtype=np.float64) / float(video_fps)

    idx = np.searchsorted(ts, frame_times, side="left")
    idx = np.clip(idx, 0, len(ts) - 1)

    prev = np.clip(idx - 1, 0, len(ts) - 1)
    choose_prev = np.abs(ts[prev] - frame_times) < np.abs(ts[idx] - frame_times)
    idx = np.where(choose_prev, prev, idx)

    return idx.astype(np.int64)


def _annotate(frame, text_lines):
    img = frame.copy()
    y = 22
    for line in text_lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += 22
    return img


def _sample_series_to_length(series, timestamps, fps, target_len):
    series = np.asarray(series)
    if len(series) == target_len:
        return series
    if timestamps is not None and len(timestamps) == len(series) and len(timestamps) >= 2:
        idx = pick_log_indices_for_video(timestamps, fps, num_frames=target_len)
        return series[idx]
    idx = np.linspace(0, len(series) - 1, target_len).round().astype(int)
    idx = np.clip(idx, 0, len(series) - 1)
    return series[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="/root/repo/VLA_CDPR/cdpr_mujoco/wrappers/desk_ketchup_wrapper.xml", help="MJCF xml used for the episode")
    ap.add_argument("--npz", default="/root/repo/cdpr_synth_10hz/videos/HUMAN_CONTROL_desk__ketchup-plate_wrapper__localpatched_teleop_2025-12-14_17-55-20/trajectory_data.npz", help="trajectory_data.npz (downsampled or raw)")
    ap.add_argument("--use", choices=["ee_position", "target_position"], default="target_position")
    ap.add_argument("--replay-fps", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--warmup-goto", action="store_true")
    ap.add_argument("--out", type=str, default="replay_out")
    ap.add_argument("--record-video", action="store_true")
    ap.add_argument("--video-fps", type=float, default=10.0)
    ap.add_argument("--wrist", action="store_true")
    ap.add_argument("--annotate", action="store_true")

    # ---- Gripper replay flags ----
    ap.add_argument("--replay-gripper", choices=["off", "auto", "control_signals"], default="auto",
                    help="Replay gripper from npz. 'auto' uses control_signals if present.")
    ap.add_argument("--gripper-open", type=float, default=0.03,
                    help="Max opening in meters (matches your sim clamp).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logs = np.load(args.npz, allow_pickle=True)
    if args.use not in logs:
        raise SystemExit(f"Key '{args.use}' not in npz. Keys={logs.files}")

    pos = np.asarray(logs[args.use], dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] < 3:
        raise SystemExit(f"{args.use} must be (T,>=3). Got {pos.shape}")
    pos = pos[:, :3]

    dpos = pos[1:] - pos[:-1]
    if args.steps is not None:
        dpos = dpos[:args.steps]
        pos = pos[:args.steps + 1]

    # import numpy as np
    d = np.load(args.npz, allow_pickle=True)
    ctrl = d["control_signals"]
    print("ctrl shape:", ctrl.shape)
    g = ctrl[:, -1]  # often last actuator is gripper
    print("gripper min/max:", g.min(), g.max())
    print("num changes:", (abs(np.diff(g)) > 1e-9).sum())


    sim = HeadlessCDPRSimulation(xml_path=args.xml, output_dir=str(out_dir))
    sim.initialize()

    sim_dt = float(sim.controller.dt)
    hold = int(round((1.0 / float(args.replay_fps)) / sim_dt))
    hold = max(1, hold)

    if args.record_video:
        capture_stride = int(round((1.0 / float(args.video_fps)) / sim_dt))
        capture_stride = max(1, capture_stride)
    else:
        capture_stride = None

    print(f"sim_dt={sim_dt:.6f}")
    print(f"apply XYZ deltas at replay_fps={args.replay_fps}  -> hold_steps={hold}")
    if args.record_video:
        print(f"recording video at video_fps={args.video_fps} -> capture_stride={capture_stride} sim steps")

    # ---- Prepare gripper command series ----
    ts = np.asarray(logs["timestamp"], dtype=np.float64) if "timestamp" in logs else None
    gr_series = None

    if args.replay_gripper != "off":
        if args.replay_gripper in ["auto", "control_signals"] and "control_signals" in logs:
            ctrl = np.asarray(logs["control_signals"], dtype=np.float64)
            if ctrl.ndim == 2 and ctrl.shape[1] > sim.act_gripper:
                gr_raw = ctrl[:, sim.act_gripper]
                gr_series = _sample_series_to_length(gr_raw, ts, args.replay_fps, len(pos))
                gr_series = np.clip(gr_series, 0.0, float(args.gripper_open))
                print(f"gripper: using control_signals[:, act_gripper={sim.act_gripper}] "
                      f"sampled to {len(gr_series)} steps, min={gr_series.min():.4f} max={gr_series.max():.4f}")
            else:
                print("gripper: control_signals present but shape/index mismatch; will not replay gripper.")
        else:
            print("gripper: no usable control_signals in npz; will not replay gripper.")

    if args.warmup_goto:
        ok, n = sim.goto(pos[0], max_steps=2000, tol=0.02, capture_every_n=999999)
        print(f"goto(start) ok={ok} steps={n}")

    cur_cmd = sim.get_end_effector_position()[:3].copy()
    errs = []
    sim_step = 0

    sim.overview_frames = []
    sim.ee_camera_frames = []

    for t in range(len(dpos)):
        cur_cmd = cur_cmd + dpos[t]
        sim.set_target_position(cur_cmd)

        # ---- Apply gripper absolute command (aligned with pos[t+1]) ----
        if gr_series is not None:
            sim.set_gripper(float(gr_series[t + 1]))

        for _ in range(hold):
            sim.run_simulation_step(capture_frame=False)
            sim_step += 1

            if args.record_video and (sim_step % capture_stride == 0):
                if getattr(sim, "gl_context", None) is not None:
                    try:
                        sim.gl_context.make_current()
                    except Exception:
                        pass

                ov = sim.capture_frame(sim.overview_cam, "overview")
                if args.annotate:
                    ee = sim.get_end_effector_position()[:3]
                    err = float(np.linalg.norm(ee - pos[min(t + 1, len(pos) - 1)]))
                    gr_txt = f"grip={float(gr_series[t+1]):.3f} m" if gr_series is not None else ""
                    ov = _annotate(ov, [x for x in [f"t={t}", f"err={err:.3f} m", gr_txt] if x])
                sim.overview_frames.append(ov)

                if args.wrist:
                    wr = sim.capture_frame(sim.ee_cam, "ee_camera")
                    if args.annotate:
                        ee = sim.get_end_effector_position()[:3]
                        err = float(np.linalg.norm(ee - pos[min(t + 1, len(pos) - 1)]))
                        gr_txt = f"grip={float(gr_series[t+1]):.3f} m" if gr_series is not None else ""
                        wr = _annotate(wr, [x for x in [f"t={t}", f"err={err:.3f} m", gr_txt] if x])
                    sim.ee_camera_frames.append(wr)

        ee = sim.get_end_effector_position()[:3]
        err = float(np.linalg.norm(ee - pos[t + 1]))
        errs.append(err)
        if t % 20 == 0:
            print(f"t={t:4d} ee={ee.round(3)} expected={pos[t+1].round(3)} err={err:.4f}")

    errs = np.asarray(errs, dtype=np.float64)
    print("\nReplay error vs chosen logged next pose:")
    print(f"  mean: {errs.mean():.6f} m")
    print(f"  max:  {errs.max():.6f} m")
    print(f"  p95:  {np.quantile(errs, 0.95):.6f} m")

    if args.record_video:
        ov_path = out_dir / "replay_overview.mp4"
        sim.save_video(sim.overview_frames, str(ov_path), fps=float(args.video_fps))
        print(f"\n✅ wrote overview replay video: {ov_path}")

        if args.wrist:
            wr_path = out_dir / "replay_wrist.mp4"
            sim.save_video(sim.ee_camera_frames, str(wr_path), fps=float(args.video_fps))
            print(f"✅ wrote wrist replay video: {wr_path}")

    sim.cleanup()


if __name__ == "__main__":
    main()