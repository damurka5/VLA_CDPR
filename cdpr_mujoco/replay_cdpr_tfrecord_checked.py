#!/usr/bin/env python3
"""
Replay a CDPR episode directly from an RLDS-style TFRecord (as produced by export_to_rlds.py),
then render an overview (and optional wrist) video from the replay simulation.

Supports two replay modes:
  --mode state   : use observation/state[t] as absolute commands (recommended)
  --mode action  : integrate action deltas to reconstruct absolute commands

Also optionally writes "dataset" videos decoded from TFRecord images for side-by-side comparison.

Example:
  python replay_cdpr_tfrecord.py --xml wrapper.xml --tfrecord episode_000001.tfrecord \
      --mode state --replay-fps 10 --record-video --video-fps 10 --out replay_from_tfrecord

Optional dataset video dump:
  python replay_cdpr_tfrecord.py ... --write-dataset-video
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

from headless_cdpr_egl import HeadlessCDPRSimulation


# -------- TFRecord parsing (matches export_to_rlds.py) --------
FEATURES = {
    "observation/primary": tf.io.FixedLenFeature([], tf.string),
    "observation/wrist": tf.io.FixedLenFeature([], tf.string),
    "observation/state": tf.io.VarLenFeature(tf.float32),
    "observation/task_description": tf.io.FixedLenFeature([], tf.string),
    "action": tf.io.VarLenFeature(tf.float32),
    "is_terminal": tf.io.VarLenFeature(tf.int64),
    "is_first": tf.io.VarLenFeature(tf.int64),
    "is_last": tf.io.VarLenFeature(tf.int64),
}


def _parse_example(serialized):
    ex = tf.io.parse_single_example(serialized, FEATURES)
    state = tf.sparse.to_dense(ex["observation/state"])
    action = tf.sparse.to_dense(ex["action"])
    return {
        "primary_jpeg": ex["observation/primary"],
        "wrist_jpeg": ex["observation/wrist"],
        "state": state,
        "action": action,
        "task": ex["observation/task_description"],
    }


def _decode_jpeg_to_bgr(jpeg_bytes: bytes):
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return img


def _annotate_bgr(frame_bgr, lines):
    img = frame_bgr.copy()
    y = 24
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        y += 24
    return img


def _wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="MJCF xml for the CDPR sim (same used in collection).")
    ap.add_argument("--tfrecord", required=True, help="Episode TFRecord path.")
    ap.add_argument("--out", default="replay_from_tfrecord", help="Output directory.")
    ap.add_argument("--mode", choices=["state", "action"], default="state",
                    help="Replay using absolute state or integrated action deltas.")
    ap.add_argument("--replay-fps", type=float, default=10.0, help="Dataset/control rate (Hz) used for replay.")
    ap.add_argument("--video-fps", type=float, default=10.0, help="FPS of the OUTPUT replay video.")
    ap.add_argument("--record-video", action="store_true", help="Write replay_overview.mp4 (and replay_wrist.mp4).")
    ap.add_argument("--wrist", action="store_true", help="Also write wrist replay video.")
    ap.add_argument("--annotate", action="store_true", help="Overlay step/state/gripper info on replay video.")
    ap.add_argument("--warmup-goto", action="store_true", help="Goto first xyz before replaying.")
    ap.add_argument("--max-steps", type=int, default=None, help="Limit number of dataset steps to replay.")
    ap.add_argument("--gripper-open", type=float, default=0.03, help="Clamp for set_gripper in meters.")
    ap.add_argument("--write-dataset-video", action="store_true",
                    help="Also decode TFRecord JPEGs and write dataset_overview.mp4 (+ dataset_wrist.mp4).")

    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load TFRecord into numpy arrays ----
    ds = tf.data.TFRecordDataset([str(args.tfrecord)])
    parsed = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    primary_jpegs = []
    wrist_jpegs = []
    states = []
    actions = []
    task = None

    for item in parsed:
        primary_jpegs.append(bytes(item["primary_jpeg"].numpy()))
        wrist_jpegs.append(bytes(item["wrist_jpeg"].numpy()))
        st = item["state"].numpy().astype(np.float64)
        ac = item["action"].numpy().astype(np.float64)
        states.append(st)
        actions.append(ac)
        if task is None:
            task = item["task"].numpy().decode("utf-8")

    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    T = len(states)

    if args.max_steps is not None:
        T = min(T, int(args.max_steps))
        states = states[:T]
        actions = actions[:T]
        primary_jpegs = primary_jpegs[:T]
        wrist_jpegs = wrist_jpegs[:T]

    if states.ndim != 2 or states.shape[1] < 5:
        raise SystemExit(f"Expected state shape (T,5+). Got {states.shape}")
    if actions.ndim != 2 or actions.shape[1] < 5:
        raise SystemExit(f"Expected action shape (T,5+). Got {actions.shape}")

    print(f"Loaded TFRecord: steps={T}, task='{task}'")
    print(f"State gripper min/max: {states[:,4].min():.4f} / {states[:,4].max():.4f}  "
          f"(changes={(np.abs(np.diff(states[:,4]))>1e-6).sum()})")
    print(f"Action gripper delta min/max: {actions[:,4].min():.4f} / {actions[:,4].max():.4f}")

    # ---- Sanity checks for correspondence ----
    # In a typical RLDS layout, action[t] should move observation/state[t] -> observation/state[t+1].
    if T >= 2:
        pred_next = states[:-1] + actions[:-1]
        err = pred_next - states[1:]
        max_abs = np.max(np.abs(err), axis=0)
        mean_abs = np.mean(np.abs(err), axis=0)
        print("State/action consistency (state[t] + action[t] ≈ state[t+1]):")
        print(f"  max_abs_err per dim:  {np.array2string(max_abs, precision=6)}")
        print(f"  mean_abs_err per dim: {np.array2string(mean_abs, precision=6)}")

        dnorm = np.linalg.norm(states[1:] - states[:-1], axis=1)
        # Find last index where state actually changes (helps detect trailing repeated states)
        nz = np.where(dnorm > 1e-9)[0]
        if len(nz) > 0:
            last_change = int(nz[-1] + 1)
            if last_change < T - 1:
                print(f"[warn] state becomes (almost) constant after t={last_change} "
                      f"({T-last_change} trailing steps with ~zero motion). "
                      f"This usually means video steps extend past the end of the log timestamps.")
        else:
            print("[warn] state never changes (all deltas ~0). Check your export.")

    # Optionally dump dataset videos from TFRecord images
    if args.write_dataset_video:
        def write_video(frames_bgr, path, fps):
            if not frames_bgr:
                return
            h, w = frames_bgr[0].shape[:2]
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
            for f in frames_bgr:
                vw.write(f)
            vw.release()

        ov_bgr = [_decode_jpeg_to_bgr(b) for b in primary_jpegs]
        write_video(ov_bgr, out_dir / "dataset_overview.mp4", args.replay_fps)
        print(f"✅ wrote dataset_overview.mp4 ({args.replay_fps} fps)")

        if args.wrist:
            wr_bgr = [_decode_jpeg_to_bgr(b) for b in wrist_jpegs]
            write_video(wr_bgr, out_dir / "dataset_wrist.mp4", args.replay_fps)
            print(f"✅ wrote dataset_wrist.mp4 ({args.replay_fps} fps)")

    # ---- Build absolute command trajectory depending on mode ----
    if args.mode == "state":
        abs5 = states.copy()
    else:
        # integrate deltas starting from state[0]
        abs5 = np.zeros_like(states)
        abs5[0] = states[0]
        for t in range(1, T):
            abs5[t] = abs5[t-1] + actions[t-1]  # action[t-1] should move from t-1 -> t
            abs5[t, 3] = _wrap_to_pi(abs5[t, 3])  # yaw wrap

    # ---- Replay in sim ----
    sim = HeadlessCDPRSimulation(xml_path=str(args.xml), output_dir=str(out_dir))
    sim.initialize()

    sim_dt = float(sim.controller.dt)
    hold = int(round((1.0 / float(args.replay_fps)) / sim_dt))
    hold = max(1, hold)

    capture_stride = int(round((1.0 / float(args.video_fps)) / sim_dt)) if args.record_video else None
    if capture_stride is not None:
        capture_stride = max(1, capture_stride)

    print(f"sim_dt={sim_dt:.6f}, replay_fps={args.replay_fps} -> hold_steps={hold}")
    if args.record_video:
        print(f"output video_fps={args.video_fps} -> capture_stride={capture_stride}")

    if args.warmup_goto:
        ok, n = sim.goto(abs5[0, 0:3], max_steps=2000, tol=0.02, capture_every_n=999999)
        print(f"goto(start) ok={ok} steps={n}")

    sim.overview_frames = []
    sim.ee_camera_frames = []

    sim_step = 0
    pos_errs = []
    yaw_errs = []
    grip_errs = []

    for t in range(T):
        cmd = abs5[t]
        sim.set_target_position(cmd[0:3])
        sim.set_yaw(float(cmd[3]))
        sim.set_gripper(float(np.clip(cmd[4], 0.0, args.gripper_open)))

        for _ in range(hold):
            sim.run_simulation_step(capture_frame=False)
            sim_step += 1

            if args.record_video and (sim_step % capture_stride == 0):
                # EGL safety
                if getattr(sim, "gl_context", None) is not None:
                    try:
                        sim.gl_context.make_current()
                    except Exception:
                        pass

                # capture returns RGB; convert to BGR for cv2 video writer used in sim.save_video
                ov_rgb = sim.capture_frame(sim.overview_cam, "overview")
                if args.annotate:
                    ee = sim.get_end_effector_position()[:3]
                    pos_err = float(np.linalg.norm(ee - cmd[0:3]))
                    ov_rgb = cv2.cvtColor(_annotate_bgr(cv2.cvtColor(ov_rgb, cv2.COLOR_RGB2BGR),
                                                        [f"t={t}/{T-1}",
                                                         f"pos_err={pos_err:.3f}m",
                                                         f"yaw={cmd[3]:.2f}",
                                                         f"grip={cmd[4]:.3f}m"]),
                                          cv2.COLOR_BGR2RGB)
                sim.overview_frames.append(ov_rgb)

                if args.wrist:
                    wr_rgb = sim.capture_frame(sim.ee_cam, "ee_camera")
                    if args.annotate:
                        ee = sim.get_end_effector_position()[:3]
                        pos_err = float(np.linalg.norm(ee - cmd[0:3]))
                        wr_rgb = cv2.cvtColor(_annotate_bgr(cv2.cvtColor(wr_rgb, cv2.COLOR_RGB2BGR),
                                                            [f"t={t}/{T-1}",
                                                             f"pos_err={pos_err:.3f}m",
                                                             f"yaw={cmd[3]:.2f}",
                                                             f"grip={cmd[4]:.3f}m"]),
                                              cv2.COLOR_BGR2RGB)
                    sim.ee_camera_frames.append(wr_rgb)

        # record simple tracking errors (command vs achieved)
        ee = sim.get_end_effector_position()[:3]
        pos_errs.append(float(np.linalg.norm(ee - cmd[0:3])))
        yaw_errs.append(abs(_wrap_to_pi(sim.get_yaw() - cmd[3])))
        grip_errs.append(abs(float(sim.data.ctrl[sim.act_gripper]) - float(np.clip(cmd[4], 0.0, args.gripper_open))))

        if t % 20 == 0:
            print(f"t={t:4d} cmd_xyz={cmd[0:3].round(3)} ee={ee.round(3)} pos_err={pos_errs[-1]:.4f}")

    pos_errs = np.asarray(pos_errs)
    print("\nTracking error (cmd vs achieved):")
    print(f"  pos mean={pos_errs.mean():.4f} m, p95={np.quantile(pos_errs,0.95):.4f} m, max={pos_errs.max():.4f} m")

    if args.record_video:
        ov_path = out_dir / "replay_overview.mp4"
        sim.save_video(sim.overview_frames, str(ov_path), fps=float(args.video_fps))
        print(f"✅ wrote replay_overview.mp4: {ov_path}")

        if args.wrist:
            wr_path = out_dir / "replay_wrist.mp4"
            sim.save_video(sim.ee_camera_frames, str(wr_path), fps=float(args.video_fps))
            print(f"✅ wrote replay_wrist.mp4: {wr_path}")

    sim.cleanup()


if __name__ == "__main__":
    main()
