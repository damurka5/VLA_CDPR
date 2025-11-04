import os, sys, json, time
import numpy as np
import imageio
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

# Make stdout unbuffered so you see logs immediately
os.environ.setdefault("PYTHONUNBUFFERED", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
# Save next to this script in .../VLA_CDPR/mujoco/openvla_trajectories
SAVE_ROOT = os.path.join(HERE, "openvla_trajectories")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def make_openvla(model_name="openvla/openvla-7b", force_cpu=False):
    import torch, os
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"OpenVLA device: {device}")

    torch.set_default_dtype(torch.float32)

    # Make sure we have a big, writable offload dir on disk
    offload_dir = os.path.join(os.path.expanduser("~"), "hf_offload")
    os.makedirs(offload_dir, exist_ok=True)

    # Limit how much memory we allow on each device so accelerate will spill to CPU/disk
    # Adjust numbers to your machine limits
    max_memory = {
        "cpu": "24GiB",          # reduce if you have less RAM
        "cuda:0": "6GiB",        # keep small to avoid GPU OOM
    }

    print("Loading AutoProcessor...")
    proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    print("Loading VLA model with device_map=auto + offload...")
    vla = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float32,      # keep fp32 in this env
        trust_remote_code=True,
        low_cpu_mem_usage=False,        # disable to let accelerate control sharding/offload
        device_map="auto",              # ← key: shard across devices
        max_memory=max_memory,          # ← key: cap memory per device
        offload_folder=offload_dir,     # ← key: spill weights to disk
        offload_state_dict=True,
    )
    print("OpenVLA loaded with offloading.")
    return proc, vla, device

def vla_action(proc, vla, device, rgb, instruction):
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = proc(prompt, Image.fromarray(rgb))
    # move to device without dtype casts in this env
    for k, v in list(inputs.items()):
        if hasattr(v, "to"):
            inputs[k] = v.to(device)
    with torch.inference_mode():
        act = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    if isinstance(act, torch.Tensor):
        action = act[:3].detach().float().cpu().numpy()
    elif isinstance(act, (list, tuple, np.ndarray)):
        action = np.asarray(act, dtype=np.float32)[:3]
    else:
        action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return action

def main():
    # 0) create absolute save folder up front
    os.makedirs(SAVE_ROOT, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SAVE_ROOT, f"openvla_trajectory_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    log(f"Output dir: {out_dir}")

    # 1) load model FIRST (so EGL destructor spam doesn’t appear on failures)
    proc, vla, device = make_openvla()

    # 2) import env only after model is alive
    from cdpr_libero_adapter import CDPRLiberoEnv

    env = None
    frames = []
    traj = []

    try:
        # If your LIBERO repo lives at /repo/LIBERO, don’t use ../LIBERO relative to this file.
        # The adapter doesn’t *require* a valid BDDL path yet, but let’s be explicit:
        bddl_path = None  # or an absolute path if you want to use one

        log("Creating MuJoCo env (EGL)...")
        env = CDPRLiberoEnv(
            xml_path=os.path.join(HERE, "cdpr.xml"),
            egl=True,
            goal_radius=0.15,
            img_hw=(480, 640)
        )
        log("Resetting env...")
        obs = env.reset(bddl_problem_path=bddl_path)
        log(f"Initial ee_pos: {obs.get('ee_pos')}, target_pos: {obs.get('target_pos')}")

        instruction = "go near the red square"
        max_steps = 200
        vla_update_interval = 10

        # Capture and save a first still, even if we exit early
        frames.append(obs["rgb"])
        first_png = os.path.join(out_dir, "frame0.png")
        Image.fromarray(obs["rgb"]).save(first_png)
        log(f"Saved first frame: {first_png}")

        t0 = time.time()
        for t in range(max_steps):
            if t % vla_update_interval == 0:
                action_xyz = vla_action(proc, vla, device, obs["rgb"], instruction)
                log(f"t={t} VLA action: {action_xyz}")

            obs, rew, done, info = env.step(action_xyz)
            frames.append(obs["rgb"])
            traj.append({
                "t": float(t),
                "reward": float(rew),
                "dist": float(info.get("dist_to_goal", np.nan)),
                "ee": obs["ee_pos"].tolist(),
                "target": obs["target_pos"].tolist()
            })

            if (t % 10) == 0:
                log(f"t={t} reward={rew:.3f} dist={info.get('dist_to_goal', np.nan):.3f}")

            if done:
                log(f"✓ Success at t={t}")
                break

        log(f"Loop done in {time.time()-t0:.2f}s. Frames: {len(frames)}")

        # 3) ALWAYS save something
        vid_path = os.path.join(out_dir, "trajectory_video.mp4")
        log(f"Writing video to {vid_path} ...")
        with imageio.get_writer(vid_path, fps=20) as w:
            for f in frames:
                w.append_data(f)
        log("Video written.")

        npz_path = os.path.join(out_dir, "trajectory_data.npz")
        log(f"Saving trajectory to {npz_path} ...")
        if traj:
            np.savez(npz_path,
                     t=np.array([x["t"] for x in traj]),
                     reward=np.array([x["reward"] for x in traj]),
                     dist=np.array([x["dist"] for x in traj]),
                     ee=np.array([x["ee"] for x in traj]),
                     target=np.array([x["target"] for x in traj]))
        else:
            # create a minimal file so you can see it exists
            np.savez(npz_path, t=np.array([]), reward=np.array([]), dist=np.array([]),
                     ee=np.zeros((0,3)), target=np.zeros((0,3)))
        log("Data saved.")

    except KeyboardInterrupt:
        log("Interrupted by user.")
    except Exception as e:
        log(f"ERROR: {e}")
        import traceback; traceback.print_exc()
    finally:
        if env is not None:
            log("Closing env...")
            try:
                env.close()
            except Exception as e:
                log(f"Env close warning: {e}")
        log("Done.")

if __name__ == "__main__":
    main()
