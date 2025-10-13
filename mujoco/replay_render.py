import os, imageio, numpy as np
from PIL import Image
from cdpr_libero_adapter import CDPRLiberoEnv

def main():
    run_dir = os.path.abspath("runs_headless/20251013_140914/")
    traj = np.load(os.path.join(run_dir, "trajectory_data.npz"))
    actions = traj["action"]

    # create render dir
    out_dir = os.path.abspath(run_dir.replace("runs_headless", "runs_render"))
    os.makedirs(out_dir, exist_ok=True)
    vid_path = os.path.join(out_dir, "trajectory_video.mp4")

    # env with EGL ON and higher resolution
    env = CDPRLiberoEnv(xml_path="VLA_CDPR/mujoco/cdpr.xml", egl=True, img_hw=(480, 640))
    obs = env.reset(bddl_problem_path="LIBERO/lifelong/data/bddl/cdpr_go_near_object.bddl")

    frames = [obs["rgb"]]
    for a in actions:
        obs, _, done, _ = env.step(a)
        frames.append(obs["rgb"])
        if done: break

    with imageio.get_writer(vid_path, fps=20) as w:
        for f in frames: w.append_data(f)

    env.close()
    print(f"✓ video saved to {vid_path}")

if __name__ == "__main__":
    main()
