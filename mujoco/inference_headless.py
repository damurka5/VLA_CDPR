# LIBERO env

import os
os.environ.setdefault("MUJOCO_GL", "egl")   # force EGL backend
os.environ.setdefault("EGL_DEVICE_ID", "0") # choose GPU 0 (optional)

import os, time, json, base64, numpy as np
from datetime import datetime
from PIL import Image
from io import BytesIO
from cdpr_libero_adapter import CDPRLiberoEnv

import http.client, json

ACTION_MODE = "absolute"   # or "delta" if you want to treat OpenVLA outputs as deltas
DELTA_SCALE = np.array([0.1, 0.1, 0.1], dtype=np.float32)  # only used if ACTION_MODE="delta"
YAW_DEFAULT = 0.0
GRIP_DEFAULT = 1.0   # 1.0=open, 0.0=closed (pick what you like)

def post_json_http(host, port, path, payload, timeout=10):
    body = json.dumps(payload)
    headers = {"Content-Type": "application/json"}
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("POST", path, body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {data[:200]!r}")
    return json.loads(data.decode("utf-8"))


POLICY_URL = "http://0.0.0.0:7071/act"  # set to your policy server

def to_action5(raw_action, ee_pos):
    """Map model outputs (3|5|7 dims) to [x,y,z,yaw,grip] for CDPR env."""
    a = np.array(raw_action, dtype=np.float32).flatten()
    # position
    if a.size >= 3:
        pos3 = a[:3]
    else:
        pos3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # interpret position as absolute or delta
    if ACTION_MODE == "delta":
        pos3 = (ee_pos.astype(np.float32) + pos3 * DELTA_SCALE)

    # yaw & gripper
    if a.size >= 5:
        yaw = float(a[3])
        grip = float(a[4])
    elif a.size >= 7:
        # some OpenVLA variants output 7-DoF: [x,y,z,roll,pitch,yaw,gripper]
        yaw = float(a[5])
        grip = float(a[6])
    else:
        yaw = YAW_DEFAULT
        grip = GRIP_DEFAULT

    # clip to sane ranges (you can tweak)
    pos3 = np.clip(pos3, -1.309, 1.309)
    yaw = float(np.clip(yaw, -np.pi, np.pi))
    grip = float(np.clip(grip, 0.0, 1.0))

    return np.array([pos3[0], pos3[1], pos3[2], yaw, grip], dtype=np.float32)

def ensure_gl_backend():
 import importlib
 try:
     import mujoco as mj
     from mujoco.egl import GLContext
     gl = GLContext(max_width=64, max_height=64); gl.make_current()
     return "egl"
 except Exception as e:
     print(f"[warn] EGL failed: {e}\n[warn] Falling back to OSMesa.")
     os.environ["MUJOCO_GL"] = "osmesa"
     importlib.reload(mj)  # reload with new backend (safe at process start)
     return "osmesa"

backend = ensure_gl_backend()
print(f"MuJoCo GL backend: {backend}")

def img_to_b64(arr):
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def main():
    # 1) choose task & instruction
    bddl = os.path.abspath("/root/repo/LIBERO/libero/libero/bddl_files/cdpr_libero/cdpr_go_near_object.bddl")
    instruction = "go near the red square"  # later: “move above the bowl”, etc.

    # 2) create output dir (LOGS ONLY, no video)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(f"runs_headless/{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    traj_path = os.path.join(out_dir, "trajectory_data.npz")

    # 3) init env (no visible rendering)
    env = CDPRLiberoEnv(xml_path="VLA_CDPR/mujoco/cdpr.xml", egl=True, img_hw=(240, 320))
    obs = env.reset(bddl_problem_path=bddl)
    traj = []

    for t in range(1000):
        # send image + instruction to policy service
        payload = {"instruction": instruction, "image_b64": img_to_b64(obs["rgb"])}
        # a = requests.post(POLICY_URL, json=payload, timeout=10).json()["action"]
        HOST, PORT, PATH = "127.0.0.1", 7071, "/act"
        resp = post_json_http(HOST, PORT, PATH, payload, timeout=10)
        raw = resp["action"]                           # list/array from the server (likely 3-D)
        a5 = to_action5(raw, obs["ee_pos"])            # map to [x,y,z,yaw,grip]
        obs, rew, done, info = env.step(a5)
        # step env
        obs, rew, done, info = env.step(a5)
        traj.append({
            "t": float(t),
            "reward": float(rew),
            "dist": float(info.get("dist_to_goal", np.nan)),
            "ee": obs["ee_pos"].tolist(),
            "target": obs["target_pos"].tolist(),
            "action": a5,
        })
        if done: break

    # 4) save logs
    np.savez(traj_path,
             t=np.array([x["t"] for x in traj], dtype=np.float32),
             reward=np.array([x["reward"] for x in traj], dtype=np.float32),
             dist=np.array([x["dist"] for x in traj], dtype=np.float32),
             ee=np.array([x["ee"] for x in traj], dtype=np.float32),
             target=np.array([x["target"] for x in traj], dtype=np.float32),
             action=np.array([x["action"] for x in traj], dtype=np.float32))
    with open(meta_path, "w") as f:
        json.dump({"bddl": bddl, "instruction": instruction, "timestamp": stamp}, f, indent=2)

    env.close()
    print(f"✓ headless run saved to {out_dir}")

if __name__ == "__main__":
    main()
