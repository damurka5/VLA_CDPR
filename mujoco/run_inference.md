# OpenVLA ↔️ LIBERO (CDPR) Inference via Server–Client (Two-Env)

**Envs used:**

* `mujoco_offscreen` → runs **OpenVLA** policy server (Torch 2.x, CUDA 12.x, Python 3.9)
* `libero` → runs **CDPR + LIBERO** simulation (Torch 1.11, CUDA 11.3, Python 3.8)

---

## 0) Folder layout (only used files)

```
repo/
├─ LIBERO/                       # libre benchmark repo (BDDL lives here)
├─ VLA_CDPR/
│  └─ mujoco/
│     ├─ cdpr.xml
│     ├─ cdpr_libero_adapter.py
│     ├─ server_policy.py        # OpenVLA HTTP server (runs in mujoco_offscreen env)
│     ├─ inference_headless.py   # client (runs episodes; no video; runs in libero env)
│     └─ replay_render.py        # replays for MP4; runs in libero env
├─ runs_headless/                # saved trajectories (.npz + meta.json)
└─ runs_render/                  # rendered MP4s from replays
```

---

## 1) Policy server (OpenVLA) — **mujoco_offscreen** env

### 1.1 Activate env and set GPU

```bash
conda activate mujoco_offscreen
export CUDA_VISIBLE_DEVICES=0
```

don't forget to `conda install -c conda-forge libegl-devel` and `conda install -c conda-forge gcc`

### 1.2 Start the OpenVLA server

`VLA_CDPR/mujoco/server_policy.py` should:

* load `openvla/openvla-7b` (Torch 2.x)
* expose `POST /act` with JSON: `{"instruction": str, "image_b64": <base64 PNG/JPEG>}`
* return `{"action": [x,y,z,...]}` (we currently use first 3 dims)

Run:

```bash
cd VLA_CDPR/mujoco
uvicorn server_policy:app --host 0.0.0.0 --port 7071
```

**Expected logs (OK):**

```
Uvicorn running on http://0.0.0.0:7071
```

> Tip: Keep this process running. If model downloads on first run, it might take a while.

---

## 2) Headless inference (no video) — **libero** env

### 2.1 Activate env and force EGL offscreen

```bash
conda activate libero
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0          # pick GPU if multiple
```

### 2.2 Run the headless client

This script:

* resets the env from your **BDDL** task,
* captures an RGB frame per step (offscreen),
* calls the policy server to get an action,
* steps the simulator,
* saves a compact trajectory `.npz` and metadata.

```bash
python VLA_CDPR/mujoco/inference_headless.py
```

**Expected output (OK):**

```
MuJoCo GL backend: egl
✓ headless run saved to /root/repo/runs_headless/2025MMDD_HHMMSS
```

Artifacts:

```
runs_headless/<stamp>/
├─ meta.json
└─ trajectory_data.npz  # {t, ee, target, action, reward, dist}
```

---

## 3) Replay and render to MP4 — **libero** env

Reproduce the stored trajectory (deterministic) and record a video.

```bash
python VLA_CDPR/mujoco/replay_render.py
```

**Expected output (OK):**

```
✓ video saved to /root/repo/runs_render/<same stamp>/trajectory_video.mp4
```

---

## 4) Configuration switches you’ll care about

In **`inference_headless.py`** (client):

* **Server location**

  ```python
  POLICY_HOST = "127.0.0.1"
  POLICY_PORT = 7071
  POLICY_PATH = "/act"
  ```

* **Instruction**

  ```python
  instruction = "go near the red square"
  # later: "move above the bowl", "go near the mug", etc.
  ```

* **Action adapter (3/5/7 → 5-D)**

  ```python
  # Maps server outputs (3, 5, or 7 dims) to [x, y, z, yaw, gripper]
  a5 = to_action5(raw_action, obs["ee_pos"])
  obs, rew, done, info = env.step(a5)
  ```

* **EGL / resolution**

  ```python
  os.environ["MUJOCO_GL"] = "egl"
  env = CDPRLiberoEnv(xml_path="VLA_CDPR/mujoco/cdpr.xml",
                      egl=True, img_hw=(240, 320))
  ```

In **`replay_render.py`**:

* Use a higher res if desired:

  ```python
  env = CDPRLiberoEnv(..., egl=True, img_hw=(480, 640))
  ```

---

## 5) BDDL: where it fits

* Put your problem file in LIBERO (e.g., `LIBERO/lifelong/data/bddl/cdpr_go_near_object.bddl`).
* The env adapter reads the BDDL (or parses the filename) to spawn the target object and define a **goal zone** (e.g., “ee-in-zone within 0.15 m”).
* **BDDL encodes semantics** (objects, init, goals). **Your env** implements placements and success checks.

---

## 6) How this matches SOTA practice

* **Two processes / two envs**: large VLA in a modern stack; simulator/benchmark in its pinned stack; communicate via lightweight RPC. This avoids CUDA/library conflicts and is common in labs.
* **Headless first, videos later**: evaluation runs are done **without live video** (fast, robust). Videos are produced via **replay** to guarantee reproducibility and exact match to logs.
* **Action adapters**: a tiny layer maps model outputs (3-D, 5-D, 7-D; deltas vs absolutes; normalization) to the robot’s control space. Keeps the sim stable while you swap models.

---

## 7) Troubleshooting

### Rendering

* **EGL works in `libero`?**

  * Inside the `libero` shell: `nvidia-smi` should work.
  * Container must be started with graphics caps:

    ```
    docker run --gpus all \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics ...
    ```
  * `ldconfig -p | grep libEGL_nvidia` should show the NVIDIA EGL lib.
  * If EGL still fails, temporarily use OSMesa:

    ```bash
    export MUJOCO_GL=osmesa
    # and pass egl=False to CDPRLiberoEnv(...)
    ```

### Network / SSL

* You’re using HTTP to `127.0.0.1:7071`; no TLS needed. If `requests` ever complains, use stdlib:

  ```python
  import http.client, json
  # see post_json_http(...) helper in your client
  ```

### OpenVLA size / memory

* Keep 7B on the **mujoco_offscreen** env with Torch 2.x + BF16/FP16.
* If you ever must load 7B in low-RAM settings, consider HF `device_map="auto"` + `offload_folder=...` (slower).

### Action dimension mismatch

* If the server returns 3 dims (pos-only), the client pads to 5: `[x,y,z,yaw=0,grip=1]`.
* When you enable yaw+gripper (and later full 7-DoF), the same adapter will pick the extra dims automatically.

---

## 8) Quick sanity commands

List last run:

```bash
ls -lah runs_headless | tail -n +1
python - <<'PY'
import numpy as np, glob
p=sorted(glob.glob("runs_headless/*/trajectory_data.npz"))[-1]
z=np.load(p)
print("t:", len(z["t"]), "ee:", z["ee"].shape, "action:", z["action"].shape)
print("sample action[0]:", z["action"][0])
PY
```

Render latest:

```bash
python VLA_CDPR/mujoco/replay_render.py
```

---

## 9) Optional: moving to 7-DoF

* Update the server to return **7 dims** `[x,y,z,roll,pitch,yaw,gripper]`.
* In the client adapter:

  * use `yaw=a[5]`, `grip=a[6]`,
  * (optional) treat `[x,y,z]` as **delta**: `pos = ee + scale * delta`,
  * clip to workspace, then pass to `env.step([x,y,z,yaw,grip])`.
