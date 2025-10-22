**`SCENE_SWITCHER_GUIDE.md`** you can drop into your `VLA_CDPR/mujoco/` folder:

---

# 🧩 CDPR Scene Switcher & OpenVLA Integration Guide

This document explains how to create **custom LIBERO-style scenes** (e.g., *desk + orange juice*) and run **OpenVLA-driven trajectories** with your CDPR robot in **headless MuJoCo**.

---

## 📁 Directory Overview

```
repo/
├─ LIBERO/                       # LIBERO benchmark repo (scenes & objects)
├─ VLA_CDPR/
│  └─ mujoco/
│     ├─ cdpr.xml                # base CDPR robot definition
│     ├─ cdpr_scene_switcher.py  # builds hybrid scenes (CDPR + LIBERO)
│     ├─ openvla_cdpr_control.py # runs OpenVLA trajectories
│     ├─ wrappers/               # generated scenes (persistent)
│     └─ ...
```

---

## ⚙️ 1. Scene Switching Overview

`cdpr_scene_switcher.py` combines:

* **A LIBERO scene** (e.g., `desk.xml`)
* **Your CDPR robot** (`cdpr.xml`)
* **One or more LIBERO objects** (e.g., `orange_juice.xml`)

into a **new MuJoCo wrapper XML** suitable for simulation or OpenVLA control.

---

## 🧠 Command Reference

### Basic usage

```bash
python cdpr_scene_switcher.py \
  --scene desk \
  --scene_z -0.85 \
  --ee_start 0,0,1.10 \
  --object orange_juice:0.50,0.50,0.00 \
  --object_on_table --table_z 0.75 \
  --object_dynamic --settle_time 1.0 \
  --wrapper_out /root/repo/VLA_CDPR/mujoco/wrappers/desk_juice_wrapper.xml
```

This builds a full hybrid MJCF in:

```
/root/repo/VLA_CDPR/mujoco/wrappers/desk_juice_wrapper.xml
```

---

## 🔧 Arguments Explained

| Argument            | Type / Example          | Description                                                                                  |
| ------------------- | ----------------------- | -------------------------------------------------------------------------------------------- |
| `--scene`           | `desk`, `kitchen`, ...  | Name of LIBERO scene (must exist in `LIBERO/libero/libero/assets/scenes/`)                   |
| `--scene_z`         | float (e.g. `-0.85`)    | Vertical offset to lower or raise the entire scene relative to CDPR base.                    |
| `--ee_start`        | comma-separated `x,y,z` | Starting pose of CDPR end effector. Useful if table is high.                                 |
| `--object`          | `name:x,y,z`            | Adds a LIBERO object at world coordinates `(x, y, z)`.<br>Multiple `--object` flags allowed. |
| `--object_on_table` | flag                    | Automatically offsets each object vertically by `--table_z` to rest on the tabletop.         |
| `--table_z`         | float (e.g. `0.75`)     | Height of the table surface (used only if `--object_on_table` is set).                       |
| `--object_dynamic`  | flag                    | Marks inserted objects as **dynamic** (fall, collide, move). Default is static.              |
| `--settle_time`     | float (sec)             | Simulates a few seconds of settling before saving scene (optional).                          |
| `--wrapper_out`     | path                    | Final destination for wrapper MJCF file. Required if you want a persistent scene.            |
| `--keep`            | flag                    | Prevents temporary directories from being deleted (for debugging).                           |

---

## 🧩 Output Files (in `wrappers/`)

Each scene build creates:

```
wrappers/
├─ desk_juice_wrapper.xml      # main include file (run this!)
├─ desk_zshift.xml             # desk scene with vertical offset
├─ cdpr_ee_override.xml        # CDPR with updated EE start
└─ placed_0_orange_juice.xml   # orange juice placement definition
```

These files reference each other using **absolute paths**, so they’re portable between runs.

---

## 🚀 2. Run OpenVLA Trajectory

Once your wrapper is ready, run the OpenVLA-controlled CDPR:

```bash
python openvla_cdpr_control.py \
  --xml /root/repo/VLA_CDPR/mujoco/wrappers/desk_juice_wrapper.xml \
  --instr "Pick up the orange juice carton, then hover over the desk." \
  --steps 800 \
  --replan 60
```

### Optional Arguments

| Flag       | Description                                        |
| ---------- | -------------------------------------------------- |
| `--steps`  | Total number of simulation steps (default 800).    |
| `--replan` | Number of steps between OpenVLA re-planning calls. |
| `--instr`  | Natural language instruction given to OpenVLA.     |

---

## 🎥 3. Results & Output

After execution, the script automatically saves:

* A video of the **headless simulation** under `openvla_trajectories/`
* `.npz` trajectory data (timestamps, ee positions, targets)
* Console logs with OpenVLA actions

You should see your **CDPR**, **desk**, and **orange juice** rendered in the video.

---

## ✅ Typical Workflow Summary

1. **Generate Scene**

   ```bash
   python cdpr_scene_switcher.py --scene desk ... --wrapper_out wrappers/desk_juice_wrapper.xml
   ```
2. **Run OpenVLA-controlled simulation**

   ```bash
   python openvla_cdpr_control.py --xml wrappers/desk_juice_wrapper.xml --instr "Pick up the juice"
   ```
3. **Review results**

   * Video & data in `openvla_trajectories/`
   * Wrapper reusable for later experiments

---

## 💡 Tips

* If objects spawn under or above the table, tweak `--table_z`.
* To view the wrapper scene manually, open it in MuJoCo viewer:

  ```bash
  python -m mujoco.viewer wrappers/desk_juice_wrapper.xml
  ```
* You can add multiple `--object` flags to populate complex scenes:

  ```bash
  --object orange_juice:0.5,0.5,0.0 --object milk:0.6,0.45,0.0
  ```
