#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time
from pathlib import Path
import numpy as np
from datetime import datetime

import mujoco as mj
from PIL import Image
import imageio

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# repo-local import of your simulator
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))  # add VLA_CDPR/mujoco
from headless_cdpr_egl import HeadlessCDPRSimulation

# -------------------- OpenVLA wrapper --------------------

class OpenVLAPlanner:
    def __init__(self, model_name="openvla/openvla-7b"):
        # pick device + dtype
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Prefer bfloat16 on CUDA; fall back to float32 if needed
        self.model_dtype = torch.bfloat16 if use_cuda else torch.float32
        print(f"OpenVLA using device: {self.device} dtype: {self.model_dtype}")

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        try:
            self.vla = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=self.model_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device)
        except Exception as e:
            # Fallback if bf16 fails (older GPUs)
            print(f"[OpenVLA] bf16 load failed ({e}). Falling back to float32.")
            self.model_dtype = torch.float32
            self.vla = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=self.model_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device)

        self.vla.eval()

    @torch.no_grad()
    def get_target_position(self, image, instruction="go near the orange juice"):
        """
        Legacy 3-DoF API (kept for compatibility). Returns np.array([x,y,z]).
        """
        act = self._plan_raw(image, instruction)
        # ensure length >= 3
        if isinstance(act, (list, tuple)):
            act = np.array(act, dtype=float)
        elif isinstance(act, torch.Tensor):
            act = act.detach().float().cpu().numpy()
        elif not isinstance(act, np.ndarray):
            act = np.zeros(5, dtype=float)

        if act.size < 3:
            act = np.pad(act, (0, 3 - act.size), mode="constant")

        return act[:3]

    @torch.no_grad()
    def plan_5dof(self, image, instruction):
        """
        New 5-DoF API: returns np.array([x, y, z, yaw, grip]).
        """
        act = self._plan_raw(image, instruction)
        if isinstance(act, torch.Tensor):
            act = act.detach().float().cpu().numpy()
        elif isinstance(act, (list, tuple)):
            act = np.array(act, dtype=float)
        elif not isinstance(act, np.ndarray):
            act = np.zeros(5, dtype=float)

        # pad/truncate to length 5
        if act.size < 5:
            act = np.pad(act, (0, 5 - act.size), mode="constant")
        else:
            act = act[:5]
        return act

    def _plan_raw(self, image, instruction, unnorm_key="bridge_orig"):
        """
        Run the model once. Cast only image tensors to model dtype (bf16/fp32),
        keep token ids as torch.long.
        """
        # Ensure PIL RGB
        if isinstance(image, np.ndarray):
            from PIL import Image as _PILImage
            image = _PILImage.fromarray(image)
        if getattr(image, "mode", None) != "RGB":
            image = image.convert("RGB")

        prompt = f"In: {instruction}\nOut:"

        # Build processor inputs (PyTorch tensors)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        # Move to device with CORRECT dtypes per field
        casted = {}
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                continue
            if "input_ids" in k or k.endswith("ids"):
                # token indices must be long
                casted[k] = v.to(self.device, dtype=torch.long)
            elif "attention_mask" in k or "token_type_ids" in k:
                # masks/segment ids: keep integer/bool, just move device
                casted[k] = v.to(self.device)
            elif "pixel" in k or v.dim() == 4:
                # image-like tensors -> model dtype (bf16/fp32)
                casted[k] = v.to(self.device, dtype=self.model_dtype)
            else:
                # default: move only
                casted[k] = v.to(self.device)

        # Use model helper if available
        if hasattr(self.vla, "predict_action"):
            # (optional) autocast for safety on CUDA bf16
            if self.device.type == "cuda" and self.model_dtype == torch.bfloat16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return self.vla.predict_action(**casted, unnorm_key=unnorm_key, do_sample=False)
            else:
                return self.vla.predict_action(**casted, unnorm_key=unnorm_key, do_sample=False)

        # Fallback: generate tokens and parse numbers
        out_ids = self.vla.generate(**casted, max_new_tokens=32)
        txt = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        import re
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt)
        return np.array([float(n) for n in nums], dtype=float) if nums else np.zeros(5, dtype=float)

# -------------------- Action mapping --------------------

class ActionMapper5DoF:
    """
    Map OpenVLA action vector (various lengths/scales) -> (xyz, yaw, grip)
    with clamping to workspace.
    """
    def __init__(self,
                 x_bound=(-0.9, 0.9),
                 y_bound=(-0.9, 0.9),
                 z_bound=(0.10, 1.30),
                 gripper_range=(0.0, 0.06)):
        self.xb, self.yb, self.zb = x_bound, y_bound, z_bound
        self.gr_min, self.gr_max = gripper_range

    @staticmethod
    def _maybe_denorm(v, low, high):
        # If looks normalized (|v| <= 1.2), map from [-1,1]→[low,high], else clip to [low,high].
        if np.isfinite(v) and abs(v) <= 1.2:
            return low + 0.5 * (v + 1.0) * (high - low)
        return np.clip(v, low, high)

    @staticmethod
    def _wrap_pi(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def map(self, act_vec: np.ndarray, fallback_xyz: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        act_vec:  (>=3) x,y,z,[yaw],[grip]
        returns: (xyz, yaw, grip_opening_m)
        """
        a = np.array(act_vec, dtype=float).flatten()
        # x,y,z
        tx = self._maybe_denorm(a[0] if a.size > 0 else fallback_xyz[0], *self.xb)
        ty = self._maybe_denorm(a[1] if a.size > 1 else fallback_xyz[1], *self.yb)
        tz = self._maybe_denorm(a[2] if a.size > 2 else fallback_xyz[2], *self.zb)
        xyz = np.array([tx, ty, tz], dtype=float)

        # yaw
        if a.size >= 4:
            yaw_raw = a[3]
            yaw = self._wrap_pi(yaw_raw if abs(yaw_raw) > 1.2 else yaw_raw * np.pi)
        else:
            yaw = 0.0

        # gripper
        if a.size >= 5:
            g = a[4]
            if -0.01 <= g <= 1.01:     # normalized 0..1 (or -1..1)
                g = max(0.0, min(1.0, g))
                grip = self.gr_min + g * (self.gr_max - self.gr_min)
            else:                      # maybe already in meters
                grip = np.clip(g, self.gr_min, self.gr_max)
        else:
            grip = self.gr_max  # open by default

        return xyz, yaw, grip

# -------------------- Runner --------------------

class OpenVLACDPRRunner:
    def __init__(self,
                 xml_path: str,
                 instruction: str = "Pick up the orange juice, then hover",
                 model_name: str = "openvla/openvla-7b",
                 out_dir: str = "openvla_runs"):
        self.xml_path = xml_path
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # sim
        self.sim = HeadlessCDPRSimulation(xml_path, output_dir=str(self.out_dir))
        # vla
        self.vla = OpenVLAPlanner(model_name=model_name)
        self.mapper = ActionMapper5DoF()

        # bookkeeping
        self.frames = []
        self.log = []

        # instruction
        self.instruction = instruction

    def initialize(self):
        self.sim.initialize()
        # Resolve body pose once so we can hold XYZ while we rotate, etc.
        self.current_target = self.sim.get_end_effector_position().copy()

    def capture_ee_rgb(self):
        # Use the EE camera that your sim already configures
        # return self.sim.capture_frame(self.sim.ee_cam, "ee_camera")
        
        # Use the overview camera 
        return self.sim.capture_frame(self.sim.overview_cam, "overview")
    

    def step_apply(self, xyz: np.ndarray, yaw: float, grip: float):
        # Position (via your cable-length mapping)
        self.sim.set_target_position(xyz)
        # Yaw and gripper
        if hasattr(self.sim, "set_yaw"):
            self.sim.set_yaw(float(yaw))
        if hasattr(self.sim, "set_gripper"):
            self.sim.set_gripper(float(grip))
        # one physics step + record
        self.sim.run_simulation_step(capture_frame=True)

    def run(self,
            horizon_steps: int = 800,
            replan_every: int = 60,
            settle_steps_after_close: int = 20):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.out_dir / f"openvla_cdpr_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Runner] instruction: {self.instruction}")
        print(f"[Runner] wrapper: {self.xml_path}")

        # Start open
        if hasattr(self.sim, "open_gripper"): self.sim.open_gripper()

        for t in range(horizon_steps):
            # Replan from the EE camera periodically
            if t % replan_every == 0:
                rgb = self.capture_ee_rgb()
                act = self.vla.plan_5dof(rgb, self.instruction)
                xyz, yaw, grip = self.mapper.map(act, self.sim.get_end_effector_position())
                self.current_target = xyz
                self.current_yaw = yaw
                self.current_grip = grip
                print(f"[t={t:04d}] VLA action -> xyz={xyz.round(3)}, yaw={yaw:.2f}, grip={grip:.3f}")

            # Apply the most recent target each step (lets the low-level controller chase it)
            self.step_apply(self.current_target, self.current_yaw, self.current_grip)

            # Simple “close on contact” heuristic (optional): if close to tabletop height and
            # fingers touch object (you already have has_finger_contact), you could auto close.
            # Example (commented):
            # if t % 10 == 0 and self.sim.has_finger_contact():
            #     self.sim.close_gripper()
            #     for _ in range(settle_steps_after_close): self.sim.run_simulation_step(capture_frame=True)

            # Log
            ee = self.sim.get_end_effector_position()
            self.log.append(dict(t=self.sim.data.time,
                                 ee=ee.copy(),
                                 target=self.current_target.copy(),
                                 yaw=self.current_yaw,
                                 grip=self.current_grip))

        # Save video + data via your existing utilities
        # (We’ll mirror your helper naming)
        self.sim.save_trajectory_results(str(run_dir), "openvla_cdpr")
        # Also save a consolidated npz
        npz_path = run_dir / "openvla_log.npz"
        import numpy as np
        np.savez(npz_path,
                 t=np.array([r["t"] for r in self.log]),
                 ee=np.vstack([r["ee"] for r in self.log]),
                 target=np.vstack([r["target"] for r in self.log]),
                 yaw=np.array([r["yaw"] for r in self.log]),
                 grip=np.array([r["grip"] for r in self.log]))
        print(f"[Runner] saved {npz_path}")

        self.sim.cleanup()

# -------------------- CLI --------------------

def main():
    import argparse
    ap = argparse.ArgumentParser("OpenVLA → CDPR 5DoF controller")
    ap.add_argument("--xml", default="cdpr.xml",
                    help="Path to MJCF to run (use wrapper from cdpr_scene_switcher.py to include scenes/objects).")
    ap.add_argument("--model", default="openvla/openvla-7b", help="OpenVLA model id")
    ap.add_argument("--instr", default="Pick up the orange juice carton, then hover.",
                    help="Natural language instruction")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--replan", type=int, default=60, help="Steps between VLA replans")
    args = ap.parse_args()

    runner = OpenVLACDPRRunner(xml_path=args.xml, instruction=args.instr, model_name=args.model)
    try:
        runner.initialize()
        runner.run(horizon_steps=args.steps, replan_every=args.replan)
    except Exception as e:
        print("Error:", e)
        import traceback; traceback.print_exc()
    finally:
        pass

if __name__ == "__main__":
    main()
