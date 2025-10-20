import os
import mujoco as mj
import numpy as np
import cv2
import time
from datetime import datetime
import imageio

# Try to import EGL for true headless rendering
try:
    from mujoco.egl import GLContext
    EGL_AVAILABLE = True
except ImportError:
    EGL_AVAILABLE = False
    print("EGL not available, falling back to software rendering")

import os
import mujoco as mj
import numpy as np
import cv2
import time
from datetime import datetime
import imageio

# Try to import EGL for true headless rendering
try:
    from mujoco.egl import GLContext
    EGL_AVAILABLE = True
except ImportError:
    EGL_AVAILABLE = False
    print("EGL not available, falling back to software rendering")

def _id(model, objtype, name):
    return mj.mj_name2id(model, objtype, name)

class HeadlessCDPRController:
    def __init__(self, frame_points, initial_pos=np.array([0, 0, 1.309])):
        self.frame_points = frame_points
        self.pos = initial_pos.astype(float)
        self.Kp = 100
        self.Kd = 130
        self.threshold = 0.03
        self.prev_lengths = np.zeros(4)
        self.dt = 1.0/60.0

    def inverse_kinematics(self, pos=None):
        if pos is None:
            pos = self.pos
        diffs = pos[None, :] - self.frame_points
        return np.linalg.norm(diffs, axis=1)

    def update_position(self, new_pos):
        self.pos = new_pos.copy()

    def compute_control(self, target_pos, current_ee_pos):
        self.update_position(current_ee_pos)
        cur_lengths = self.inverse_kinematics()
        target_lengths = self.inverse_kinematics(target_pos)
        # Map target cable lengths -> slider targets via a fixed offset
        slider_positions = 0.9 - target_lengths  # preserves your original "magic number"
        return slider_positions

class HeadlessCDPRSimulation:
    def __init__(self, xml_path, output_dir="trajectory_videos"):
        self.xml_path = xml_path
        self.output_dir = output_dir
        self.model = None
        self.data = None
        self.gl_context = None

        # CDPR frame anchor points (must match XML)
        self.frame_points = np.array([
            [-0.535, -0.755, 1.309],
            [0.755, -0.525, 1.309],
            [0.535,  0.755, 1.309],
            [-0.755, 0.525, 1.309],
        ], dtype=float)

        self.controller = HeadlessCDPRController(self.frame_points)
        self.target_pos = np.array([0, 0, 1.309], dtype=float)

        # Recording
        self.overview_frames = []
        self.ee_camera_frames = []
        self.trajectory_data = []

        os.makedirs(output_dir, exist_ok=True)

    def initialize(self):
        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data = mj.MjData(self.model)
        self.model.opt.timestep = self.controller.dt
        
        self.jnt_yaw = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "ee_yaw")
        self.jnt_yaw_qadr = self.model.jnt_qposadr[self.jnt_yaw]  # index into qpos for yaw angle

        # IDs we’ll need
        self.body_ee   = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "ee_base")  # EE body
        self.cam_id    = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera")

        # Sliders (cables)
        self.act_sliders = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "slider_1_pos"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "slider_2_pos"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "slider_3_pos"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "slider_4_pos"),
        ]
        # Tool actuators
        self.act_yaw     = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "act_ee_yaw")
        self.act_gripper = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "act_gripper")

        # Target object (body + geom). If the geom is unnamed, we’ll find it by body.
        self.body_target = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "target_object")
        try:
            self.geom_target = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "target_box")
        except Exception:
            # Fallback: first geom belonging to target body
            self.geom_target = int(np.where(self.model.geom_bodyid == self.body_target)[0][0])
                
        self._setup_offscreen_rendering()
        mj.mj_forward(self.model, self.data)

        print("Headless CDPR Simulation initialized successfully!")
        print(f"Using {'EGL' if EGL_AVAILABLE else 'software'} rendering")

    def _setup_offscreen_rendering(self):
        self.overview_cam = mj.MjvCamera()
        self.ee_cam = mj.MjvCamera()

        self.overview_cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.overview_cam.distance = 3.0
        self.overview_cam.azimuth = 0
        self.overview_cam.elevation = -25

        self.ee_cam.type = mj.mjtCamera.mjCAMERA_FIXED
        self.ee_cam.fixedcamid = self.cam_id

        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.opt = mj.MjvOption()
        mj.mjv_defaultOption(self.opt)

        self.offwidth, self.offheight = 640, 480
        self.offviewport = mj.MjrRect(0, 0, self.offwidth, self.offheight)

        if EGL_AVAILABLE:
            self.gl_context = GLContext(max_width=self.offwidth, max_height=self.offheight)
            self.gl_context.make_current()
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

    def capture_frame(self, camera, camera_name):
        try:
            mj.mjv_updateScene(self.model, self.data, self.opt, None, camera,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(self.offviewport, self.scene, self.context)
            rgb = np.zeros((self.offheight, self.offwidth, 3), dtype=np.uint8)
            depth = np.zeros((self.offheight, self.offwidth), dtype=np.float32)
            mj.mjr_readPixels(rgb, depth, self.offviewport, self.context)
            return np.flipud(rgb)
        except Exception as e:
            print(f"Error capturing frame from {camera_name}: {e}")
            return np.zeros((self.offheight, self.offwidth, 3), dtype=np.uint8)

    def get_end_effector_position(self):
        return self.data.xpos[self.body_ee].copy()

    def get_target_position(self):
        return self.data.xpos[self.body_target].copy()

    def set_gripper(self, opening_m):
        opening = float(np.clip(opening_m, 0.0, 0.06))  # your bigger range
        self.data.ctrl[self.act_gripper] = opening

    def open_gripper(self):  self.set_gripper(0.06)
    def close_gripper(self): self.set_gripper(0.0)

    def set_yaw(self, yaw_rad):
        self.data.ctrl[self.act_yaw] = float(np.clip(yaw_rad, -np.pi, np.pi))
    
    def set_target_position(self, target_pos):
        target_pos = np.asarray(target_pos, dtype=float)
        if np.all((-1.309 <= target_pos) & (target_pos <= 1.309)):
            self.target_pos = target_pos
            ee_pos = self.get_end_effector_position()
            self.controller.prev_lengths = self.controller.inverse_kinematics(ee_pos)
            return True
        return False

    def check_success(self):
        ee_pos = self.get_end_effector_position()
        return np.linalg.norm(ee_pos - self.target_pos) < self.controller.threshold

    # === Gripper / yaw helpers ===
    def set_gripper(self, opening_m):
        """Set desired opening for left finger (right follows). Range [0, 0.03]."""
        opening = float(np.clip(opening_m, 0.0, 0.03))
        self.data.ctrl[self.act_gripper] = opening

    def open_gripper(self):
        self.set_gripper(0.03)

    def close_gripper(self):
        self.set_gripper(0.0)

    def set_yaw(self, yaw_rad):
        self.data.ctrl[self.act_yaw] = float(np.clip(yaw_rad, -np.pi, np.pi))

    def record_trajectory_step(self):
        ee_pos = self.get_end_effector_position()
        slider_q = [self.data.qpos[j] for j in range(4)]  # fine for logging
        cable_lengths = self.controller.inverse_kinematics(ee_pos)
        self.trajectory_data.append({
            'timestamp': self.data.time,
            'ee_position': ee_pos.copy(),
            'target_position': self.target_pos.copy(),
            'slider_positions': slider_q.copy(),
            'cable_lengths': cable_lengths.copy(),
            'control_signals': self.data.ctrl.copy() if self.model.nu > 0 else np.zeros(0),
        })
        
    def goto(self, world_xyz, max_steps=900, tol=0.03, capture_every_n=3):
        """Drive EE to world_xyz with your cable controller."""
        self.set_target_position(np.asarray(world_xyz, dtype=float))
        steps, total = 0, 0
        while steps < max_steps:
            capture = (total % capture_every_n == 0)
            self.run_simulation_step(capture_frame=capture)
            total += 1; steps += 1
            if np.linalg.norm(self.get_end_effector_position() - self.target_pos) < tol:
                return True, steps
        return False, steps

    def has_finger_contact(self):
        # any contact involving target geom and a finger geom
        finger_geoms = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "finger_l_link"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "finger_r_link"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "finger_l_tip"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "finger_r_tip"),
        ]
        tgt = self.geom_target
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 == tgt and c.geom2 in finger_geoms) or (c.geom2 == tgt and c.geom1 in finger_geoms):
                return True
        return False

    def lifted_enough(self, z_start, min_rise=0.02):
        z_now = self.get_target_position()[2]
        return (z_now - z_start) >= min_rise
    
    def _angle_wrap(self, a):
        """Wrap angle to [-pi, pi]."""
        return (a + np.pi) % (2*np.pi) - np.pi

    def rotate_to(self, yaw_target, duration=1.0, hold_xyz=None, capture_every_n=2):
        """
        Smoothly rotate yaw to yaw_target [rad] over 'duration' seconds.
        If hold_xyz is provided, we keep position controller targeting that XYZ while rotating.
        """
        dt = float(self.controller.dt)
        steps = max(1, int(round(duration / dt)))

        # Read current yaw directly from joint qpos
        yaw_now = float(self.data.qpos[self.jnt_yaw_qadr])
        # shortest arc
        dyaw = self._angle_wrap(yaw_target - yaw_now)

        # Fix the translational target if requested
        if hold_xyz is not None:
            self.set_target_position(np.asarray(hold_xyz, dtype=float))

        for k in range(steps):
            alpha = (k + 1) / steps
            yaw_cmd = yaw_now + dyaw * alpha
            self.set_yaw(yaw_cmd)
            capture = (k % capture_every_n == 0)
            self.run_simulation_step(capture_frame=capture)


    def run_simulation_step(self, capture_frame=True):
        ee_pos = self.get_end_effector_position()
        control_signals = self.controller.compute_control(self.target_pos, ee_pos)
        # Apply slider targets by actuator index
        for j, act_id in enumerate(self.act_sliders):
            self.data.ctrl[act_id] = control_signals[j]

        mj.mj_step(self.model, self.data)

        if capture_frame:
            self.overview_frames.append(self.capture_frame(self.overview_cam, "overview"))
            self.ee_camera_frames.append(self.capture_frame(self.ee_cam, "ee_camera"))

        self.record_trajectory_step()

    def run_trajectory(self, target_positions, trajectory_name="trajectory",
                       max_steps_per_target=600, capture_every_n=3):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_dir = os.path.join(self.output_dir, f"{trajectory_name}_{timestamp}")
        os.makedirs(trajectory_dir, exist_ok=True)
        print(f"Starting trajectory: {trajectory_name}")

        self.overview_frames, self.ee_camera_frames, self.trajectory_data = [], [], []
        total_steps = 0
        trajectory_success = True

        # Demo: open before motion
        self.open_gripper()
        for i, target_pos in enumerate(target_positions):
            print(f"Moving to target {i+1}/{len(target_positions)}: {target_pos}")
            if not self.set_target_position(target_pos):
                print(f"Invalid target position: {target_pos}")
                trajectory_success = False
                break

            steps_for_target = 0
            target_reached = False
            while steps_for_target < max_steps_per_target:
                capture = (total_steps % capture_every_n == 0)
                self.run_simulation_step(capture_frame=capture)
                total_steps += 1
                steps_for_target += 1

                if self.check_success():
                    print(f"Target {i+1} reached in {steps_for_target} steps")
                    target_reached = True
                    break

            if not target_reached:
                print(f"Timeout reaching target {i+1} after {max_steps_per_target} steps")
                trajectory_success = False

        # Demo: close after motion
        self.close_gripper()
        # step a little to see closing motion in video
        for _ in range(20):
            self.run_simulation_step(capture_frame=True)

        self.save_trajectory_results(trajectory_dir, trajectory_name)
        return trajectory_success
    
    def run_grasp_demo(self, object_xy=(0.5, 0.5), hover_z=0.35, grasp_z=0.06,
                   lift_z=0.35, yaw=0.0, name="grasp_demo"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_dir = os.path.join(self.output_dir, f"{name}_{timestamp}")
        os.makedirs(trajectory_dir, exist_ok=True)
        self.overview_frames, self.ee_camera_frames, self.trajectory_data = [], [], []

        ox, oy = object_xy
        hold_hover = np.array([ox, oy, hover_z], dtype=float)

        # Start open, move above object
        self.open_gripper()
        print("➡️  Move above object (hover).")
        ok, _ = self.goto(hold_hover, max_steps=120);  print("  done" if ok else "  timeout")

        # --- YAW TEST 1: rotate to the requested yaw while hovering ---
        print(f"🔁  Yaw to {yaw:.2f} rad while hovering.")
        self.rotate_to(yaw_target=float(2*yaw), duration=2.5, hold_xyz=hold_hover)

        # Descend to grasp height (keep that yaw)
        print("➡️  Descend to grasp height.")
        ok, _ = self.goto([ox, oy, grasp_z], max_steps=120);  print("  done" if ok else "  timeout")

        for _ in range(20): self.run_simulation_step(capture_frame=True)

        # Close & squeeze
        print("➡️  Close gripper.")
        self.close_gripper()
        for _ in range(40): self.run_simulation_step(capture_frame=True)

        z_before = self.get_target_position()[2]

        # Lift up
        print("➡️  Lift.")
        ok, _ = self.goto([ox, oy, lift_z], max_steps=150);  print("  done" if ok else "  timeout")

        # --- YAW TEST 2: rotate 90° while lifted (tests stability while carrying) ---
        yaw2 = float(self._angle_wrap(yaw + np.pi/2.0))
        print(f"🔁  Yaw to {yaw2:.2f} rad while lifted.")
        self.rotate_to(yaw_target=-yaw2, duration=2.5, hold_xyz=[ox, oy, lift_z])

        # Optional: yaw back to original
        print(f"🔁  Yaw back to {yaw:.2f} rad.")
        # self.rotate_to(yaw_target=float(yaw), duration=1.0, hold_xyz=[ox, oy, lift_z])

        # Return to hover, open/close a couple times (exercise fingers under new yaw)
        print("➡️  Move above object again (hover).")
        ok, _ = self.goto(hold_hover, max_steps=120);  print("  done" if ok else "  timeout")

        self.open_gripper()
        for _ in range(30): self.run_simulation_step(capture_frame=True)
        self.close_gripper()
        for _ in range(30): self.run_simulation_step(capture_frame=True)

        # Check grasp success
        for _ in range(30): self.run_simulation_step(capture_frame=True)
        grasp_ok = self.lifted_enough(z_before, min_rise=0.02) or self.has_finger_contact()

        self.save_trajectory_results(trajectory_dir, name)
        print(f"✅ Grasp {'SUCCEEDED' if grasp_ok else 'FAILED'}")
        return grasp_ok

    def save_trajectory_results(self, trajectory_dir, trajectory_name):
        """Save all trajectory data and videos"""
        print("Saving trajectory results...")
        
        # Save videos if we have frames
        if self.overview_frames:
            try:
                overview_video_path = os.path.join(trajectory_dir, "overview_video.mp4")
                self.save_video(self.overview_frames, overview_video_path, fps=20)
                print(f"Overview video saved: {overview_video_path}")
            except Exception as e:
                print(f"Error saving overview video: {e}")
        
        if self.ee_camera_frames:
            try:
                ee_video_path = os.path.join(trajectory_dir, "ee_camera_video.mp4")
                self.save_video(self.ee_camera_frames, ee_video_path, fps=20)
                print(f"End-effector video saved: {ee_video_path}")
            except Exception as e:
                print(f"Error saving EE camera video: {e}")
        
        # Save trajectory data
        try:
            trajectory_file = os.path.join(trajectory_dir, "trajectory_data.npz")
            self.save_trajectory_data(trajectory_file)
            print(f"Trajectory data saved: {trajectory_file}")
        except Exception as e:
            print(f"Error saving trajectory data: {e}")
        
        # Save summary
        try:
            self.save_summary(trajectory_dir, trajectory_name)
        except Exception as e:
            print(f"Error saving summary: {e}")
        
        print(f"Results saved to: {trajectory_dir}")
    
    def save_video(self, frames, filepath, fps=30):
        """Save frames as MP4 video using imageio (more reliable than OpenCV)"""
        if not frames:
            return
        
        # Use imageio for more reliable video writing
        with imageio.get_writer(filepath, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
    
    def save_trajectory_data(self, filepath):
        """Save trajectory data as numpy file"""
        if not self.trajectory_data:
            return
            
        trajectory_dict = {}
        
        # Extract arrays from trajectory data
        arrays_to_save = ['timestamp', 'ee_position', 'target_position', 
                         'slider_positions', 'cable_lengths', 'control_signals']
        
        for key in arrays_to_save:
            if key in self.trajectory_data[0]:
                trajectory_dict[key] = np.array([data[key] for data in self.trajectory_data])
        
        np.savez(filepath, **trajectory_dict)
    
    def save_summary(self, trajectory_dir, trajectory_name):
        """Save a text summary of the trajectory"""
        summary_file = os.path.join(trajectory_dir, "summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"CDPR Trajectory Summary\n")
            f.write(f"=======================\n")
            f.write(f"Trajectory: {trajectory_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total frames captured: {len(self.overview_frames)}\n")
            f.write(f"Total simulation steps: {len(self.trajectory_data)}\n")
            f.write(f"Simulation time: {self.data.time:.2f} seconds\n")
            if self.trajectory_data:
                f.write(f"Final EE position: {self.trajectory_data[-1]['ee_position']}\n")
                f.write(f"Final target position: {self.trajectory_data[-1]['target_position']}\n")
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'context'):
            try:
                mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW, self.context)
            except:
                pass
        if self.gl_context:
            try:
                self.gl_context.free()
            except:
                pass
        print("Simulation cleanup completed")

# def main():
#     # Path to your XML file
#     xml_path = "cdpr.xml"
    
#     # Check if XML file exists
#     if not os.path.exists(xml_path):
#         print(f"Error: XML file not found at {xml_path}")
#         print("Please ensure the cdpr.xml file exists")
#         return
    
#     # Create simulation
#     sim = HeadlessCDPRSimulation(xml_path, output_dir="trajectory_results")
    
#     try:
#         sim.initialize()
        
#         # Define trajectory waypoints
#         trajectories = {
#             "simple_test": [
#                 np.array([0.5, 0.5, 0.5]),
#                 np.array([0, 0, 0.1])
#             ],
#             # "square_trajectory": [
#             #     [0.5, 0.5, 1.0],
#             #     [0.5, -0.5, 1.0],
#             #     [-0.5, -0.5, 1.0],
#             #     [-0.5, 0.5, 1.0],
#             #     [0.0, 0.0, 1.309]
#             # ]
#         }
        
#         # Run each trajectory
#         for traj_name, waypoints in trajectories.items():
#             print(f"\n{'='*50}")
#             print(f"Running trajectory: {traj_name}")
#             print(f"{'='*50}")
            
#             success = sim.run_trajectory(waypoints, traj_name, max_steps_per_target=300)
            
#             if success:
#                 print(f"✓ Trajectory '{traj_name}' completed successfully!")
#             else:
#                 print(f"⚠ Trajectory '{traj_name}' had some issues")
            
#             # Small pause between trajectories
#             time.sleep(1)
            
#     except KeyboardInterrupt:
#         print("\nTrajectory recording interrupted by user")
#     except Exception as e:
#         print(f"Error during simulation: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         sim.cleanup()

def main():
    xml_path = "cdpr.xml"
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return

    sim = HeadlessCDPRSimulation(xml_path, output_dir="trajectory_results")
    try:
        sim.initialize()

        # Try a grasp at the red block's XY
        success = sim.run_grasp_demo(object_xy=(0.5, 0.5), hover_z=0.35, grasp_z=0.06, lift_z=0.35, yaw=0.0)
        print("✓ grasp flow finished:", success)

    except Exception as e:
        print("Error during simulation:", e)
        import traceback; traceback.print_exc()
    finally:
        sim.cleanup()

if __name__ == "__main__":
    main()