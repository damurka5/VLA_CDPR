# VLA_CDPR/mujoco/cdpr_libero_adapter.py
import os
import numpy as np
import mujoco as mj
from PIL import Image

class CDPRLiberoEnv:
    """
    Minimal Gym-like adapter for LIBERO.
    - Action: 3D target position (absolute or delta).
    - Observation: dict with 'rgb' (overview) and 'ee_pos' (3,), plus 'target_pos' (3,).
    - Reward: negative distance to goal (shaping), success when within radius.
    """
    def __init__(self, xml_path, egl=True, goal_radius=0.15, img_hw=(480, 640)):
        self.xml_path = xml_path
        self.goal_radius = goal_radius
        self.img_h, self.img_w = img_hw

        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data = mj.MjData(self.model)

        # camera setup: overview & ee_camera (from your XML)
        self.overview_cam = mj.MjvCamera()
        self.overview_cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.overview_cam.distance = 3.0
        self.overview_cam.azimuth = 0
        self.overview_cam.elevation = -25

        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.opt = mj.MjvOption()
        mj.mjv_defaultOption(self.opt)
        
        self.yaw_joint_range = np.array([-np.pi, np.pi], dtype=float)
        self.gripper_range   = np.array([0.0, 0.03], dtype=float)
        # find actuator indices once (optional)
        self.act_id_yaw = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "act_ee_yaw")
        self.act_id_gr  = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "act_gripper")


        self.viewport = mj.MjrRect(0, 0, self.img_w, self.img_h)

        # EGL (optional)
        if egl:
            try:
                from mujoco.egl import GLContext
                self.gl = GLContext(max_width=self.img_w, max_height=self.img_h)
                self.gl.make_current()
            except Exception as e:
                print(f"EGL unavailable, falling back: {e}")
                self.gl = None

        self.ctx = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)

        # convenience
        self.dt = self.model.opt.timestep
        self.goal_pos = np.zeros(3)
        self._rng = np.random.RandomState(0)

        # joints: 4 sliders, same as your controller mapping
        self._ctrl_low = np.array([-4.5]*4)
        self._ctrl_high = np.array([ 4.5]*4)

        # state cache
        self._target_pos = np.array([0,0,1.309], dtype=float)

    def _render_overview(self):
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.overview_cam,
                           mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(self.viewport, self.scene, self.ctx)
        rgb = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        depth = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        mj.mjr_readPixels(rgb, depth, self.viewport, self.ctx)
        return np.flipud(rgb)

    def _ee_pos(self):
        # matches your script
        return self.data.qpos[4:7].copy()

    def _inverse_kinematics(self, pos):
        # use the same frame points you used in controller
        frame = np.array([
            [-0.535, -0.755, 1.309],
            [ 0.755, -0.525, 1.309],
            [ 0.535,  0.755, 1.309],
            [-0.755,  0.525, 1.309],
        ])
        return np.linalg.norm(pos[None, :] - frame, axis=1)

    def _position_to_slider_ctrl(self, target_pos):
        # same mapping as your controller.compute_control() returned:
        target_lengths = self._inverse_kinematics(target_pos)
        slider_positions = 0.9 - target_lengths
        return np.clip(slider_positions, self._ctrl_low, self._ctrl_high)

    def _apply_bddl_init(self, bddl_problem_path):
        # Very light parsing: look for keywords
        problem = (os.path.basename(bddl_problem_path) or "").lower()
        if "bowl" in problem:
            obj_body = "bowl"
            obj_xyz  = np.array([0.5, 0.0, 0.0])  # spawn on table/floor
        elif "mug" in problem:
            obj_body = "mug"
            obj_xyz  = np.array([0.4, -0.2, 0.0])
        else:
            obj_body = "target_object"  # fallback red square from your XML
            obj_xyz  = self.model.body_pos[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, obj_body)]

        # set body position programmatically (if authored as a movable body)
        bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, obj_body)
        self.model.body_pos[bid][:] = obj_xyz

        # define goal just above the object
        self.goal_pos = np.array([obj_xyz[0], obj_xyz[1], 1.0], dtype=float)
        return obj_xyz


    # ---- Gym-like API ----
    def reset(self, bddl_problem_path=None, seed=None):
        if seed is not None:
            self._rng.seed(seed)
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

        obj_pos = self._apply_bddl_init(bddl_problem_path)
        # set initial ee near the center (already in XML at 0,0,1.309)
        self._target_pos = self._rng.normal(loc=[0,0,1.309], scale=[0.0,0.0,0.0])

        obs = {
            "rgb": self._render_overview(),
            "ee_pos": self._ee_pos(),
            "target_pos": self.goal_pos.copy(),
            "object_pos": obj_pos,
        }
        return obs

    def step(self, action_5d):
        a = np.asarray(action_5d, dtype=float).copy()
        if a.shape[0] != 5:
            raise ValueError("Action must be [x,y,z,yaw,gripper]")

        # --- position (cables) ---
        pos = np.clip(a[:3], -1.309, 1.309)
        ctrl_cables = self._position_to_slider_ctrl(pos)

        # --- yaw ---
        yaw = np.clip(a[3], self.yaw_joint_range[0], self.yaw_joint_range[1])

        # --- gripper (0..1) -> joint length 0..0.03 ---
        g = np.clip(a[4], 0.0, 1.0)
        g_len = self.gripper_range[0] + g * (self.gripper_range[1] - self.gripper_range[0])

        # Apply to data.ctrl: first 4 are cables as before; then yaw & gripper
        self.data.ctrl[0:4] = ctrl_cables

        # If actuators are appended after cables, set their ctrl slots using IDs
        self.data.ctrl[self.act_id_yaw] = yaw
        self.data.ctrl[self.act_id_gr]  = g_len

        # substeps
        for _ in range(5):
            mj.mj_step(self.model, self.data)

        ee = self._ee_pos()
        dist = np.linalg.norm(ee - self.goal_pos)
        done = bool(dist < self.goal_radius)
        reward = -dist

        obs = {
            "rgb": self._render_overview(),
            "ee_pos": ee.copy(),
            "target_pos": self.goal_pos.copy(),
            # you might also expose yaw and gripper state if a policy needs them
        }
        info = {"dist_to_goal": float(dist)}
        return obs, reward, done, info

    
    def close(self):
        try:
            mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW, self.ctx)
        except:
            pass
