import os
import glfw
import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
import cv2
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import time

class CDPR4:
    def __init__(self, approx=1, pos=np.array([0, 0, 0])):
        self.approx = approx
        self.pos = pos
        self.frame_points = np.array([
            [-1.154, -1.404, 3.220],
            [1.154, -1.404, 3.220],
            [1.154, 1.404, 3.220],
            [-1.154, 1.404, 3.220]
        ])
        self.ee_points = np.array([
            [-0.05, -0.05, 0.05],
            [0.05, -0.05, 0.05],
            [0.05, 0.05, 0.05],
            [-0.05, 0.05, 0.05]
        ])
    
    def inverse_kinematics(self):
        lengths = []
        for i in range(4):
            ee_pos_global = self.pos + self.ee_points[i]
            vec = ee_pos_global - self.frame_points[i]
            lengths.append(np.linalg.norm(vec))
        return tuple(lengths)

class OpenVLAController:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load OpenVLA model and processor
        self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)
        
        # Warm up the model
        self._warm_up()
    
    def _warm_up(self):
        """Initial warm-up run for the model"""
        dummy_image = Image.new('RGB', (640, 480), color='black')
        dummy_prompt = "In: Move to center\nOut:"
        inputs = self.processor(dummy_prompt, dummy_image).to(self.device, dtype=torch.bfloat16)
        _ = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    
    def process_frame(self, cv2_image, instruction):
        """Process a frame and generate cable length adjustments"""
        # Convert OpenCV frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        
        # Prepare prompt
        prompt = f"In: What cable adjustments should the robot make to {instruction}?\nOut:"
        
        # Process image and generate action
        inputs = self.processor(prompt, pil_image).to(self.device, dtype=torch.bfloat16)
        
        start_time = time.time()
        with torch.no_grad():
            action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        inference_time = time.time() - start_time
        
        print(f"Inference time: {inference_time:.4f} seconds")
        
        # Convert action to 4 cable length changes (delta L1, delta L2, delta L3, delta L4)
        if isinstance(action, dict) and 'cable_adjustments' in action:
            return action['cable_adjustments']
        elif isinstance(action, (list, np.ndarray)) and len(action) == 4:
            return action
        else:
            print(f"Unexpected action format: {action}")
            return [0, 0, 0, 0]  # Default to no movement

# Initialize simulation
xml_path = 'mujoco/cdpr.xml'
simend = 100
print_camera_config = 0

# Initialize GLFW and MuJoCo
glfw.init()
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
window = glfw.create_window(1200, 900, "CDPR Simulation with OpenVLA", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# Initialize visualization
cam = mj.MjvCamera()
opt = mj.MjvOption()
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# Initialize controllers
robot = CDPR4(approx=1, pos=np.array([0, 0, 0]))
vla_controller = OpenVLAController()

# Main simulation loop
cv2.namedWindow("End-Effector Camera View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("End-Effector Camera View", 640, 480)

current_instruction = None
executing_action = False
action_start_time = 0
action_duration = 2.0  # seconds to complete action
target_cable_lengths = None

while not glfw.window_should_close(window):
    # Get current state
    Ac = data.qpos[4:7]
    robot.pos = Ac
    cur_L = robot.inverse_kinematics()
    
    # Get camera frame
    offwidth, offheight = 640, 480
    offviewport = mj.MjrRect(0, 0, offwidth, offheight)
    con = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, con)
    rgb_buffer = np.zeros((offheight, offwidth, 3), dtype=np.uint8)
    depth_buffer = np.zeros((offheight, offwidth), dtype=np.float32)
    
    ee_cam = mj.MjvCamera()
    ee_cam.type = mj.mjtCamera.mjCAMERA_FIXED
    ee_cam.fixedcamid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera")
    
    mj.mjv_updateScene(model, data, opt, None, ee_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(offviewport, scene, con)
    mj.mjr_readPixels(rgb_buffer, depth_buffer, offviewport, con)
    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW, con)
    
    rgb_buffer = np.flipud(rgb_buffer)
    bgr_buffer = cv2.cvtColor(rgb_buffer, cv2.COLOR_RGB2BGR)
    cv2.imshow("End-Effector Camera View", bgr_buffer)
    cv2.waitKey(1)
    
    # Get new instruction
    if not executing_action and data.time > 2 and current_instruction is None:
        print("\nCurrent position:", Ac)
        instruction = input("Enter instruction (or 'quit' to exit): ")
        if instruction.lower() == 'quit':
            break
        
        current_instruction = instruction
        executing_action = True
        action_start_time = data.time
        
        # Process frame with OpenVLA
        pil_image = Image.fromarray(cv2.cvtColor(bgr_buffer, cv2.COLOR_BGR2RGB))
        cable_deltas = vla_controller.process_frame(bgr_buffer, instruction)
        
        # Convert deltas to target lengths
        target_cable_lengths = [cur_L[i] + cable_deltas[i] for i in range(4)]
        print("Target cable lengths:", target_cable_lengths)
    
    # Execute action
    if executing_action:
        k_p = 200
        for i in range(4):
            data.ctrl[i] = k_p * (target_cable_lengths[i] - cur_L[i])
        
        # Check if action is complete
        if all(abs(cur_L[i] - target_cable_lengths[i]) < 0.01 for i in range(4)):
            print("Action completed!")
            executing_action = False
            current_instruction = None
        elif data.time - action_start_time > action_duration:
            print("Action timed out")
            executing_action = False
            current_instruction = None
    
    # Step simulation
    step_time = data.time
    while (data.time - step_time < 1.0/60.0):
        mj.mj_step(model, data)
    
    if data.time >= simend:
        break
    
    # Render main window
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

# Cleanup
cv2.destroyAllWindows()
glfw.terminate()