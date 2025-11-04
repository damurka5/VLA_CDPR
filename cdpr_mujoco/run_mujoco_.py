import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
import cv2  # OpenCV for image display
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# Initialize OpenVLA components (do this once, not in the function)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)


def get_vla_action(image: Image.Image, time: float) -> list[float]:
    """
    VLA model that takes an image and returns an action.
    The action is a list of 4 float values for the cable powers (forces).
    First tries to use OpenVLA with "move to red square" prompt,
    falls back to mocked implementation if OpenVLA fails.
    """
    # Try using OpenVLA first
    try:
        prompt = "In: What action should the robot take to move to red square?\nOut:"
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        
        # Convert the action to the expected format (list of 4 floats)
        if action is not None:
            # You may need to adjust this conversion based on OpenVLA's output format
            return [float(x) for x in action[:4]]  # Take first 4 values if more are returned
    
    except Exception as e:
        print(f"OpenVLA failed, falling back to mock implementation: {e}")
    
    # Fallback to mocked implementation if OpenVLA fails
    frequency = 0.5
    amplitude = 20
    base_force = 130  # to counteract gravity, this needs tuning

    actions = [
        base_force + amplitude * np.sin(2 * np.pi * frequency * time),
        base_force + amplitude * np.sin(2 * np.pi * frequency * time + np.pi),
        base_force + amplitude * np.cos(2 * np.pi * frequency * time),
        base_force + amplitude * np.cos(2 * np.pi * frequency * time + np.pi),
    ]
    return actions


class CDPR4:
    def __init__(self, approx=1, pos=np.array([0, 0, 0])):
        self.approx = approx
        self.pos = pos
        self.frame_points = np.array(
            [
                [-1.154, -1.404, 3.220],
                [1.154, -1.404, 3.220],
                [1.154, 1.404, 3.220],
                [-1.154, 1.404, 3.220],
            ]
        )
        self.ee_points = np.array(
            [
                [-0.05, -0.05, 0.05],
                [0.05, -0.05, 0.05],
                [0.05, 0.05, 0.05],
                [-0.05, 0.05, 0.05],
            ]
        )

    def inverse_kinematics(self):
        lengths = []
        for i in range(4):
            ee_pos_global = self.pos + self.ee_points[i]
            vec = ee_pos_global - self.frame_points[i]
            lengths.append(np.linalg.norm(vec))
        return tuple(lengths)


# Initialize simulation
xml_path = "mujoco/cdpr.xml"
simend = 100
print_camera_config = 0

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0


def init_controller(model, data):
    pass


def controller(model, data):
    pass


def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)


def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    button_middle = (
        glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    )
    button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    glfw.get_cursor_pos(window)


def mouse_move(window, xpos, ypos):
    global lastx, lasty
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    if not (button_left or button_middle or button_right):
        return

    width, height = glfw.get_window_size(window)
    mod_shift = (
        glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    )

    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = (
            mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
        )
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)


def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)


# Load model
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()

# Initialize GLFW
glfw.init()
window = glfw.create_window(1200, 900, "CDPR Simulation", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# Initialize visualization
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# Set callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Initialize controller
init_controller(model, data)
mj.set_mjcb_control(controller)

# Initialize robot controller
robot = CDPR4(approx=1, pos=np.array([0, 0, 0]))

# Create camera window
cv2.namedWindow("End-Effector Camera View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("End-Effector Camera View", 640, 480)

# Main simulation loop
while not glfw.window_should_close(window):
    time_prev = data.time

    # Get current end-effector position
    Ac = data.qpos[4:7]
    robot.pos = Ac

    # Get current cable lengths for monitoring
    cur_L_1, cur_L_2, cur_L_3, cur_L_4 = robot.inverse_kinematics()

    # Capture end-effector camera view for VLA model
    ee_cam = mj.MjvCamera()
    ee_cam.type = mj.mjtCamera.mjCAMERA_FIXED
    ee_cam.fixedcamid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera")
    offwidth, offheight = 224, 224  # Standard VLA input size
    offviewport = mj.MjrRect(0, 0, offwidth, offheight)
    off_context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, off_context)
    rgb_buffer = np.zeros((offheight, offwidth, 3), dtype=np.uint8)
    depth_buffer = np.zeros((offheight, offwidth), dtype=np.float32)

    mj.mjv_updateScene(
        model, data, opt, None, ee_cam, mj.mjtCatBit.mjCAT_ALL.value, scene
    )
    mj.mjr_render(offviewport, scene, off_context)
    mj.mjr_readPixels(rgb_buffer, depth_buffer, offviewport, off_context)
    rgb_buffer = np.flipud(rgb_buffer)
    pil_image = Image.fromarray(rgb_buffer)

    # Get action from VLA model
    actions = get_vla_action(pil_image, data.time)
    data.ctrl[0] = actions[0]
    data.ctrl[1] = actions[1]
    data.ctrl[2] = actions[2]
    data.ctrl[3] = actions[3]

    # Step simulation
    step_time = data.time
    while data.time - step_time < 1.0 / 60.0:
        mj.mj_step(model, data)

    if data.time >= simend:
        break

    # Render main window
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # Show camera view in separate window
    bgr_buffer = cv2.cvtColor(rgb_buffer, cv2.COLOR_RGB2BGR)
    cv2.imshow("End-Effector Camera View", bgr_buffer)
    cv2.waitKey(1)

    # Restore default framebuffer
    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW, off_context)

    glfw.swap_buffers(window)
    glfw.poll_events()

# Cleanup
cv2.destroyAllWindows()
glfw.terminate()
