import os
import glfw
import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
import cv2  # OpenCV for image display

class CDPR4:
    def __init__(self, approx=1, pos=np.array([0, 0, 0])):
        self.approx = approx
        self.pos = pos  # Current end-effector position in world frame
        self.frame_points = np.array([
            [-0.535, -0.755, 1.309],
            [0.755, -0.525, 1.309],
            [0.535, 0.755, 1.309],
            [-0.755, 0.525, 1.309]
        ])
        self.ee_points = np.array([
            [-0.05, -0.05, 0.05],
            [0.05, -0.05, 0.05],
            [0.05, 0.05, 0.05],
            [-0.05, 0.05, 0.05]
        ])
        # Store previous cable lengths for derivative term
        self.prev_lengths = np.zeros(4)
    
    def inverse_kinematics(self, pos=None):
        if pos is None:
            pos = self.pos
            
        lengths = []
        for i in range(4):
            vec = pos - self.frame_points[i]
            lengths.append(np.linalg.norm(vec))
        return tuple(lengths)

# Initialize simulation
xml_path = 'mujoco/cdpr.xml'
simend = 100
print_camera_config = 0

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# PD controller parameters
Kp = 100  # Proportional gain
Kd = 130   # Derivative gain

def init_controller(model, data):
    # Initialize any controller-specific parameters here
    pass

def controller(model, data):
    # This function is called by MuJoCo at each simulation step
    # We'll keep the main control logic in the main loop for clarity
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
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
    mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)

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

# Initialize robot controller with correct initial position
initial_pos = np.array([0, 0, 1.309])
robot = CDPR4(approx=1, pos=initial_pos)

# Create camera window
cv2.namedWindow("End-Effector Camera View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("End-Effector Camera View", 640, 480)

# Main simulation loop
received_goal = False
threshold = 0.03
dt = 1.0/60.0  # Simulation time step
target_pos = np.array([0, 0, 1.309])  # Initialize target position

while not glfw.window_should_close(window):
    time_prev = data.time
    
    # Get current end-effector position from simulation and update robot's position
    Ac = data.qpos[4:7]  # Current end-effector position in world frame
    robot.pos = Ac.copy()  # Update the robot's current position
    
    # User input for new position
    # if data.time > 2 and not received_goal:
    if not received_goal:
        print(f"\nCurrent position: {Ac[0]:.3f} {Ac[1]:.3f} {Ac[2]:.3f}")
        print("Enter new desired position for the center of box (X Y Z):")
        try:
            ax_new, ay_new, az_new = map(float, input("Coordinates: ").split())
            if all(-1.309 <= coord <= 1.309 for coord in [ax_new, ay_new, az_new]):
                received_goal = True
                target_pos = np.array([ax_new, ay_new, az_new])
                # Reset previous lengths when setting a new goal
                cur_L_1, cur_L_2, cur_L_3, cur_L_4 = robot.inverse_kinematics()
                robot.prev_lengths = np.array([cur_L_1, cur_L_2, cur_L_3, cur_L_4])
                print(f"Target position set to: {target_pos}")
            else:
                print("Coordinates must be in range (-1.309, 1.309). Try again.")
        except:
            print("Invalid input. Please enter three numbers separated by spaces.")
    
    # Get current cable lengths (using robot's current position)
    cur_L_1, cur_L_2, cur_L_3, cur_L_4 = robot.inverse_kinematics()
    cur_lengths = np.array([cur_L_1, cur_L_2, cur_L_3, cur_L_4])
    
    # Get target cable lengths (using target position)
    target_L_1, target_L_2, target_L_3, target_L_4 = robot.inverse_kinematics(target_pos)
    target_lengths = np.array([target_L_1, target_L_2, target_L_3, target_L_4])
    
    # PD Control logic
    if received_goal:
        # # Calculate cable length errors
        length_errors = target_lengths - cur_lengths
        
        # # Calculate cable velocity (derivative term)
        # # Note: This is a simple finite difference approximation
        # cable_velocities = (cur_lengths - robot.prev_lengths) / dt
        
        print(f'length_errors: {length_errors}')
        
        # # PD control law: control = Kp * error + Kd * (-velocity)
        # # We use negative velocity because we want to counteract motion away from target
        # control_signals = Kp * length_errors - Kd * cable_velocities
        
        # # print(f'control_signals: {-control_signals}')
        # # Apply control signals
        # for i in range(4):
        #     data.ctrl[i] = -control_signals[i]
        
        # # Store current lengths for next derivative calculation
        # robot.prev_lengths = cur_lengths.copy()
        
        # Convert cable length changes to slider position changes
        slider_positions = 0.9 - target_lengths  # Assuming initial cable length is 4.235

        for i in range(4):
            data.ctrl[i] = slider_positions[i]
        
        print(f'forces: {data.actuator_force}')
        
        # Check if we've reached the goal
        if np.all(np.abs(length_errors) < threshold):
            print("Reached goal position!")
            received_goal = False
    
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
    
    # Capture end-effector camera view
    ee_cam = mj.MjvCamera()
    ee_cam.type = mj.mjtCamera.mjCAMERA_FIXED
    ee_cam.fixedcamid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "ee_camera")
    
    # Offscreen rendering setup
    offwidth, offheight = 640, 480
    offviewport = mj.MjrRect(0, 0, offwidth, offheight)
    con = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, con)
    
    # Allocate buffers
    rgb_buffer = np.zeros((offheight, offwidth, 3), dtype=np.uint8)
    depth_buffer = np.zeros((offheight, offwidth), dtype=np.float32)
    
    # Render camera view
    mj.mjv_updateScene(model, data, opt, None, ee_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(offviewport, scene, con)
    mj.mjr_readPixels(rgb_buffer, depth_buffer, offviewport, con)
    
    # Process image for OpenCV
    rgb_buffer = np.flipud(rgb_buffer)
    bgr_buffer = cv2.cvtColor(rgb_buffer, cv2.COLOR_RGB2BGR)
    
    # Show in separate window
    cv2.imshow("End-Effector Camera View", bgr_buffer)
    cv2.waitKey(1)
    
    # Restore default framebuffer
    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW, con)
    
    glfw.swap_buffers(window)
    glfw.poll_events()

# Cleanup
cv2.destroyAllWindows()
glfw.terminate()