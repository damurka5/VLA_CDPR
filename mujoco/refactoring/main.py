import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import *
from CDPRDomain import CDPRDomain

def main():
    # Initialize CDPR environment
    env = CDPRDomain(
        bddl_file_name="cdpr_problem.bddl",
        use_camera_obs=True,
        camera_names=["ee_camera", "overview"],
        camera_heights=480,
        camera_widths=640,
        control_freq=60,
    )
    
    # Reset environment
    obs = env.reset()
    
    # Main control loop
    try:
        while True:
            # Get current state
            ee_pos = env.sim.data.body_xpos[env.ee_body]
            print(f"Current position: {ee_pos[0]:.3f} {ee_pos[1]:.3f} {ee_pos[2]:.3f}")
            
            # User input for target position
            print("Enter target position (X Y Z) or 'q' to quit:")
            user_input = input().strip()
            
            if user_input.lower() == 'q':
                break
                
            try:
                target_coords = list(map(float, user_input.split()))
                if len(target_coords) == 3:
                    success = env.set_target_position(target_coords)
                    if not success:
                        print("Coordinates must be in range (-1.309, 1.309)")
                        continue
                    
                    # Run until target reached
                    while not env._check_success():
                        env.step(np.zeros(env.action_dim))  # Empty action, CDPR controls itself
                        env.render()
                        
                    print("Target reached!")
                    
            except ValueError:
                print("Invalid input. Please enter three numbers separated by spaces.")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    
    env.close()

if __name__ == "__main__":
    main()