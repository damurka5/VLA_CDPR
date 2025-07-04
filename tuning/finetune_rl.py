import mujoco
import imageio
from PIL import Image
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import numpy as np
from cdpr_config import CDPRFinetuneConfig

class CDPRSimEnv:
    def __init__(self, model_path, config):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.config = config
        self.viewer = None
        self.current_step = 0
        self.max_steps = config.max_steps_per_episode
        
        # Load OpenVLA model
        self.tokenizer = AutoTokenizer.from_pretrained(config.vla_path)
        self.vla_model = AutoModelForCausalLM.from_pretrained(
            config.vla_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load prompts
        with open(config.prompt_file) as f:
            self.prompts = json.load(f)
    
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        prompt = random.choice(self.prompts)
        return self._get_obs(), prompt
    
    def _get_obs(self):
        # Get current frame
        if self.viewer is None and self.config.render:
            self.viewer = mujoco.MjViewer(self.model)
        
        if self.config.render:
            self.viewer.render()
            img = self.viewer.read_pixels()
            img = Image.fromarray(img)
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)  # Dummy image
            
        # Get end effector position
        ee_pos = self.data.geom_xpos[self.model.geom("end_effector").id]
        
        return {
            "image": img,
            "ee_pos": ee_pos,
            "cable_lengths": self.data.actuator_length.copy()
        }
    
    def step(self, action):
        # Apply action (cable length changes)
        for i, delta in enumerate(action):
            self.data.actuator_length[i] += delta
            
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward (example: distance to target)
        target_pos = np.array([0.5, 0.5, 0.5])  # Example target
        distance = np.linalg.norm(obs["ee_pos"] - target_pos)
        reward = -distance  # Negative distance as reward
        
        # Check termination
        done = self.current_step >= self.max_steps or distance < 0.05
        
        return obs, reward, done, {}

def train_rl_openvla(config):
    # Initialize environment
    env = CDPRSimEnv(config.mujoco_model_path, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize RL algorithm
    if config.rl_algorithm == "PPO":
        from stable_baselines3 import PPO
        model = PPO("MultiInputPolicy", env, verbose=1, 
                    learning_rate=config.lr_actor,
                    batch_size=config.batch_size,
                    gamma=config.gamma)
    elif config.rl_algorithm == "SAC":
        from stable_baselines3 import SAC
        model = SAC("MultiInputPolicy", env, verbose=1,
                    learning_rate=config.lr_actor,
                    batch_size=config.batch_size,
                    gamma=config.gamma,
                    tau=config.tau)
    elif config.rl_algorithm == "DDPG":
        from stable_baselines3 import DDPG
        model = DDPG("MultiInputPolicy", env, verbose=1,
                     learning_rate=config.lr_actor,
                     batch_size=config.batch_size,
                     gamma=config.gamma,
                     tau=config.tau)
    elif config.rl_algorithm == "TD3":
        from stable_baselines3 import TD3
        model = TD3("MultiInputPolicy", env, verbose=1,
                    learning_rate=config.lr_actor,
                    batch_size=config.batch_size,
                    gamma=config.gamma,
                    tau=config.tau,
                    policy_delay=config.policy_freq)
    else:
        raise ValueError(f"Unknown RL algorithm: {config.rl_algorithm}")
    
    # Training loop
    for episode in range(config.num_episodes):
        obs, prompt = env.reset()
        episode_reward = 0
        frames = []
        
        # Process prompt with OpenVLA
        inputs = env.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            vla_output = env.vla_model.generate(**inputs, max_new_tokens=50)
        vla_instruction = env.tokenizer.decode(vla_output[0], skip_special_tokens=True)
        
        for step in range(config.max_steps_per_episode):
            # Get action from RL policy
            action, _ = model.predict(obs, deterministic=False)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Store transition
            model.replay_buffer.add(obs, next_obs, action, reward, done)
            
            # Update policy
            if len(model.replay_buffer) > config.batch_size:
                model.train(batch_size=config.batch_size)
            
            obs = next_obs
            if done:
                break
        
        # Evaluation
        if episode % config.eval_freq == 0:
            eval_rewards = []
            for _ in range(config.num_eval_episodes):
                obs, _ = env.reset()
                total_reward = 0
                for _ in range(config.max_steps_per_episode):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
                eval_rewards.append(total_reward)
            avg_reward = np.mean(eval_rewards)
            print(f"Episode {episode}, Eval Reward: {avg_reward:.2f}")
        
        # Save model
        if episode % config.save_freq == 0:
            model.save(f"models/{config.rl_algorithm}_cdpr_{episode}")

if __name__ == "__main__":
    config = CDPRFinetuneConfig()
    train_rl_openvla(config)