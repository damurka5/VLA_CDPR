import torch
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import numpy as np

class PPOTrainer:
    def __init__(self, config, policy, device):
        self.config = config
        self.policy = policy
        self.device = device
        self.optimizer = optim.Adam(policy.parameters(), lr=config.lr_actor)
        self.memory = deque(maxlen=config.buffer_size)
        self.clip_param = 0.2
        self.ppo_epoch = 4
        self.mini_batch_size = 64
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def update(self):
        if len(self.memory) < self.config.batch_size:
            return
        
        # Sample from memory
        states, actions, rewards, next_states, dones = zip(*random.sample(self.memory, self.config.batch_size))
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        # Calculate advantages
        with torch.no_grad():
            _, next_value = self.policy(next_states)
            target_value = rewards + (1 - dones) * self.config.gamma * next_value
            _, value = self.policy(states)
            advantages = target_value - value
        
        # PPO updates
        for _ in range(self.ppo_epoch):
            for index in range(0, self.config.batch_size, self.mini_batch_size):
                mini_states = states[index:index+self.mini_batch_size]
                mini_actions = actions[index:index+self.mini_batch_size]
                mini_advantages = advantages[index:index+self.mini_batch_size]
                mini_old_log_probs = self.policy.get_log_prob(mini_states, mini_actions).detach()
                
                # Calculate new policy's log prob and value
                dist, value = self.policy(mini_states)
                new_log_probs = dist.log_prob(mini_actions).sum(-1, keepdim=True)
                entropy = dist.entropy().mean()
                
                # Policy ratio
                ratio = (new_log_probs - mini_old_log_probs).exp()
                surr1 = ratio * mini_advantages
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * mini_advantages
                
                # Losses
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (value - target_value[index:index+self.mini_batch_size]).pow(2).mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                
class SACTrainer:
    def __init__(self, config, policy, device):
        self.config = config
        self.policy = policy
        self.device = device
        self.memory = deque(maxlen=config.buffer_size)
        self.total_it = 0
        
        # Networks
        self.critic1 = policy.critic1
        self.critic2 = policy.critic2
        self.critic_target1 = policy.critic_target1
        self.critic_target2 = policy.critic_target2
        self.actor = policy.actor
        self.log_alpha = policy.log_alpha
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.lr_critic)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr_actor)
        
        # Target entropy
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def update(self):
        if len(self.memory) < self.config.batch_size:
            return
            
        # Sample batch
        states, actions, rewards, next_states, dones = zip(*random.sample(self.memory, self.config.batch_size))
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            # Target actions
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Target Q-values
            q1_next = self.critic_target1(next_states, next_actions)
            q2_next = self.critic_target2(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.config.gamma * q_next
        
        # Critic losses
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic1_loss = 0.5 * F.mse_loss(current_q1, target_q)
        critic2_loss = 0.5 * F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Actor loss
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Alpha loss
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        # Update target networks
        for param, target_param in zip(self.critic1.parameters(), self.critic_target1.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.critic_target2.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)