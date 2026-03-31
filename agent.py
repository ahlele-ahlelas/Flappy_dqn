import argparse
from pickletools import optimize
import flappy_bird_gymnasium
import gymnasium as gym
import torch
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch.nn as nn
import torch
import torch.optim as optim
import random
import os

# Set up directory for saving runs/logs
RUN_DIR = "runs"
os.makedirs(RUN_DIR, exist_ok=True)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class Agent:
    def __init__(self, param_set):
        with open("parameters.yaml", 'r') as f:
            self.param_set = param_set
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUN_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUN_DIR, f"{self.param_set}_model.pth")

    def run(self, is_training=False, render=False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=True)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            replay_memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_reward = float('-inf')

        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            episode_reward = 0
            terminated = False
            while not terminated and episode_reward < self.reward_threshold:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze(dim=0).argmax()

                next_state, reward, terminated, _, info = env.step(action.item())

                next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)

                if is_training:
                    replay_memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                state = next_state
                episode_reward += reward.item()

            print(f"Episode {episode} finished with reward {episode_reward} & epsilon_decay {epsilon:.4f}")

            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if episode_reward > best_reward:
                    log_msg = f"New best reward: {episode_reward:.2f} at episode {episode}\n"
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_msg + "\n")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                if len(replay_memory) >= self.mini_batch_size:
                    mini_batch = replay_memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    if steps >= self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)  # fixed: was self.device

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the DQN agent on Flappy Bird")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--hyperparameters", type=str, required=True, help="Parameter set name from parameters.yaml")  # ADDED
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True, render=False)
    else:
        dql.run(is_training=False, render=True)