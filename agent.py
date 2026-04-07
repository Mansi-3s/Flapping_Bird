import flappy_bird_gymnasium
import gymnasium
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random

# -------------------------
# Device
# -------------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

RUNS_DIR = "run"
os.makedirs(RUNS_DIR, exist_ok=True)

# -------------------------
# Simple DQN (inline)
# -------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Simple Replay Memory
# -------------------------
class ReplayMEMORY:
    def __init__(self, size):
        self.buffer = []
        self.size = size

    def append(self, s, a, ns, r, d):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append((s, a, ns, r, d))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# -------------------------
# Agent
# -------------------------
class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def run(self, is_training=True, render=False):

        env = gymnasium.make(
            "FlappyBird-v0",
            render_mode="human" if render else None
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMEMORY(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_reward = float("-inf")

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
                        action = policy_dqn(state.unsqueeze(0)).argmax()

                next_state, reward, terminated, _, _ = env.step(action.item())

                episode_reward += reward

                reward = torch.tensor(reward, dtype=torch.float32, device=device)
                new_state = torch.tensor(next_state, dtype=torch.float32, device=device)

                if is_training:
                    # ✅ FIX: store CPU numpy instead of tensor (important)
                    memory.append(
                        state.cpu().numpy(),
                        action.item(),
                        next_state,
                        reward.item(),
                        terminated
                    )
                    steps += 1

                state = new_state

            print(f"episode={episode+1} total reward={episode_reward} epsilon={epsilon if is_training else 0}")

            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if episode_reward > best_reward:
                    log_msg = f"best reward = {episode_reward} for episode={episode+1}"

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

            if is_training and len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)

                self.optimize(mini_batch, policy_dqn, target_dqn)

                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        for state, action, next_state, reward, terminated in mini_batch:

            # ✅ convert back to tensors
            state = torch.tensor(state, dtype=torch.float32, device=device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            reward = torch.tensor(reward, dtype=torch.float32, device=device)
            action = torch.tensor(action, dtype=torch.long, device=device)

            if terminated:
                target_q = reward
            else:
                with torch.no_grad():
                    target_q = reward + self.gamma * target_dqn(next_state).max()

            current_q = policy_dqn(state)[action]

            loss = self.loss_fn(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters", help="Parameter set name")
    parser.add_argument("--train", help="Training mode", action="store_true")

    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)