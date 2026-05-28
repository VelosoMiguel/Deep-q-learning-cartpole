import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# --- Hyperparameters ---
BUFFER_SIZE = 5000           # Replay buffer size
BATCH_SIZE = 32              # Mini-batch size for training
GAMMA = 0.99                 # Discount factor for future rewards
EPS_START = 1.0              # Initial epsilon for epsilon-greedy
EPS_END = 0.01               # Minimum epsilon
EPS_DECAY = 0.995            # Epsilon decay factor per episode
LEARNING_RATE = 1e-3         # Learning rate for the optimizer
TARGET_UPDATE = 200           # Update target network every N steps
NUM_EPISODES = 200            # Maximum number of training episodes
TRAIN_FREQUENCY = 1           # Train every N steps

# --- Replay Buffer ---
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# --- Deep Q-Network ---
class DQN(nn.Module):
    """Neural network for approximating Q-values."""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values for each action

# --- Epsilon-greedy action selection ---
def select_action(state, policy_net, eps, n_actions):
    """Select an action using epsilon-greedy policy."""
    if random.random() > eps:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = q_values.max(1)[1].item()  # Choose action with highest Q-value
    else:
        action = random.randrange(n_actions)  # Random action (exploration)
    return action

# --- Optimize the model ---
def optimize_model(policy_net, target_net, optimizer, buffer, device):
    """Sample a batch from replay buffer and update the network."""
    if len(buffer) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.BoolTensor(dones).to(device)

    # Compute current Q-values
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
    # Compute next Q-values from target network
    next_q_values = target_net(next_states).max(1)[0].detach()
    # Compute the target Q-values
    targets = rewards + GAMMA * next_q_values * (1 - dones.float())

    # Compute Huber loss
    loss = F.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# --- Training and visualization ---
def train_and_play():
    """Train the agent using DQN and visualize training progress."""
    env = gym.make('CartPole-v1', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(BUFFER_SIZE)

    scores = []
    eps = EPS_START
    steps_done = 0

    # Set up live plot
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlim(0, NUM_EPISODES)
    ax.set_ylim(0, 210)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average Score (last 100)")
    ax.set_title("DQN CartPole Training (Visual)")

    for episode in range(1, NUM_EPISODES+1):
        state, _ = env.reset()
        score = 0
        done = False

        while not done:
            # Select and perform an action
            action = select_action(state, policy_net, eps, action_size)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store the transition in replay buffer
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            score += reward
            steps_done += 1

            # Train the policy network
            if steps_done % TRAIN_FREQUENCY == 0:
                optimize_model(policy_net, target_net, optimizer, buffer, device)

            # Update target network periodically
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        eps = max(EPS_END, eps * EPS_DECAY)
        scores.append(score)

        # Compute average score over last 100 episodes
        avg_score = np.mean(scores[-100:])

        # Update live plot
        line.set_data(range(len(scores)), [np.mean(scores[max(0,i-100):i+1]) for i in range(len(scores))])
        ax.set_xlim(0, max(100, len(scores)))
        ax.set_ylim(0, max(200, max([np.mean(scores[max(0,i-100):i+1]) for i in range(len(scores))])+10))
        ax.set_title(f"DQN CartPole Training\nEpisode {episode}, Score: {score:.1f}, Avg 100: {avg_score:.2f}, Epsilon: {eps:.3f}")
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Print progress to console
        print(f"Episode {episode}, Score: {score:.1f}, Avg: {avg_score:.2f}, Epsilon: {eps:.3f}")

        # Stop if solved
        if avg_score >= 195:
            print(f"Solved in {episode} episodes!")
            break

    env.close()
    plt.ioff()
    plt.show()
    return policy_net

if __name__ == "__main__":
    trained_policy = train_and_play()

