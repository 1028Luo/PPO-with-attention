# Submission for ME5418 project Milestone 2

import panda_gym
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from model import Actor, Critic
import time
import matplotlib.pyplot as plt

# Hyperparameters
env_name = 'PandaReachJointsDense-v3'
num_episodes = 3000
learning_rate = 1e-5 # better than 1e-4
gamma = 0.99
epsilon = 0.2  # Clip parameter for PPO
update_steps = 5  # Number of updates per episode
TIME_LIMIT = 5 # time limit for an episode


# Initialize the environment
env = gym.make(env_name, render_mode='human')

# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get dim of state and action
state_dim = 0
for key, value in env.observation_space.items():
    state_dim = state_dim + value.shape[0]
print("total num of state of this env is: ", state_dim)
action_dim = env.action_space.shape[0]
print("total num of action of this env is: ", action_dim)

# Instantiate actor and critic and move them to the GPU
actor = Actor(input_dim=state_dim, output_dim=action_dim).to(device)
critic = Critic(input_dim=state_dim).to(device)

# Optimizers
optimizer_actor = optim.Adam(actor.parameters(), lr=learning_rate)
optimizer_critic = optim.Adam(critic.parameters(), lr=learning_rate)

episode_rewards_avg_100 = []
reward_log_buffer = []
# PPO Training Loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    log_probs = []
    values = []
    rewards = []
    states = []
    actions = []
    
    start_time = time.time()
    while not done:
        
        state = np.concatenate(list(state.values()))
        
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)

        # Get action from the actor, here action is tensor
        action, log_prob = actor(state_tensor)

        # convert action back to numpy array to add noise
        action = action.detach().cpu().numpy() + np.random.normal(0, 0.1, size=action_dim)  # Adding noise
        action = np.clip(action, env.action_space.low, env.action_space.high)

        action = action.squeeze(0) # not sure why
        # state_tensor = state_tensor.squeeze(0)

        # print(state_tensor)
        # print(action)

        next_state, reward, terminated, truncated, _ = env.step(action)

        # Store the data
        states.append(state_tensor)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        
        # get action tensor
        action_tensor = torch.tensor(action, dtype=torch.float32).to(device).unsqueeze(0)
        # print(state_tensor)
        # print(action_tensor)

        values.append(critic(state_tensor).detach())

        state = next_state

        if terminated or truncated:
            done = True

        # Check if the step took too long
        elapsed_time = time.time() - start_time
        if elapsed_time > TIME_LIMIT:
            print(f"Step timed out after {elapsed_time:.2f} seconds.")
            done = True  # End the episode if timeout occurs

    
    print('One episode sampled')
    # Compute advantages and returns
    returns = []
    discounted_return = 0
    for r in rewards[::-1]:
        discounted_return = r + gamma * discounted_return
        returns.insert(0, discounted_return)

    returns = torch.tensor(returns, dtype=torch.float32).to(device).unsqueeze(1)
    states = torch.cat(states)
    actions = torch.tensor(actions, dtype=torch.float32).to(device)
    log_probs = torch.cat(log_probs)
    values = torch.cat(values)

    # PPO Update
    for _ in range(update_steps):
        # Calculate the advantage
        advantage = returns - values

        # Update Actor
        optimizer_actor.zero_grad()
        new_log_probs = actor(states)[1]
        ratio = (new_log_probs - log_probs.detach()).exp()
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        actor_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
        actor_loss.backward()
        optimizer_actor.step()


        # Update Critic
        optimizer_critic.zero_grad()
        critic_loss = (returns - critic(states)).pow(2).mean()
        critic_loss.backward()
        optimizer_critic.step()


        # Log gradients
        print("Actor Gradients:")
        for name, param in actor.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.norm().item()}")  # Log the gradient norm

        print("Critic Gradients:")
        for name, param in critic.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.norm().item()}")  # Log the gradient norm
               



    reward_log_buffer.append(sum(rewards))
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {sum(rewards)}")

    # for logging
    if (episode % 100 == 0) and (episode != 0):
        episode_rewards_avg_100.append(sum(reward_log_buffer)/100)
        reward_log_buffer = []


env.close()


plt.figure(figsize=(10, 5))
plt.plot(episode_rewards_avg_100)
plt.xlabel("Episode")
plt.ylabel("Average reward every 100 episodes")
plt.title("Rewards per Episode")
plt.show()