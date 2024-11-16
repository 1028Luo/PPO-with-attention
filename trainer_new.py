import panda_gym
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from model_new import Actor, Critic
import time
import matplotlib.pyplot as plt

# Hyperparameters
env_name = 'PandaReachJointsDense-v3'
num_episodes = 1000
learning_rate = 1e-5 # better than 1e-4
gamma = 0.99
epsilon = 0.2  # Clip parameter for PPO
update_steps = 5  # Number of updates per episode
TIME_LIMIT = 5 # time limit for an episode
CLIP_COEF = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01

# Helper functions for PPO training
def get_deltas(rewards, values, next_values, next_nonterminal, gamma):
    """Compute the temporal difference (TD) error.

    Args:
        rewards (torch.Tensor): Rewards at each time step, shape: (batch_size,).
        values (torch.Tensor): Predicted values for each state, shape: (batch_size,).
        next_values (torch.Tensor): Predicted value for the next state, shape: (batch_size,).
        gamma (float): Discount factor.

    Returns:
        torch.Tensor: Computed TD errors, shape: (batch_size,).
    """

    deltas = rewards + gamma * next_values * next_nonterminal - values
    # next_nonterminal: indicate if the next state is non-terminal
    # dummy_next_nonterminal = torch.tensor([1,0,1]) means env 1 and env 3 are non-terminal

    return deltas

def get_ratio(logprob, logprob_old):
    """Compute the probability ratio between the new and old policies.

    This function calculates the ratio of the probabilities of actions under
    the current policy compared to the old policy, using their logarithmic values.

    Args:
        logprob (torch.Tensor): Log probability of the action under the current policy,
                                shape: (batch_size,).
        logprob_old (torch.Tensor): Log probability of the action under the old policy,
                                    shape: (batch_size,).

    Returns:
        torch.Tensor: The probability ratio of the new policy to the old policy,
                      shape: (batch_size,).
    """

    # Just the property of log
    logratio = logprob - logprob_old  # Compute the log ratio
    ratio = torch.exp(logratio)  # Exponentiate to get the probability ratio

    return ratio


def get_policy_objective(advantages, ratio, clip_coeff=CLIP_COEF):
    """Compute the clipped surrogate policy objective.

    This function calculates the policy objective using the advantages and the
    probability ratio, applying clipping to stabilize training.

    Args:
        advantages (torch.Tensor): The advantage estimates, shape: (batch_size,).
        ratio (torch.Tensor): The probability ratio of the new policy to the old policy,
                             shape: (batch_size,).
        clip_coeff (float, optional): The clipping coefficient for the policy objective.
                                       Defaults to CLIP_COEF.

    Returns:
        torch.Tensor: The computed policy objective, a scalar value.
    """
    ### ------------ TASK 3.1.2 ---------- ###
    ### ----- YOUR CODES START HERE ------ ###

    policy_objective1 = ratio * advantages  # Calculate the first policy loss term
    policy_objective2 = (torch.clamp(ratio, 1-clip_coeff, 1+clip_coeff)) * advantages  # Calculate the clipped policy loss term
    policy_objective = torch.min(policy_objective1, policy_objective2).mean()  # Take the minimum and average over the batch

    ### ------ YOUR CODES END HERE ------- ###
    return policy_objective

def get_value_loss(values, values_old, returns):
    """Compute the combined value loss with clipping.

    This function calculates the unclipped and clipped value losses
    and returns the maximum of the two to stabilize training.

    Args:
        values (torch.Tensor): Predicted values from the critic, shape: (batch_size, 1).
        values_old (torch.Tensor): Old predicted values from the critic, shape: (batch_size, 1).
        returns (torch.Tensor): Computed returns for the corresponding states, shape: (batch_size, 1).

    Returns:
        torch.Tensor: The combined value loss, a scalar value.
    """

    value_loss_unclipped = (0.5 * (values - returns) ** 2).mean()   # Calculate unclipped value loss

    value_loss_clipped = (0.5 * (values_old + torch.clamp(values - values_old, -CLIP_COEF, CLIP_COEF)- returns) ** 2).mean()  # Calculate clipped value loss

    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

    return value_loss  # Return the final combined value loss

def get_entropy_objective(entropy):
    """Compute the entropy objective.

    This function calculates the average entropy of the action distribution,
    which encourages exploration by penalizing certainty.

    Args:
        entropy (torch.Tensor): Entropy values for the action distribution, shape: (batch_size,).

    Returns:
        torch.Tensor: The computed entropy objective, a scalar value.
    """
    return entropy.mean()  # Return the average entropy

def get_total_loss(policy_objective, value_loss, entropy_objective, value_loss_coeff=VALUE_LOSS_COEF, entropy_coeff=ENTROPY_COEF):
    """Compute the total loss for the actor-critic agent.

    This function combines the policy objective, value loss, and entropy objective
    into a single loss value for optimization. It applies coefficients to scale
    the contribution of the value loss and entropy objective.

    Args:
        policy_objective (torch.Tensor): The policy objective, a scalar value.
        value_loss (torch.Tensor): The computed value loss, a scalar value.
        entropy_objective (torch.Tensor): The computed entropy objective, a scalar value.
        value_loss_coeff (float, optional): Coefficient for scaling the value loss. Defaults to VALUE_LOSS_COEF.
        entropy_coeff (float, optional): Coefficient for scaling the entropy loss. Defaults to ENTROPY_COEF.

    Returns:
        torch.Tensor: The total computed loss, a scalar value.
    """

    total_loss = -policy_objective + value_loss_coeff * value_loss - entropy_coeff * entropy_objective  # Combine losses

    return total_loss













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

        probs = actor.get_probs(state_tensor)
        action = actor.get_action(probs)
        log_prob = actor.get_action_logprob(probs, action)

        print('probs is: ', probs)


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

        # new_log_probs = actor(states)[1]

        probs = actor.get_probs(states)
        action = actor.get_action(probs)
        new_log_probs = actor.get_action_logprob(probs, action)


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


        # # Log gradients
        # print("Actor Gradients:")
        # for name, param in actor.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: {param.grad.norm().item()}")  # Log the gradient norm

        # print("Critic Gradients:")
        # for name, param in critic.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: {param.grad.norm().item()}")  # Log the gradient norm
               



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