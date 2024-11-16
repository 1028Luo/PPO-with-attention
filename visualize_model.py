import torch
from torchviz import make_dot
from model import Actor, Critic
import os
os.environ["PATH"] += os.pathsep + r"c:\users\jiexing\miniconda3\envs\panda_gym\lib\site-packages"

# Assuming your Actor and Critic classes are defined as before
# Create instances of Actor and Critic
input_dim = 12  # Update this to your actual input dimension
output_dim = 7  # Update this to your actual output dimension
actor = Actor(12, 7)
critic = Critic(12, 7)

# Create a random input tensor to pass through the Actor network
state_tensor = torch.randn(1, input_dim)  # Batch size of 1

# Forward pass through Actor to get action and log_prob
action, log_prob = actor(state_tensor)



print(state_tensor)
print(action)

value = critic(state_tensor, action)



# Visualize the Actor model
actor_graph = make_dot(action, params=dict(actor.named_parameters()))
actor_graph.render("actor_model", format="png")  # Save as PNG

# Visualize the Critic model
critic_graph = make_dot(value, params=dict(critic.named_parameters()))
critic_graph.render("critic_model", format="png")  # Save as PNG