# Submission for ME5418 project Milestone 2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize the weights and biases of a layer.

    Args:
        layer (nn.Module): The layer to initialize.
        std (float): Standard deviation for orthogonal initialization.
        bias_const (float): Constant value for bias initialization.

    Returns:
        nn.Module: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)  # Orthogonal initialization
    torch.nn.init.constant_(layer.bias, bias_const)  # Constant bias
    return layer

class UnsqueezeLayer(nn.Module):
    def __init__(self, dim=1):
        super(UnsqueezeLayer, self).__init__()
        self.dim = dim  # Dimension to unsqueeze on, default is 1 (unsqueeze to add a new dimension at index 1)

    def forward(self, x):
        return x.unsqueeze(self.dim)  # Add a new dimension to the input tensor

class MultiHeadAttention(nn.Module):
    """
    Overview:
        Multi-head attention mechanism

    Arguments:
        - embed_size: size of the embedding space, will be set by actor or critic
        - heads: number of heads
    Return:
    """
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads

        # Defines three component for Attention, K, Q, V
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)

        # output
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        # Split embedding into heads
        values = values.view(values.shape[0], values.shape[1], self.heads, -1)
        keys = keys.view(keys.shape[0], keys.shape[1], self.heads, -1)
        queries = queries.view(queries.shape[0], queries.shape[1], self.heads, -1)

        # Computes dot-product for the weights of the input
        energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("bhqk,bkhd->bqhd", [attention, values]).reshape(
            values.shape[0], values.shape[1], -1
        )

        return self.fc_out(out)


class Actor(nn.Module):
    """
    Overview:
        Actor for PPO with a Self-Attention layer

    Arguments:
        - input_dim: size of observation
        - output_dim: size of action
    Return:
    """
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()

        # Define the actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, 256)),
            nn.ReLU(),
            UnsqueezeLayer(dim=1),  # Add UnsqueezeLayer before attention
            MultiHeadAttention(embed_size=256, heads=4),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, output_dim), std=0.01),  # Final layer with small std for output
        )

    def get_probs(self, x):
        """Calculate the action probabilities for a given state.

        Args:
            x (torch.Tensor): Input state, shape: (batch_size, observation_size)

        Returns:
            torch.distributions.Categorical: Categorical distribution over actions.
        """

        logits = self.actor(x)  # Get logits from the actor network
        probs = Categorical(logits=logits)  # Create a categorical distribution from the logits

        return probs

    def get_action(self, probs):
        """Sample an action from the action probabilities.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.

        Returns:
            torch.Tensor: Sampled action, shape: (batch_size, 1)
        """

        action = probs.sample()  # Sample an action based on the probabilities

        return action

    def get_action_logprob(self, probs, action):
        """Compute the log probability of a given action.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.
            action (torch.Tensor): Selected action, shape: (batch_size, 1)

        Returns:
            torch.Tensor: Log probability of the action, shape: (batch_size, 1)
        """

        logprob = probs.log_prob(action)  # Calculate log probability of the sampled action

        return logprob

    def get_entropy(self, probs):
        """Calculate the entropy of the action distribution.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.

        Returns:
            torch.Tensor: Entropy of the distribution, shape: (batch_size, 1)
        """
        return probs.entropy()  # Return the entropy of the probabilities

    def get_action_logprob_entropy(self, x):
        """Get action, log probability, and entropy for a given state.

        Args:
            x (torch.Tensor): Input state.

        Returns:
            tuple: (action, logprob, entropy)
                - action (torch.Tensor): Sampled action.
                - logprob (torch.Tensor): Log probability of the action.
                - entropy (torch.Tensor): Entropy of the action distribution.
        """
        probs = self.get_probs(x)  # Get the action probabilities
        action = self.get_action(probs)  # Sample an action
        logprob = self.get_action_logprob(probs, action)  # Compute log probability of the action
        entropy = self.get_entropy(probs)  # Compute entropy of the action distribution
        return action, logprob, entropy  # Return action, log probability, and entropy






class Critic(nn.Module):
    """
    Overview:
        Critic for PPO with a Self-Attention layer

    Arguments:
        - input_dim: size of observation
    Return:
    """
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, 256)
        
        # Attention layer
        self.attention = MultiHeadAttention(embed_size=256, heads=4)
        
        # Fully connected layers in nn.Sequential
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        # Input layer with activation
        x = F.relu(self.input_layer(state))
        
        # Attention layer with residual connection
        x = x.unsqueeze(1)  # Add sequence dimension for attention
        x_attention = self.attention(x)
        x = x.squeeze(1) + x_attention.squeeze(1)  # Residual connection
        
        # Fully connected layers
        return self.fc_layers(x)