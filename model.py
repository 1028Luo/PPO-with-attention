# Submission for ME5418 project Milestone 2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


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
        self.input_layer = nn.Linear(input_dim, 256)

        self.attention = MultiHeadAttention(embed_size=256, heads=4)

        self.fc1 = nn.Linear(256, 128)
        self.input_layer_residual = nn.Linear(256, 128) # residual from input to fc1

        self.fc2_mean = nn.Linear(128, output_dim)
        self.fc2_log_std = nn.Linear(128, output_dim)

        # Layer Normalization, for solving vanishing gradient
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(128)

        # Weight initialization, for solving vanishing gradient
        nn.init.kaiming_normal_(self.input_layer.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc2_mean.weight)
        nn.init.xavier_uniform_(self.fc2_log_std.weight)

    def forward(self, state):
        # First layer with normalization
        x = F.relu(self.ln1(self.input_layer(state)))

        x = x.unsqueeze(1)  
        x_attention = self.attention(x)
        x = x.squeeze(1) + x_attention.squeeze(1)

        # Second layer with residual connection and normalization
        x_fc1 = F.relu(self.ln2(self.fc1(x)))
        x_residual = self.input_layer_residual(x)
        x = x_fc1 + x_residual

        # Get output
        mean = self.fc2_mean(x)
        log_std = self.fc2_log_std(x).clamp(min=-20, max=2)  # Clamping log_std
        std = torch.exp(log_std)

        # Create normal distribution for PPO,
        # because PPO compares policies with their distribution
        distribution = dist.Normal(mean, std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1)

        return action, log_prob


class Critic(nn.Module):
    """
    Overview:
        Critic for PPO with a Self-Attention layer

    Arguments:
        - input_dim: size of observation
        - action_dim: size of action
    Return:
    """
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.input_layer = nn.Linear(input_dim, 256)

        # attention layer
        self.attention = MultiHeadAttention(embed_size=256, heads=4)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.cat([state], dim=-1)
        x = F.relu(self.input_layer(x))
        
        # 
        x = x.unsqueeze(1)  # Necessary for attention, not sure why
        x_attention = self.attention(x)
        x = x.squeeze(1) + x_attention.squeeze(1)  # Residual connection

        # Remaining layers
        x = F.relu(self.fc1(x))
        return self.fc2(x)
