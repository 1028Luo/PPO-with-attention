import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3_mean = nn.Linear(128, output_dim)  # Direct output for mean
        self.fc3_log_std = nn.Linear(128, output_dim)  # Direct output for log_std

        # Layer Normalization
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(128)

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc3_mean.weight)
        nn.init.xavier_uniform_(self.fc3_log_std.weight)

    def forward(self, state):
        # First layer with normalization
        x = F.relu(self.ln1(self.fc1(state)))

        # Second layer with normalization
        x = F.relu(self.ln2(self.fc2(x)))

        # Output layers with constrained log_std
        mean = self.fc3_mean(x)
        log_std = self.fc3_log_std(x).clamp(min=-20, max=2)  # Clamping log_std
        std = torch.exp(log_std)

        # Create a normal distribution
        distribution = dist.Normal(mean, std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1)

        return action, log_prob

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Direct output layer for value

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)  # Final output layer
