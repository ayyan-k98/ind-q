"""
Neural Network Architectures for MARL Coverage System
Contains ConvDQN and DuelingConvDQN implementations.
"""

import torch
import torch.nn as nn


class ConvDQN(nn.Module):
    """Convolutional DQN for spatial awareness."""

    def __init__(self, input_channels, grid_size, num_actions, feature_dim=2):
        """
        Initialize ConvDQN network.

        Args:
            input_channels: Number of input channels (e.g., 2 for coverage + obstacles)
            grid_size: Size of the grid (height = width)
            num_actions: Number of possible actions
            feature_dim: Dimension of additional features (e.g., orientation)
        """
        super(ConvDQN, self).__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim

        # Convolutional layers for spatial processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Calculate conv output size dynamically
        conv_h = conv_w = grid_size
        final_conv_output_size = conv_h * conv_w * 64  # Based on last layer's channels

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(final_conv_output_size + feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x, features):
        """
        Forward pass through the network.

        Args:
            x: Grid input tensor [batch, channels, height, width]
            features: Additional features [batch, feature_dim]

        Returns:
            Q-values for each action [batch, num_actions]
        """
        # Ensure correct dimensions upon entry
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add channel dim
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dim
        if len(features.shape) == 1:
            features = features.unsqueeze(0)  # Add batch dim

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten conv output: [batch, flattened_size]
        combined = torch.cat((x, features), dim=1)  # Concatenate along feature dimension
        return self.fc_layers(combined)


class DuelingConvDQN(nn.Module):
    """Dueling architecture for better value estimation."""

    def __init__(self, input_channels, grid_size, num_actions, feature_dim=2):
        """
        Initialize DuelingConvDQN network.

        Args:
            input_channels: Number of input channels
            grid_size: Size of the grid
            num_actions: Number of possible actions
            feature_dim: Dimension of additional features
        """
        super(DuelingConvDQN, self).__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim

        # Convolutional layers (shared) - Match ConvDQN structure
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Calculate conv output size
        conv_h = conv_w = grid_size
        final_conv_output_size = conv_h * conv_w * 64

        # Combined feature size
        combined_feature_size = final_conv_output_size + feature_dim

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(combined_feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Outputs scalar value V(s)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(combined_feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)  # Outputs advantage A(s,a) for each action
        )

    def forward(self, x, features):
        """
        Forward pass through the dueling network.

        Args:
            x: Grid input tensor [batch, channels, height, width]
            features: Additional features [batch, feature_dim]

        Returns:
            Q-values for each action [batch, num_actions]
        """
        # Ensure correct dimensions upon entry
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if len(features.shape) == 1:
            features = features.unsqueeze(0)

        batch_size = x.size(0)
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)  # Flatten conv output

        combined = torch.cat((x, features), dim=1)

        # Calculate Value and Advantage
        value = self.value_stream(combined)  # Shape: [batch, 1]
        advantage = self.advantage_stream(combined)  # Shape: [batch, num_actions]

        # Combine using Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
