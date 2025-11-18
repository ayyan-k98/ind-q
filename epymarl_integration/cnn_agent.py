"""
CNN Agent for Coverage Environment - Scale Invariant Architecture

This agent uses convolutional neural networks to process spatial observations
and adaptive pooling to achieve scale invariance across different grid sizes.

Compatible with EPyMARL QMIX framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoverageCNN(nn.Module):
    """
    Convolutional feature extractor for coverage environment.

    Processes 4-channel spatial input:
    - Channel 0: Coverage grid [0, 1]
    - Channel 1: Obstacle belief (normalized) [0, 0.5, 1]
    - Channel 2: All agent positions [0, 1]
    - Channel 3: Own position [0, 1]

    Uses adaptive pooling to handle variable grid sizes (scale invariance).
    """

    def __init__(self, input_channels=4, hidden_channels=[32, 64, 64], pooled_size=(4, 4)):
        """
        Initialize CNN feature extractor.

        Args:
            input_channels: Number of input channels (default: 4)
            hidden_channels: List of channel sizes for conv layers
            pooled_size: Fixed output size after adaptive pooling (e.g., (4, 4))
        """
        super(CoverageCNN, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.pooled_size = pooled_size

        # Build convolutional layers
        layers = []
        in_c = input_channels

        for out_c in hidden_channels:
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_c = out_c

        self.conv_layers = nn.Sequential(*layers)

        # Adaptive pooling (handles variable input sizes!)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(pooled_size)

        # Calculate output feature size
        self.feature_size = hidden_channels[-1] * pooled_size[0] * pooled_size[1]

    def forward(self, spatial_input):
        """
        Forward pass through CNN.

        Args:
            spatial_input: (batch, 4, H, W) spatial grids

        Returns:
            features: (batch, feature_size) flattened spatial features
        """
        # Convolutional layers
        x = self.conv_layers(spatial_input)  # (B, C, H, W)

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # (B, C, pooled_h, pooled_w)

        # Flatten
        x = x.view(x.size(0), -1)  # (B, feature_size)

        return x


class CNNAgent(nn.Module):
    """
    Complete CNN agent for coverage environment.

    Combines spatial features from CNN with scalar features,
    then produces Q-values for each action.

    Scale Invariant: Works across different grid sizes due to:
    - Coordinate normalization in observations
    - Adaptive pooling in CNN
    - Grid-size agnostic scalar features
    """

    def __init__(
        self,
        input_channels=4,
        scalar_size=56,
        hidden_dim=256,
        n_actions=9,
        cnn_channels=[32, 64, 64],
        pooled_size=(4, 4),
    ):
        """
        Initialize CNN agent.

        Args:
            input_channels: Number of spatial channels (default: 4)
            scalar_size: Number of scalar features (default: 56)
            hidden_dim: Hidden layer size for MLP
            n_actions: Number of actions
            cnn_channels: Channel sizes for CNN layers
            pooled_size: Pooled spatial size (e.g., (4, 4))
        """
        super(CNNAgent, self).__init__()

        self.input_channels = input_channels
        self.scalar_size = scalar_size
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        # CNN feature extractor
        self.cnn = CoverageCNN(
            input_channels=input_channels,
            hidden_channels=cnn_channels,
            pooled_size=pooled_size,
        )

        # Combined feature size: spatial + scalar
        combined_size = self.cnn.feature_size + scalar_size

        # MLP layers
        self.fc1 = nn.Linear(combined_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, spatial_obs, scalar_obs):
        """
        Forward pass through agent network.

        Args:
            spatial_obs: (batch, 4, H, W) spatial grids
            scalar_obs: (batch, 56) scalar features

        Returns:
            q_values: (batch, n_actions) Q-values for each action
        """
        # Extract spatial features
        spatial_features = self.cnn(spatial_obs)  # (batch, cnn_feature_size)

        # Concatenate with scalar features
        combined = torch.cat([spatial_features, scalar_obs], dim=1)

        # MLP
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values

    def get_q_values(self, obs_dict, action_mask=None):
        """
        Get Q-values from observation dict.

        Args:
            obs_dict: Dict with 'spatial' and 'scalars' keys
            action_mask: (batch, n_actions) mask of valid actions (1=valid, 0=invalid)

        Returns:
            q_values: (batch, n_actions) Q-values (masked if action_mask provided)
        """
        spatial = obs_dict['spatial']
        scalars = obs_dict['scalars']

        q_values = self.forward(spatial, scalars)

        # Mask invalid actions with very negative values
        if action_mask is not None:
            q_values = q_values.masked_fill(action_mask == 0, -1e10)

        return q_values


class RNNAgent(nn.Module):
    """
    RNN-based agent wrapper (for compatibility with EPyMARL).

    Wraps CNNAgent with GRU for temporal processing.
    Optional: Can be used if temporal dependencies are important.
    """

    def __init__(
        self,
        input_channels=4,
        scalar_size=56,
        hidden_dim=256,
        n_actions=9,
        cnn_channels=[32, 64, 64],
        pooled_size=(4, 4),
        use_rnn=True,
    ):
        """
        Initialize RNN agent.

        Args:
            input_channels: Number of spatial channels
            scalar_size: Number of scalar features
            hidden_dim: Hidden dimension
            n_actions: Number of actions
            cnn_channels: CNN channel sizes
            pooled_size: Pooled size
            use_rnn: Whether to use RNN (default: True)
        """
        super(RNNAgent, self).__init__()

        self.use_rnn = use_rnn
        self.hidden_dim = hidden_dim

        # CNN feature extractor
        self.cnn = CoverageCNN(
            input_channels=input_channels,
            hidden_channels=cnn_channels,
            pooled_size=pooled_size,
        )

        # Combined feature size
        combined_size = self.cnn.feature_size + scalar_size

        # Pre-RNN FC
        self.fc1 = nn.Linear(combined_size, hidden_dim)

        # RNN layer (optional)
        if use_rnn:
            self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.rnn = None

        # Output layer
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self):
        """Initialize hidden state for RNN."""
        return torch.zeros(1, self.hidden_dim)

    def forward(self, spatial_obs, scalar_obs, hidden_state=None):
        """
        Forward pass with optional RNN.

        Args:
            spatial_obs: (batch, 4, H, W)
            scalar_obs: (batch, 56)
            hidden_state: (batch, hidden_dim) for RNN (if use_rnn=True)

        Returns:
            q_values: (batch, n_actions)
            hidden_state: (batch, hidden_dim) new hidden state
        """
        batch_size = spatial_obs.size(0)

        # Extract spatial features
        spatial_features = self.cnn(spatial_obs)

        # Combine with scalars
        combined = torch.cat([spatial_features, scalar_obs], dim=1)

        # FC layer
        x = F.relu(self.fc1(combined))

        # RNN layer (if enabled)
        if self.use_rnn:
            if hidden_state is None:
                hidden_state = self.init_hidden().expand(batch_size, -1).to(x.device)

            h = self.rnn(x, hidden_state)
            x = h
        else:
            h = None

        # Output Q-values
        q_values = self.fc2(x)

        return q_values, h


def test_cnn_agent():
    """Test CNN agent with different grid sizes (scale invariance)."""
    print("=" * 60)
    print("Testing CNN Agent - Scale Invariance")
    print("=" * 60)

    # Create agent
    agent = CNNAgent(
        input_channels=4,
        scalar_size=56,
        hidden_dim=256,
        n_actions=9,
        cnn_channels=[32, 64, 64],
        pooled_size=(4, 4),
    )

    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # Test with 20×20 grid
    print("\nTest 1: 20×20 grid")
    spatial_20 = torch.randn(1, 4, 20, 20)
    scalar_20 = torch.randn(1, 56)

    q_values_20 = agent(spatial_20, scalar_20)
    print(f"Input spatial shape: {spatial_20.shape}")
    print(f"Input scalar shape: {scalar_20.shape}")
    print(f"Output Q-values shape: {q_values_20.shape}")
    print(f"Q-values: {q_values_20[0].detach().numpy()}")

    # Test with 30×30 grid (scale invariance!)
    print("\nTest 2: 30×30 grid (scale invariance test)")
    spatial_30 = torch.randn(1, 4, 30, 30)
    scalar_30 = torch.randn(1, 56)

    q_values_30 = agent(spatial_30, scalar_30)
    print(f"Input spatial shape: {spatial_30.shape}")
    print(f"Input scalar shape: {scalar_30.shape}")
    print(f"Output Q-values shape: {q_values_30.shape}")
    print(f"Q-values: {q_values_30[0].detach().numpy()}")

    print("\n✓ Scale invariance confirmed:")
    print("  - Same agent handles both 20×20 and 30×30 grids")
    print("  - Adaptive pooling ensures fixed-size features")
    print("  - Output Q-values have same shape regardless of input size")

    # Test batch processing
    print("\nTest 3: Batch processing (4 agents)")
    spatial_batch = torch.randn(4, 4, 20, 20)
    scalar_batch = torch.randn(4, 56)

    q_values_batch = agent(spatial_batch, scalar_batch)
    print(f"Batch spatial shape: {spatial_batch.shape}")
    print(f"Batch scalar shape: {scalar_batch.shape}")
    print(f"Batch Q-values shape: {q_values_batch.shape}")

    print("\n✓ Batch processing works correctly")

    print("\n" + "=" * 60)
    print("✓ CNN Agent Tests Passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_cnn_agent()
