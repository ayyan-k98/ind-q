"""
CNN Multi-Agent Controller (MAC) for EPyMARL

This MAC uses CNN agents instead of RNN agents.
Compatible with QMIX mixer - operates on Q-values regardless of how they're computed.

To integrate with EPyMARL:
1. Copy this file to: epymarl/src/controllers/cnn_mac.py
2. Register in epymarl/src/controllers/__init__.py
3. Use with: --config=qmix_cnn
"""

import torch as th
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY


class CNNMAC:
    """
    Multi-Agent Controller using CNN agents.

    Handles spatial observations and produces Q-values for QMIX.
    """

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args

        # Get input shapes from scheme
        # For CNN: expect dict with 'spatial_shape' and 'scalar_size'
        obs_shape = scheme["obs"]["vshape"]

        if isinstance(obs_shape, dict):
            # CNN observations
            self.use_cnn = True
            self.spatial_shape = obs_shape['spatial_shape']  # (4, H, W)
            self.scalar_size = obs_shape['scalar_size']  # 56
        else:
            # Flat observations (fallback to MLP)
            self.use_cnn = False
            self.input_shape = obs_shape

        # Action space
        self.n_actions = args.n_actions

        # Create agent
        if self.use_cnn:
            # CNN agent
            from .cnn_agent import CNNAgent

            self.agent = CNNAgent(
                input_channels=self.spatial_shape[0],
                scalar_size=self.scalar_size,
                hidden_dim=getattr(args, "cnn_hidden_dim", 256),
                n_actions=self.n_actions,
                cnn_channels=getattr(args, "cnn_channels", [32, 64, 64]),
                pooled_size=tuple(getattr(args, "cnn_pooled_size", [4, 4])),
            )
        else:
            # Fallback to standard agent (MLP/RNN)
            self.agent = agent_REGISTRY[args.agent](self.input_shape, args)

        # Action selector
        self.action_selector = action_REGISTRY[args.action_selector](args)

        # Hidden states (for RNN compatibility - not used with CNN)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """
        Select actions for all agents.

        Args:
            ep_batch: Episode batch
            t_ep: Timestep in episode
            t_env: Global timestep
            bs: Batch slice
            test_mode: Whether in test mode

        Returns:
            actions: (batch, n_agents, 1) selected actions
        """
        # Get available actions
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        # Forward pass
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)

        # Select actions
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )

        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        """
        Forward pass through agents.

        Args:
            ep_batch: Episode batch
            t: Timestep
            test_mode: Whether in test mode

        Returns:
            agent_outs: (batch, n_agents, n_actions) Q-values
        """
        if self.use_cnn:
            return self._forward_cnn(ep_batch, t)
        else:
            return self._forward_standard(ep_batch, t)

    def _forward_cnn(self, ep_batch, t):
        """Forward pass with CNN agent."""
        batch_size = ep_batch.batch_size

        # Get observations
        obs = ep_batch["obs"][:, t]  # (batch, n_agents, obs)

        # Observations are dicts with 'spatial' and 'scalars'
        # Need to extract and batch them properly

        # Extract spatial observations (batch, n_agents, 4, H, W)
        spatial_list = []
        scalar_list = []

        for batch_idx in range(batch_size):
            for agent_idx in range(self.n_agents):
                obs_dict = obs[batch_idx, agent_idx]
                spatial_list.append(obs_dict['spatial'])
                scalar_list.append(obs_dict['scalars'])

        # Stack into tensors
        spatial_batch = th.stack(spatial_list)  # (batch*n_agents, 4, H, W)
        scalar_batch = th.stack(scalar_list)  # (batch*n_agents, 56)

        # Forward through agent
        q_values = self.agent(spatial_batch, scalar_batch)  # (batch*n_agents, n_actions)

        # Reshape to (batch, n_agents, n_actions)
        q_values = q_values.view(batch_size, self.n_agents, -1)

        return q_values

    def _forward_standard(self, ep_batch, t):
        """Forward pass with standard agent (MLP/RNN)."""
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs, self.hidden_states = self.agent(
            agent_inputs, self.hidden_states
        )

        return agent_outs

    def _build_inputs(self, ep_batch, t):
        """Build inputs for standard agent."""
        # Observations
        bs = ep_batch.batch_size
        inputs = []

        # Observation
        inputs.append(ep_batch["obs"][:, t])

        # Last actions (one-hot)
        if t == 0:
            inputs.append(th.zeros_like(ep_batch["actions_onehot"][:, t]))
        else:
            inputs.append(ep_batch["actions_onehot"][:, t - 1])

        # Agent ID (one-hot)
        inputs.append(
            th.eye(self.n_agents, device=ep_batch.device)
            .unsqueeze(0)
            .expand(bs, -1, -1)
        )

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def init_hidden(self, batch_size):
        """Initialize hidden states."""
        if self.use_cnn:
            # CNN doesn't use hidden states
            self.hidden_states = None
        else:
            # Standard agent hidden states
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(
                batch_size, self.n_agents, -1
            )

    def parameters(self):
        """Get agent parameters for optimization."""
        return self.agent.parameters()

    def load_state(self, other_mac):
        """Load state from another MAC."""
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        """Move to CUDA."""
        self.agent.cuda()

    def save_models(self, path):
        """Save agent model."""
        th.save(self.agent.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        """Load agent model."""
        self.agent.load_state_dict(
            th.load(f"{path}/agent.th", map_location=lambda storage, loc: storage)
        )

    def _get_input_shape(self, scheme):
        """Get input shape from scheme."""
        input_shape = scheme["obs"]["vshape"]

        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]

        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape


# Register the controller
REGISTRY = {}
REGISTRY["cnn_mac"] = CNNMAC
