"""
Mock MultiAgentEnv base class for standalone testing.
In EPyMARL, this is provided by the framework.
"""

class MultiAgentEnv:
    """Base class for multi-agent environments."""

    def __init__(self):
        pass

    def step(self, actions):
        """Execute one timestep."""
        raise NotImplementedError

    def get_obs(self):
        """Get observations for all agents."""
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """Get observation for single agent."""
        raise NotImplementedError

    def get_obs_size(self):
        """Get observation size."""
        raise NotImplementedError

    def get_state(self):
        """Get global state."""
        raise NotImplementedError

    def get_state_size(self):
        """Get state size."""
        raise NotImplementedError

    def get_avail_actions(self):
        """Get available actions for all agents."""
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """Get available actions for single agent."""
        raise NotImplementedError

    def get_total_actions(self):
        """Get number of actions."""
        raise NotImplementedError

    def reset(self):
        """Reset environment."""
        raise NotImplementedError

    def render(self):
        """Render environment."""
        pass

    def close(self):
        """Close environment."""
        pass

    def seed(self):
        """Set random seed."""
        pass

    def get_env_info(self):
        """Get environment info."""
        raise NotImplementedError
