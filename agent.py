"""
MARL Coverage Agent
Contains the agent logic for multi-agent coverage planning.
"""

import os
import copy
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from typing import Tuple, List, Optional

from data_structures import RobotState, CommunicationEvent
from networks import ConvDQN, DuelingConvDQN
from replay_memory import ReplayMemory
from utils import ensure_dir


class MARLCoverageAgent:
    """Multi-Agent Reinforcement Learning agent for coverage planning."""

    def __init__(
        self,
        agent_id: int,
        initial_state: RobotState,
        environment,
        grid_size: int,
        sensor_range: int,
        comm_range: float = 5.0,
        memory_capacity: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.999,
        learning_rate: float = 0.0005,
        target_update_freq: int = 1000,
        soft_update_tau: float = 0.005,
        use_soft_update: bool = True,
        use_dueling: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize MARL Coverage Agent.

        Args:
            agent_id: Unique identifier for this agent
            initial_state: Initial robot state (position, orientation)
            environment: Reference to the environment
            grid_size: Size of the grid
            sensor_range: Range of the sensor
            comm_range: Communication range
            memory_capacity: Size of replay memory
            batch_size: Batch size for training
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for epsilon
            learning_rate: Learning rate for optimizer
            target_update_freq: Frequency of target network updates
            soft_update_tau: Tau for soft target updates
            use_soft_update: Whether to use soft or hard target updates
            use_dueling: Whether to use Dueling DQN architecture
            device: Device to use (cpu or cuda)
        """
        self.agent_id = agent_id
        self.state = initial_state
        self.env = environment
        self.grid_size = grid_size
        self.sensor_range = sensor_range
        self.comm_range = comm_range
        self.device = torch.device(device)
        self.steps_done = 0  # Tracks total steps agent has taken
        self.optimization_steps = 0  # Tracks how many times optimize_model was called

        # DQN parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_soft_update = use_soft_update
        self.soft_update_tau = soft_update_tau

        # Define movement options: 8 directions plus "stay"
        self.actions = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),  (0, 0),  (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]
        self.action_size = len(self.actions)

        # Network setup
        self.input_channels = 2  # Coverage map + Obstacle map
        self.feature_dim = 2     # Orientation features (sin, cos)

        # Choose network architecture
        NetworkClass = DuelingConvDQN if use_dueling else ConvDQN
        self.policy_net = NetworkClass(
            self.input_channels, grid_size, self.action_size, self.feature_dim
        ).to(self.device)
        self.target_net = NetworkClass(
            self.input_channels, grid_size, self.action_size, self.feature_dim
        ).to(self.device)

        # Initialize target network with policy network's weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for inference

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = ReplayMemory(memory_capacity)

        # Local knowledge (agent's perspective)
        self.local_map = nx.Graph()  # Agent's own map built from observations/communication
        self.known_positions = {}  # Other agents' last known positions {agent_id: (r, c)}
        self.communication_history = []

        # Metrics tracking for this agent
        self.losses = []
        self.q_values_history = []  # History of Q-values

        # Priority (not currently used, but could be for comms/conflicts)
        self.priority = random.random()

    def _initialize_local_map(self):
        """Initialize agent's local map based on the current global map."""
        self.local_map = nx.Graph()  # Reset map
        # Check if environment and its world state/graph exist
        if self.env and self.env.world_state and self.env.world_state.graph:
            # Deep copy nodes and edges from the global graph
            self.local_map = self.env.world_state.graph.copy()
        else:
            print(f"Warning: Agent {self.agent_id} could not initialize local map. Environment state not ready.")

    def update_local_map(self, node: Tuple[int, int], pc_value: float, node_type: Optional[str] = None):
        """
        Update local map with new coverage or type information.

        Args:
            node: Node position to update
            pc_value: Coverage probability value
            node_type: Type of node ('covered', 'free', 'occupied')

        Returns:
            bool: Whether an update was made
        """
        if not self.local_map.has_node(node):
            if node_type == 'occupied':
                pass  # Obstacle info handled by get_state_tensor
            return False

        # Node exists in local (traversable) graph, update its data
        updated = False
        current_pc = self.local_map.nodes[node].get('pc', -1.0)
        if pc_value > current_pc:
            self.local_map.nodes[node]['pc'] = pc_value
            updated = True
            # Update type based on coverage threshold
            if pc_value >= self.env.coverage_threshold:
                self.local_map.nodes[node]['type'] = 'covered'

        # Update type information if provided
        if node_type == 'covered' and self.local_map.nodes[node].get('type') != 'covered':
            self.local_map.nodes[node]['type'] = 'covered'
            updated = True

        return updated

    def communicate(self, other_agent: 'MARLCoverageAgent', env_time: float) -> bool:
        """
        Exchange local map information with another agent if within communication range.

        Args:
            other_agent: The other agent to communicate with
            env_time: Current environment time

        Returns:
            bool: Whether communication occurred
        """
        if self._is_within_comm_range(other_agent):
            # Exchange coverage and type information
            map_updated = self._exchange_map_info(other_agent)

            # Exchange current position information
            self.known_positions[other_agent.agent_id] = other_agent.state.position
            other_agent.known_positions[self.agent_id] = self.state.position

            # Record a communication event if map info was actually exchanged
            if map_updated:
                event = CommunicationEvent(
                    sender_id=self.agent_id,
                    receiver_id=other_agent.agent_id,
                    data_type='map_exchange',
                    timestamp=env_time
                )
                # Record symmetrically
                self.communication_history.append(event)
                other_agent.communication_history.append(copy.deepcopy(event))
                return True
        return False

    def _is_within_comm_range(self, other_agent: 'MARLCoverageAgent') -> bool:
        """Check if another agent is within communication range."""
        dist_sq = sum((p1 - p2)**2 for p1, p2 in zip(self.state.position, other_agent.state.position))
        return dist_sq <= self.comm_range**2

    def _exchange_map_info(self, other_agent: 'MARLCoverageAgent') -> bool:
        """Merge knowledge from other agent's local map into this agent's map."""
        updates_made_by_me = False
        updates_made_by_other = False

        # Iterate through nodes the other agent knows about
        nodes_to_process = list(other_agent.local_map.nodes())

        for node in nodes_to_process:
            other_data = other_agent.local_map.nodes[node]
            other_pc = other_data.get('pc', 0.0)
            other_type = other_data.get('type', None)

            # Try to update my map with the other agent's info
            updated_me = self.update_local_map(node, other_pc, other_type)
            updates_made_by_me |= updated_me

            # Symmetrically, update the other agent with my info for that node
            if self.local_map.has_node(node):
                my_data = self.local_map.nodes[node]
                my_pc = my_data.get('pc', 0.0)
                my_type = my_data.get('type', None)
                updated_other = other_agent.update_local_map(node, my_pc, my_type)
                updates_made_by_other |= updated_other

        return updates_made_by_me or updates_made_by_other

    def get_state_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create state tensors for the neural network input.

        Returns:
            tuple: (grid_tensor, feature_tensor)
        """
        # Initialize grids
        coverage_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        # Obstacle grid from environment
        obstacle_grid = self.env.obstacle_grid.astype(np.float32) if self.env.obstacle_grid is not None else np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Populate coverage grid using the agent's local map knowledge
        for node, data in self.local_map.nodes(data=True):
            r, c = node
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                if obstacle_grid[r, c] == 0.0:
                    coverage_grid[r, c] = data.get('pc', 0.0)

        # Stack grids: [channels, height, width]
        grid_stack = np.stack([coverage_grid, obstacle_grid], axis=0)
        grid_tensor = torch.from_numpy(grid_stack).to(self.device)

        # Create feature tensor (orientation)
        orientation_features = np.array([
            math.sin(self.state.orientation),
            math.cos(self.state.orientation)
        ], dtype=np.float32)
        feature_tensor = torch.from_numpy(orientation_features).to(self.device)

        return grid_tensor, feature_tensor

    def select_action(self, state: Tuple[torch.Tensor, torch.Tensor], valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select an action using epsilon-greedy strategy.

        Args:
            state: Current state tensors
            valid_actions: List of valid actions

        Returns:
            Selected action
        """
        self.steps_done += 1

        if not valid_actions:
            return (0, 0)  # Default to staying still if trapped

        sample = random.random()
        if sample < self.epsilon:
            action = random.choice(valid_actions)
        else:
            grid_tensor, feature_tensor = state
            with torch.no_grad():
                q_values = self.policy_net(grid_tensor.unsqueeze(0), feature_tensor.unsqueeze(0))
                q_values_np = q_values.squeeze(0).cpu().numpy()

            self.q_values_history.append(q_values_np.tolist())
            best_q = -float('inf')
            best_action = valid_actions[0]
            found_valid_q = False

            for act in valid_actions:
                try:
                    idx = self.actions.index(act)
                    if 0 <= idx < len(q_values_np):
                        if q_values_np[idx] > best_q:
                            best_q = q_values_np[idx]
                            best_action = act
                            found_valid_q = True
                    else:
                        print(f"Error Agent {self.agent_id}: Action index {idx} out of bounds.")
                except ValueError:
                    print(f"Error Agent {self.agent_id}: Action {act} not standard.")

            if not found_valid_q:
                action = random.choice(valid_actions)
            else:
                action = best_action

        return action

    def optimize_model(self):
        """
        Perform one step of optimization on the policy network.

        Returns:
            Loss value if optimization occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None

        self.optimization_steps += 1
        transitions = self.memory.sample(self.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*transitions)

        state_grids = torch.stack([s[0] for s in state_batch]).to(self.device)
        state_features = torch.stack([s[1] for s in state_batch]).to(self.device)
        non_final_mask = torch.tensor([not d for d in done_batch], device=self.device, dtype=torch.bool)
        non_final_next_grids = torch.stack([ns[0] for ns, d in zip(next_state_batch, done_batch) if not d]).to(self.device)
        non_final_next_features = torch.stack([ns[1] for ns, d in zip(next_state_batch, done_batch) if not d]).to(self.device)
        action_indices = [self.actions.index(a) for a in action_batch]
        action_tensor = torch.tensor(action_indices, device=self.device, dtype=torch.long).unsqueeze(1)
        reward_tensor = torch.tensor(reward_batch, device=self.device, dtype=torch.float)

        state_action_q_values = self.policy_net(state_grids, state_features).gather(1, action_tensor)
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        if non_final_next_grids.size(0) > 0:
            with torch.no_grad():
                next_q_values = self.target_net(non_final_next_grids, non_final_next_features)
                next_state_values[non_final_mask] = next_q_values.max(1)[0].detach()

        expected_state_action_values = reward_tensor + (self.gamma * next_state_values)
        loss = F.smooth_l1_loss(state_action_q_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update the target network weights."""
        if self.use_soft_update:
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.soft_update_tau + target_net_state_dict[key] * (1 - self.soft_update_tau)
            self.target_net.load_state_dict(target_net_state_dict)
        else:  # Hard Update
            if self.optimization_steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        """Decay exploration rate epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_state(self, new_state: RobotState):
        """Update the agent's internal knowledge of its own state."""
        self.state = new_state

    def save_model(self, path):
        """Save the agent's model state."""
        ensure_dir(os.path.dirname(path))
        try:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps_done': self.steps_done,
                'optimization_steps': self.optimization_steps,
            }, path)
        except Exception as e:
            print(f"Error saving model for agent {self.agent_id} to {path}: {e}")

    def load_model(self, path):
        """
        Load the agent's model state.

        Args:
            path: Path to model file

        Returns:
            bool: Whether loading was successful
        """
        if not os.path.exists(path):
            print(f"Error: Model file not found at {path} for agent {self.agent_id}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
            self.steps_done = checkpoint.get('steps_done', 0)
            self.optimization_steps = checkpoint.get('optimization_steps', 0)
            self.policy_net.to(self.device)
            self.target_net.to(self.device)
            self.target_net.eval()
            print(f"Agent {self.agent_id}: Model loaded successfully from {path}")
            return True
        except Exception as e:
            print(f"Error loading model for agent {self.agent_id} from {path}: {e}")
            return False
