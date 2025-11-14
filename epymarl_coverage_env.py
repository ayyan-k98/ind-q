"""
EPyMARL-Compatible Multi-Agent Coverage Environment
Implements the MultiAgentEnv interface required by EPyMARL
"""

import numpy as np
import networkx as nx
import math
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

# EPyMARL imports (will be available after setup)
try:
    from envs.multiagentenv import MultiAgentEnv
except ImportError:
    # Fallback for development/testing
    class MultiAgentEnv:
        def step(self, actions): raise NotImplementedError
        def get_obs(self): raise NotImplementedError
        def get_obs_agent(self, agent_id): raise NotImplementedError
        def get_obs_size(self): raise NotImplementedError
        def get_state(self): raise NotImplementedError
        def get_state_size(self): raise NotImplementedError
        def get_avail_actions(self): raise NotImplementedError
        def get_avail_agent_actions(self, agent_id): raise NotImplementedError
        def get_total_actions(self): raise NotImplementedError
        def reset(self): raise NotImplementedError
        def render(self): pass
        def close(self): pass
        def seed(self): pass
        def get_env_info(self): raise NotImplementedError


class CoverageEnvironment(MultiAgentEnv):
    """
    Multi-Agent Coverage Environment for EPyMARL

    Agents must cooperatively cover a 2D grid with obstacles.
    Uses raycasting for realistic coverage simulation.
    """

    def __init__(
        self,
        n_agents: int = 4,
        grid_size: int = 20,
        episode_limit: int = 200,
        sensor_range: int = 5,
        fov_degrees: float = 120.0,
        coverage_threshold: float = 0.80,
        completion_threshold: float = 95.0,
        obs_size: int = 64,
        map_type: str = 'empty',  # 'empty', 'rooms', 'random'
        obstacle_density: float = 0.15,  # For random maps
        reward_scale_coverage: float = 100.0,
        reward_scale_shaping: float = 1.0,
        reward_frontier_bonus: float = 0.5,
        reward_spread_bonus: float = 0.1,
        reward_stay_penalty: float = 0.2,
        reward_step_penalty: float = 0.1,
        seed: int = None
    ):
        """
        Initialize coverage environment.

        Args:
            n_agents: Number of agents
            grid_size: Size of the grid (grid_size × grid_size)
            episode_limit: Maximum steps per episode
            sensor_range: Range of agent sensors for coverage
            fov_degrees: Field of view in degrees
            coverage_threshold: PC value to consider a cell "covered"
            completion_threshold: Coverage % to end episode
            obs_size: Fixed observation vector size
            map_type: Type of map ('empty', 'rooms', 'random')
            obstacle_density: Density of obstacles for random maps
            reward_*: Reward scaling parameters
            seed: Random seed
        """
        super().__init__()

        # Environment parameters
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.episode_limit = episode_limit
        self.sensor_range = sensor_range
        self.fov_radians = math.radians(fov_degrees)
        self.coverage_threshold = coverage_threshold
        self.completion_threshold = completion_threshold
        self.obs_size = obs_size
        self.map_type = map_type
        self.obstacle_density = obstacle_density

        # Reward parameters
        self.reward_scale_coverage = reward_scale_coverage
        self.reward_scale_shaping = reward_scale_shaping
        self.reward_frontier_bonus = reward_frontier_bonus
        self.reward_spread_bonus = reward_spread_bonus
        self.reward_stay_penalty = reward_stay_penalty
        self.reward_step_penalty = reward_step_penalty

        # Action space: 9 actions (8 directions + stay)
        self.actions = [
            (-1, -1), (-1, 0), (-1, 1),  # NW, N, NE
            (0, -1),  (0, 0),  (0, 1),   # W, Stay, E
            (1, -1),  (1, 0),  (1, 1)    # SW, S, SE
        ]
        self.n_actions = len(self.actions)

        # State
        self.steps = 0
        self.agent_positions = []  # List of (r, c) tuples
        self.agent_orientations = []  # List of radians
        self.obstacle_grid = None  # [grid_size, grid_size] binary
        self.coverage_grid = None  # [grid_size, grid_size] float [0, 1]
        self.world_graph = None  # NetworkX graph of traversable cells

        # Metrics
        self.cumulative_reward = 0.0
        self.previous_coverage_sum = 0.0

        # Random seed
        if seed is not None:
            np.random.seed(seed)

        # Initialize environment
        self.reset()

    # ========== EPyMARL Required Methods ==========

    def reset(self):
        """Reset environment to initial state."""
        self.steps = 0
        self.cumulative_reward = 0.0

        # Generate map
        self.obstacle_grid = self._generate_map(self.map_type)
        self.coverage_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Create graph of traversable cells
        self._build_world_graph()

        # Initialize agents
        self._initialize_agents()

        # Initial coverage from agent positions
        for agent_id in range(self.n_agents):
            self._raycast_update(agent_id)

        self.previous_coverage_sum = self._calculate_coverage_sum()

        return self.get_obs(), self.get_state()

    def step(self, actions: List[int]):
        """
        Execute one step.

        Args:
            actions: List of action indices [a1, a2, ..., an]

        Returns:
            reward: Team reward (float)
            terminated: Whether episode ended (bool)
            info: Additional information (dict)
        """
        self.steps += 1

        # Convert action indices to movements
        action_moves = [self.actions[a] for a in actions]

        # Resolve conflicts and move agents
        executed_moves = self._resolve_conflicts(action_moves)
        for agent_id, move in enumerate(executed_moves):
            self._move_agent(agent_id, move)

        # Update coverage
        for agent_id in range(self.n_agents):
            self._raycast_update(agent_id)

        # Calculate reward
        reward, reward_info = self._calculate_reward(actions, executed_moves)
        self.cumulative_reward += reward

        # Check termination
        coverage_pct = self._calculate_coverage_percentage()
        terminated = (
            self.steps >= self.episode_limit or
            coverage_pct >= self.completion_threshold
        )

        # Info
        info = {
            **reward_info,
            'coverage_pct': coverage_pct,
            'steps': self.steps,
            'episode_limit': terminated and self.steps >= self.episode_limit,
        }

        if terminated:
            info['episode'] = {
                'r': self.cumulative_reward,
                'l': self.steps,
                'coverage': coverage_pct
            }

        return reward, terminated, info

    def get_obs(self):
        """Get observations for all agents."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id: int):
        """
        Get observation for a single agent.

        Returns:
            obs: Numpy array of shape [obs_size]
        """
        obs = []
        pos = self.agent_positions[agent_id]

        # === 1. Own State [5] ===
        obs.extend([
            pos[0] / self.grid_size,  # Normalized row
            pos[1] / self.grid_size,  # Normalized col
            np.sin(self.agent_orientations[agent_id]),  # Orientation
            np.cos(self.agent_orientations[agent_id]),
            self.steps / self.episode_limit  # Time remaining
        ])

        # === 2. Sensor Region Info [6] ===
        sensor_info = self._get_sensor_info(agent_id)
        obs.extend([
            sensor_info['uncovered_ratio'],
            sensor_info['covered_ratio'],
            sensor_info['frontier_distance'] / self.grid_size,
            np.sin(sensor_info['frontier_angle']),
            np.cos(sensor_info['frontier_angle']),
            sensor_info['coverage_rate']
        ])

        # === 3. Other Agents [4 × max_agents] ===
        max_other_agents = 10  # Support up to 11 agents total
        for other_id in range(self.n_agents):
            if other_id == agent_id:
                continue

            other_pos = self.agent_positions[other_id]
            dx = (other_pos[0] - pos[0]) / self.grid_size
            dy = (other_pos[1] - pos[1]) / self.grid_size
            dist = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)

            obs.extend([dx, dy, dist, angle])

        # Pad other agents to max
        current_others = self.n_agents - 1
        if current_others < max_other_agents:
            obs.extend([0.0] * (4 * (max_other_agents - current_others)))

        # === 4. Global Statistics [3] ===
        obs.extend([
            self._calculate_coverage_percentage() / 100.0,
            self._calculate_coverage_sum() / (self.grid_size * self.grid_size),
            (self._calculate_coverage_sum() - self.previous_coverage_sum)
        ])

        # === Total: 5 + 6 + 40 + 3 = 54, pad to 64 ===
        current_size = len(obs)
        if current_size < self.obs_size:
            obs.extend([0.0] * (self.obs_size - current_size))
        elif current_size > self.obs_size:
            obs = obs[:self.obs_size]

        return np.array(obs, dtype=np.float32)

    def get_obs_size(self):
        """Return observation size."""
        return self.obs_size

    def get_state(self):
        """
        Get global state for centralized training.

        Returns:
            state: Numpy array of shape [state_size]
        """
        # Concatenate all agent observations
        all_obs = np.concatenate([self.get_obs_agent(i) for i in range(self.n_agents)])

        # Add coverage grid (flattened)
        coverage_flat = self.coverage_grid.flatten()

        # Add metadata
        metadata = np.array([
            self.steps / self.episode_limit,
            self._calculate_coverage_percentage() / 100.0,
            self.n_agents / 10.0,  # Normalized
        ], dtype=np.float32)

        state = np.concatenate([all_obs, coverage_flat, metadata])

        return state.astype(np.float32)

    def get_state_size(self):
        """Return state size."""
        all_obs_size = self.obs_size * self.n_agents
        coverage_size = self.grid_size * self.grid_size
        metadata_size = 3
        return all_obs_size + coverage_size + metadata_size

    def get_avail_actions(self):
        """Get available actions for all agents."""
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id: int):
        """
        Get available actions for a single agent.

        Returns:
            avail: List of 0/1 indicating valid actions [n_actions]
        """
        avail = [0] * self.n_actions
        pos = self.agent_positions[agent_id]

        for i, (dr, dc) in enumerate(self.actions):
            target = (pos[0] + dr, pos[1] + dc)

            # Check if valid
            if self._is_valid_position(target, agent_id):
                avail[i] = 1

        # Always allow staying (action 4)
        avail[4] = 1

        return avail

    def get_total_actions(self):
        """Return number of actions per agent."""
        return self.n_actions

    def get_env_info(self):
        """Return environment information for EPyMARL."""
        return {
            'n_agents': self.n_agents,
            'n_actions': self.n_actions,
            'state_shape': self.get_state_size(),
            'obs_shape': self.get_obs_size(),
            'episode_limit': self.episode_limit,
        }

    def render(self):
        """Render environment (optional)."""
        # Print ASCII representation
        print(f"\nStep {self.steps}/{self.episode_limit}")
        print(f"Coverage: {self._calculate_coverage_percentage():.1f}%")

        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)

        # Add obstacles
        grid[self.obstacle_grid == 1] = '#'

        # Add coverage
        covered = self.coverage_grid >= self.coverage_threshold
        grid[covered & (self.obstacle_grid == 0)] = '·'

        # Add agents
        for i, pos in enumerate(self.agent_positions):
            grid[pos[0], pos[1]] = str(i)

        # Print
        print('  ' + ''.join(str(i % 10) for i in range(self.grid_size)))
        for i, row in enumerate(grid):
            print(f"{i % 10:2d}" + ''.join(row))
        print()

    def close(self):
        """Cleanup."""
        pass

    def seed(self, seed: int = None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)

    # ========== Internal Methods ==========

    def _generate_map(self, map_type: str) -> np.ndarray:
        """Generate obstacle map."""
        if map_type == 'empty':
            return self._generate_empty_map()
        elif map_type == 'rooms':
            return self._generate_room_map()
        elif map_type == 'random':
            return self._generate_random_map()
        else:
            raise ValueError(f"Unknown map type: {map_type}")

    def _generate_empty_map(self) -> np.ndarray:
        """Generate empty map with borders."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        grid[0, :] = 1.0
        grid[-1, :] = 1.0
        grid[:, 0] = 1.0
        grid[:, -1] = 1.0
        return grid

    def _generate_room_map(self) -> np.ndarray:
        """Generate map with rooms (simplified version)."""
        grid = np.ones((self.grid_size, self.grid_size), dtype=np.float32)

        # Create a few rooms
        rooms = [
            (2, 2, 6, 6),   # (r, c, h, w)
            (2, 10, 6, 6),
            (10, 2, 6, 6),
            (10, 10, 6, 6),
        ]

        for r, c, h, w in rooms:
            if r + h < self.grid_size and c + w < self.grid_size:
                grid[r:r+h, c:c+w] = 0.0

        # Connect rooms with corridors
        grid[4:6, :] = 0.0  # Horizontal corridor
        grid[:, 8:10] = 0.0  # Vertical corridor

        # Borders
        grid[0, :] = 1.0
        grid[-1, :] = 1.0
        grid[:, 0] = 1.0
        grid[:, -1] = 1.0

        return grid

    def _generate_random_map(self) -> np.ndarray:
        """Generate map with random obstacles."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Random obstacles in interior
        interior = grid[1:-1, 1:-1]
        interior[:] = (np.random.rand(self.grid_size-2, self.grid_size-2) < self.obstacle_density).astype(np.float32)

        # Borders
        grid[0, :] = 1.0
        grid[-1, :] = 1.0
        grid[:, 0] = 1.0
        grid[:, -1] = 1.0

        return grid

    def _build_world_graph(self):
        """Build NetworkX graph of traversable cells."""
        self.world_graph = nx.Graph()

        # Add all free cells as nodes
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.obstacle_grid[r, c] == 0:
                    self.world_graph.add_node((r, c), pc=0.0, type='free')

        # Add edges (4-connectivity for simplicity)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) in self.world_graph:
                    # Check neighbors
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in self.world_graph:
                            self.world_graph.add_edge((r, c), (nr, nc))

    def _initialize_agents(self):
        """Initialize agent positions and orientations."""
        self.agent_positions = []
        self.agent_orientations = []

        # Get free positions
        free_positions = list(self.world_graph.nodes())

        if len(free_positions) < self.n_agents:
            raise ValueError(f"Not enough free positions for {self.n_agents} agents")

        # Try to place in corners first
        corners = [
            (1, 1), (1, self.grid_size-2),
            (self.grid_size-2, 1), (self.grid_size-2, self.grid_size-2)
        ]

        placed = []
        for corner in corners:
            if corner in free_positions and len(placed) < self.n_agents:
                placed.append(corner)

        # Fill remaining with random positions
        while len(placed) < self.n_agents:
            pos = free_positions[np.random.randint(len(free_positions))]
            if pos not in placed:
                placed.append(pos)

        self.agent_positions = placed[:self.n_agents]
        self.agent_orientations = [np.random.uniform(0, 2*np.pi) for _ in range(self.n_agents)]

    def _is_valid_position(self, pos: Tuple[int, int], agent_id: int) -> bool:
        """Check if position is valid for agent."""
        r, c = pos

        # Out of bounds
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return False

        # Obstacle
        if self.obstacle_grid[r, c] == 1:
            return False

        # Another agent (allow same position as self)
        for other_id, other_pos in enumerate(self.agent_positions):
            if other_id != agent_id and pos == other_pos:
                return False

        return True

    def _resolve_conflicts(self, moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Resolve movement conflicts (multiple agents to same cell).
        Uses ID-based priority: lower ID wins.
        """
        target_positions = {}  # target -> list of agent_ids

        for agent_id, move in enumerate(moves):
            current = self.agent_positions[agent_id]
            target = (current[0] + move[0], current[1] + move[1])

            if target != current:  # If actually moving
                if target not in target_positions:
                    target_positions[target] = []
                target_positions[target].append(agent_id)

        # Resolve conflicts
        executed = list(moves)
        for target, competing_agents in target_positions.items():
            if len(competing_agents) > 1:
                # Sort by ID, lowest wins
                competing_agents.sort()
                winner = competing_agents[0]

                # Others stay in place
                for loser in competing_agents[1:]:
                    executed[loser] = (0, 0)  # Stay

        return executed

    def _move_agent(self, agent_id: int, move: Tuple[int, int]):
        """Move agent and update orientation."""
        current = self.agent_positions[agent_id]
        target = (current[0] + move[0], current[1] + move[1])

        # Validate (should already be valid from conflict resolution)
        if self._is_valid_position(target, agent_id):
            self.agent_positions[agent_id] = target

            # Update orientation if moving
            if move != (0, 0):
                self.agent_orientations[agent_id] = math.atan2(move[1], move[0])

    def _raycast_update(self, agent_id: int):
        """Update coverage using raycasting from agent position."""
        pos = self.agent_positions[agent_id]
        orientation = self.agent_orientations[agent_id]

        center = np.array(pos, dtype=float)

        # Cover current cell
        if self.world_graph.has_node(pos):
            self.coverage_grid[pos[0], pos[1]] = 1.0
            self.world_graph.nodes[pos]['pc'] = 1.0

        # Raycast within FOV
        half_fov = self.fov_radians / 2
        start_angle = orientation - half_fov
        end_angle = orientation + half_fov
        num_rays = 32
        step_size = 0.5
        max_range = float(self.sensor_range)

        for i in range(num_rays):
            fraction = i / max(num_rays - 1, 1)
            angle = start_angle + fraction * (end_angle - start_angle)

            ray_dist = 0.0
            while ray_dist <= max_range:
                ray_dist += step_size
                offset = np.array([math.cos(angle), math.sin(angle)]) * ray_dist
                ray_pos = center + offset
                ray_cell = tuple(np.round(ray_pos).astype(int))

                # Out of bounds
                if not (0 <= ray_cell[0] < self.grid_size and 0 <= ray_cell[1] < self.grid_size):
                    break

                # Hit obstacle
                if self.obstacle_grid[ray_cell[0], ray_cell[1]] == 1:
                    break

                # Update coverage
                if self.world_graph.has_node(ray_cell):
                    if ray_dist <= 1.0:
                        new_pc = 1.0
                    else:
                        # Sigmoid decay
                        r0 = max_range / 2.5
                        k = 2.0
                        new_pc = np.clip(1.0 / (1.0 + np.exp(k * (ray_dist - r0))), 0.0, 1.0)

                    # Monotonic update
                    current_pc = self.coverage_grid[ray_cell[0], ray_cell[1]]
                    if new_pc > current_pc:
                        self.coverage_grid[ray_cell[0], ray_cell[1]] = new_pc
                        self.world_graph.nodes[ray_cell]['pc'] = new_pc

    def _calculate_coverage_sum(self) -> float:
        """Calculate sum of coverage probabilities."""
        return float(np.sum(self.coverage_grid[self.obstacle_grid == 0]))

    def _calculate_coverage_percentage(self) -> float:
        """Calculate percentage of cells above threshold."""
        free_cells = self.obstacle_grid == 0
        covered_cells = (self.coverage_grid >= self.coverage_threshold) & free_cells
        total_free = np.sum(free_cells)

        if total_free == 0:
            return 100.0

        return (np.sum(covered_cells) / total_free) * 100.0

    def _calculate_reward(self, actions: List[int], executed_moves: List[Tuple[int, int]]) -> Tuple[float, dict]:
        """Calculate team reward with shaping."""

        # === Primary Reward: Coverage Increase ===
        new_coverage = self._calculate_coverage_sum()
        coverage_increase = new_coverage - self.previous_coverage_sum
        coverage_reward = coverage_increase * self.reward_scale_coverage

        self.previous_coverage_sum = new_coverage

        # === Shaping Rewards ===
        shaping = 0.0

        # 1. Frontier bonus
        frontier_count = 0
        for agent_id in range(self.n_agents):
            if self._is_near_frontier(agent_id, threshold=2):
                frontier_count += 1
        shaping += frontier_count * self.reward_frontier_bonus

        # 2. Spread bonus
        avg_dist = self._average_pairwise_distance()
        target_spread = self.grid_size / 4
        spread_reward = -abs(avg_dist - target_spread) * self.reward_spread_bonus
        shaping += spread_reward

        # 3. Stay penalty
        stay_count = sum(1 for move in executed_moves if move == (0, 0))
        shaping -= stay_count * self.reward_stay_penalty

        # 4. Step penalty
        shaping -= self.reward_step_penalty

        # === Total Reward ===
        shaping *= self.reward_scale_shaping
        total_reward = coverage_reward + shaping

        # Clip to reasonable range
        total_reward = np.clip(total_reward, -10.0, 50.0)

        # Info for logging
        info = {
            'reward_coverage': coverage_reward,
            'reward_shaping': shaping,
            'reward_total': total_reward,
            'coverage_increase': coverage_increase,
        }

        return total_reward, info

    def _get_sensor_info(self, agent_id: int) -> dict:
        """Get information about agent's sensor region."""
        pos = self.agent_positions[agent_id]

        # Count cells in sensor range
        uncovered = 0
        covered = 0
        total = 0

        nearest_frontier_dist = float('inf')
        nearest_frontier_angle = 0.0

        for r in range(max(0, pos[0] - self.sensor_range), min(self.grid_size, pos[0] + self.sensor_range + 1)):
            for c in range(max(0, pos[1] - self.sensor_range), min(self.grid_size, pos[1] + self.sensor_range + 1)):
                dist = math.sqrt((r - pos[0])**2 + (c - pos[1])**2)

                if dist <= self.sensor_range and self.obstacle_grid[r, c] == 0:
                    total += 1

                    if self.coverage_grid[r, c] >= self.coverage_threshold:
                        covered += 1
                    else:
                        uncovered += 1

                        # Track nearest frontier
                        if dist < nearest_frontier_dist:
                            nearest_frontier_dist = dist
                            nearest_frontier_angle = math.atan2(c - pos[1], r - pos[0])

        return {
            'uncovered_ratio': uncovered / max(total, 1),
            'covered_ratio': covered / max(total, 1),
            'frontier_distance': nearest_frontier_dist,
            'frontier_angle': nearest_frontier_angle,
            'coverage_rate': 0.0  # Could track change over time
        }

    def _is_near_frontier(self, agent_id: int, threshold: int = 2) -> bool:
        """Check if agent is near unexplored area."""
        pos = self.agent_positions[agent_id]

        for r in range(max(0, pos[0] - threshold), min(self.grid_size, pos[0] + threshold + 1)):
            for c in range(max(0, pos[1] - threshold), min(self.grid_size, pos[1] + threshold + 1)):
                if self.obstacle_grid[r, c] == 0:
                    if self.coverage_grid[r, c] < self.coverage_threshold:
                        return True

        return False

    def _average_pairwise_distance(self) -> float:
        """Calculate average distance between all agent pairs."""
        if self.n_agents < 2:
            return 0.0

        total_dist = 0.0
        count = 0

        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                pos_i = self.agent_positions[i]
                pos_j = self.agent_positions[j]
                dist = math.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                total_dist += dist
                count += 1

        return total_dist / max(count, 1)


# Register environment with EPyMARL
try:
    from envs import REGISTRY as env_REGISTRY
    env_REGISTRY["coverage"] = CoverageEnvironment
except:
    print("EPyMARL not loaded yet, environment will be registered later")
