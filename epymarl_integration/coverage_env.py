"""
Coverage Environment for EPyMARL
Properly structured for integration with EPyMARL framework.

This file should be placed at: epymarl/src/envs/coverage/coverage_env.py
"""

import numpy as np
import networkx as nx
import math
from typing import List, Dict, Tuple, Optional

# Import MultiAgentEnv (relative for EPyMARL, absolute for standalone)
try:
    from ..multiagentenv import MultiAgentEnv
except (ImportError, ValueError):
    from multiagentenv import MultiAgentEnv


class CoverageEnv(MultiAgentEnv):
    """
    Multi-Agent Coverage Environment for EPyMARL.

    Agents cooperatively explore and cover a 2D grid environment.
    Uses raycasting sensor model for realistic coverage dynamics.
    Compatible with QMIX, VDN, COMA, and other EPyMARL algorithms.
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
        map_type: str = 'empty',
        obstacle_density: float = 0.15,
        reward_scale_coverage: float = 0.5,  # Fixed: was 100.0 (caused gradient explosion)
        reward_scale_shaping: float = 0.1,   # Fixed: was 1.0 (caused gradient explosion)
        reward_frontier_bonus: float = 0.5,
        reward_spread_bonus: float = 0.1,
        reward_stay_penalty: float = 0.2,
        reward_step_penalty: float = 0.1,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize coverage environment.

        Args:
            n_agents: Number of agents
            grid_size: Size of the grid (grid_size × grid_size)
            episode_limit: Maximum steps per episode
            sensor_range: Range of agent sensors
            fov_degrees: Field of view in degrees
            coverage_threshold: PC value to consider cell "covered"
            completion_threshold: Coverage % to end episode
            obs_size: Fixed observation vector size
            map_type: Type of map ('empty', 'rooms', 'random')
            obstacle_density: Density of obstacles for random maps
            reward_scale_coverage: Scaling for coverage reward
            reward_scale_shaping: Scaling for shaping rewards
            reward_frontier_bonus: Bonus for being near frontier
            reward_spread_bonus: Bonus for spreading out
            reward_stay_penalty: Penalty for staying still
            reward_step_penalty: Penalty per step
            seed: Random seed
        """
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

        # Action space: 9 discrete actions (8 directions + stay)
        self.actions = [
            (-1, -1), (-1, 0), (-1, 1),  # NW, N, NE
            (0, -1),  (0, 0),  (0, 1),   # W, Stay, E
            (1, -1),  (1, 0),  (1, 1)    # SW, S, SE
        ]
        self.n_actions = len(self.actions)

        # State variables
        self.steps = 0
        self.agent_positions = []
        self.agent_orientations = []
        self.obstacle_grid = None  # True obstacle map (ground truth)
        self.obstacle_belief = None  # Agent's belief: -1=unknown, 0=free, 1=obstacle
        self.coverage_grid = None
        self.world_graph = None
        self.cumulative_reward = 0.0
        self.previous_coverage_sum = 0.0

        # Random seed
        if seed is not None:
            np.random.seed(seed)
            self._seed = seed

        # Initialize
        self.reset()

    # ==================== EPyMARL Interface Methods ====================

    def reset(self):
        """Reset environment to initial state."""
        self.steps = 0
        self.cumulative_reward = 0.0

        # Generate map
        self.obstacle_grid = self._generate_map(self.map_type)
        self.coverage_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Initialize obstacle belief map (POMDP: all cells unknown initially)
        self.obstacle_belief = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)

        # Build graph of traversable cells
        self._build_world_graph()

        # Initialize agents
        self._initialize_agents()

        # Initial coverage from agent positions
        for agent_id in range(self.n_agents):
            self._raycast_update(agent_id)
            self._update_obstacle_belief(agent_id)  # Reveal obstacles in initial sensor range

        self.previous_coverage_sum = self._calculate_coverage_sum()

        return self.get_obs(), self.get_state()

    def step(self, actions: List[int]):
        """
        Execute one environment step.

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

        # Update coverage and obstacle beliefs
        for agent_id in range(self.n_agents):
            self._raycast_update(agent_id)
            self._update_obstacle_belief(agent_id)  # Update belief after each move

        # Calculate reward
        reward, reward_info = self._calculate_reward(actions, executed_moves)
        self.cumulative_reward += reward

        # Check termination
        coverage_pct = self._calculate_coverage_percentage()
        terminated = (
            self.steps >= self.episode_limit or
            coverage_pct >= self.completion_threshold
        )

        # Info dictionary
        # EPyMARL expects flat numeric values (no nested dicts) for accumulation
        info = {
            **reward_info,
            'coverage_pct': coverage_pct,
            'steps': self.steps,
            'episode_limit': int(terminated and self.steps >= self.episode_limit),
        }

        # Add episode stats when terminated (flat format for EPyMARL compatibility)
        if terminated:
            info['episode_return'] = self.cumulative_reward
            info['episode_length'] = self.steps
            info['episode_coverage'] = coverage_pct

        # EPyMARL expects 5 values: _, reward, terminated, truncated, info (Gymnasium API)
        # First value is placeholder (obs retrieved separately via get_obs())
        # truncated is always False for us (we only use terminated)
        truncated = False
        return None, reward, terminated, truncated, info

    def get_obs(self):
        """Get observations for all agents."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id: int):
        """Get observation for a single agent."""
        obs = []
        pos = self.agent_positions[agent_id]

        # 1. Own State [5]
        obs.extend([
            pos[0] / self.grid_size,
            pos[1] / self.grid_size,
            np.sin(self.agent_orientations[agent_id]),
            np.cos(self.agent_orientations[agent_id]),
            self.steps / self.episode_limit
        ])

        # 2. Sensor Region Info [6]
        sensor_info = self._get_sensor_info(agent_id)
        # Clip frontier distance to prevent inf values from propagating to network
        frontier_dist = min(sensor_info['frontier_distance'], self.grid_size * 2)
        obs.extend([
            sensor_info['uncovered_ratio'],
            sensor_info['covered_ratio'],
            frontier_dist / self.grid_size,
            np.sin(sensor_info['frontier_angle']),
            np.cos(sensor_info['frontier_angle']),
            sensor_info['coverage_rate']
        ])

        # 3. Other Agents [40] (max 10 agents)
        max_other_agents = 10
        for other_id in range(self.n_agents):
            if other_id == agent_id:
                continue

            other_pos = self.agent_positions[other_id]
            dx = (other_pos[0] - pos[0]) / self.grid_size  # row_delta
            dy = (other_pos[1] - pos[1]) / self.grid_size  # col_delta
            dist = np.sqrt(dx**2 + dy**2)
            # Corrected: arctan2(row_delta, col_delta) for consistent angle calculation
            angle = np.arctan2(dx, dy)

            obs.extend([dx, dy, dist, angle])

        # Pad other agents to max
        current_others = self.n_agents - 1
        if current_others < max_other_agents:
            obs.extend([0.0] * (4 * (max_other_agents - current_others)))

        # 4. Global Statistics [3]
        obs.extend([
            self._calculate_coverage_percentage() / 100.0,
            self._calculate_coverage_sum() / (self.grid_size * self.grid_size),
            (self._calculate_coverage_sum() - self.previous_coverage_sum)
        ])

        # Pad to obs_size
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
        POMDP + CTDE: Includes true obstacle map for centralized value function.
        """
        # All agent observations (use partial observability via belief)
        all_obs = np.concatenate([self.get_obs_agent(i) for i in range(self.n_agents)])

        # Global coverage grid
        coverage_flat = self.coverage_grid.flatten()

        # True obstacle map (for CTDE - centralized training can see true state)
        obstacle_flat = self.obstacle_grid.flatten()

        # Metadata
        metadata = np.array([
            self.steps / self.episode_limit,
            self._calculate_coverage_percentage() / 100.0,
            self.n_agents / 10.0,
        ], dtype=np.float32)

        state = np.concatenate([all_obs, coverage_flat, obstacle_flat, metadata])
        return state.astype(np.float32)

    def get_state_size(self):
        """Return state size."""
        all_obs_size = self.obs_size * self.n_agents
        coverage_size = self.grid_size * self.grid_size
        obstacle_size = self.grid_size * self.grid_size  # True obstacle map for CTDE
        metadata_size = 3
        return all_obs_size + coverage_size + obstacle_size + metadata_size

    def get_avail_actions(self):
        """Get available actions for all agents."""
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id: int):
        """Get available actions for a single agent."""
        avail = [0] * self.n_actions
        pos = self.agent_positions[agent_id]

        for i, (dr, dc) in enumerate(self.actions):
            target = (pos[0] + dr, pos[1] + dc)
            if self._is_valid_position(target, agent_id):
                avail[i] = 1

        # Always allow staying
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
        """Render environment (ASCII)."""
        print(f"\nStep {self.steps}/{self.episode_limit}")
        print(f"Coverage: {self._calculate_coverage_percentage():.1f}%")

        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        grid[self.obstacle_grid == 1] = '#'

        covered = self.coverage_grid >= self.coverage_threshold
        grid[covered & (self.obstacle_grid == 0)] = '·'

        for i, pos in enumerate(self.agent_positions):
            grid[pos[0], pos[1]] = str(i)

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
            self._seed = seed

    def save_replay(self):
        """Save replay (not implemented)."""
        pass

    def get_stats(self):
        """Get environment statistics."""
        return {
            'coverage': self._calculate_coverage_percentage(),
            'steps': self.steps,
        }

    # ==================== Internal Methods ====================

    def _generate_map(self, map_type: str):
        """Generate obstacle map."""
        if map_type == 'empty':
            return self._generate_empty_map()
        elif map_type == 'rooms':
            return self._generate_room_map()
        elif map_type == 'random':
            return self._generate_random_map()
        else:
            raise ValueError(f"Unknown map type: {map_type}")

    def _generate_empty_map(self):
        """Generate empty map with borders."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        grid[0, :] = 1.0
        grid[-1, :] = 1.0
        grid[:, 0] = 1.0
        grid[:, -1] = 1.0
        return grid

    def _generate_room_map(self):
        """Generate map with rooms."""
        grid = np.ones((self.grid_size, self.grid_size), dtype=np.float32)

        rooms = [(2, 2, 6, 6), (2, 10, 6, 6), (10, 2, 6, 6), (10, 10, 6, 6)]

        for r, c, h, w in rooms:
            if r + h < self.grid_size and c + w < self.grid_size:
                grid[r:r+h, c:c+w] = 0.0

        grid[4:6, :] = 0.0  # Horizontal corridor
        grid[:, 8:10] = 0.0  # Vertical corridor

        grid[0, :] = 1.0
        grid[-1, :] = 1.0
        grid[:, 0] = 1.0
        grid[:, -1] = 1.0

        return grid

    def _generate_random_map(self):
        """Generate map with random obstacles."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        interior = grid[1:-1, 1:-1]
        interior[:] = (np.random.rand(self.grid_size-2, self.grid_size-2) < self.obstacle_density).astype(np.float32)

        grid[0, :] = 1.0
        grid[-1, :] = 1.0
        grid[:, 0] = 1.0
        grid[:, -1] = 1.0

        return grid

    def _build_world_graph(self):
        """Build NetworkX graph of traversable cells."""
        self.world_graph = nx.Graph()

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.obstacle_grid[r, c] == 0:
                    self.world_graph.add_node((r, c), pc=0.0, type='free')

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) in self.world_graph:
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in self.world_graph:
                            self.world_graph.add_edge((r, c), (nr, nc))

    def _initialize_agents(self):
        """Initialize agent positions and orientations."""
        self.agent_positions = []
        self.agent_orientations = []

        free_positions = list(self.world_graph.nodes())

        if len(free_positions) < self.n_agents:
            raise ValueError(f"Not enough free positions for {self.n_agents} agents")

        corners = [(1, 1), (1, self.grid_size-2), (self.grid_size-2, 1), (self.grid_size-2, self.grid_size-2)]

        placed = []
        for corner in corners:
            if corner in free_positions and len(placed) < self.n_agents:
                placed.append(corner)

        while len(placed) < self.n_agents:
            pos = free_positions[np.random.randint(len(free_positions))]
            if pos not in placed:
                placed.append(pos)

        self.agent_positions = placed[:self.n_agents]
        self.agent_orientations = [np.random.uniform(0, 2*np.pi) for _ in range(self.n_agents)]

    def _is_valid_position(self, pos: Tuple[int, int], agent_id: int):
        """Check if position is valid for agent."""
        r, c = pos

        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return False

        if self.obstacle_grid[r, c] == 1:
            return False

        for other_id, other_pos in enumerate(self.agent_positions):
            if other_id != agent_id and pos == other_pos:
                return False

        return True

    def _resolve_conflicts(self, moves: List[Tuple[int, int]]):
        """Resolve movement conflicts."""
        target_positions = {}

        for agent_id, move in enumerate(moves):
            current = self.agent_positions[agent_id]
            target = (current[0] + move[0], current[1] + move[1])

            if target != current:
                if target not in target_positions:
                    target_positions[target] = []
                target_positions[target].append(agent_id)

        executed = list(moves)
        for target, competing_agents in target_positions.items():
            if len(competing_agents) > 1:
                competing_agents.sort()
                winner = competing_agents[0]
                for loser in competing_agents[1:]:
                    executed[loser] = (0, 0)

        return executed

    def _move_agent(self, agent_id: int, move: Tuple[int, int]):
        """Move agent and update orientation."""
        current = self.agent_positions[agent_id]
        target = (current[0] + move[0], current[1] + move[1])

        if self._is_valid_position(target, agent_id):
            self.agent_positions[agent_id] = target
            if move != (0, 0):
                # Corrected: atan2(row_delta, col_delta) for proper orientation
                self.agent_orientations[agent_id] = math.atan2(move[0], move[1])

    def _raycast_update(self, agent_id: int):
        """Update coverage using raycasting."""
        pos = self.agent_positions[agent_id]
        orientation = self.agent_orientations[agent_id]

        center = np.array(pos, dtype=float)

        if self.world_graph.has_node(pos):
            self.coverage_grid[pos[0], pos[1]] = 1.0
            self.world_graph.nodes[pos]['pc'] = 1.0

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
                # Corrected: [row, col] = [sin, cos] for proper angle-to-grid mapping
                offset = np.array([math.sin(angle), math.cos(angle)]) * ray_dist
                ray_pos = center + offset
                ray_cell = tuple(np.round(ray_pos).astype(int))

                if not (0 <= ray_cell[0] < self.grid_size and 0 <= ray_cell[1] < self.grid_size):
                    break

                if self.obstacle_grid[ray_cell[0], ray_cell[1]] == 1:
                    break

                if self.world_graph.has_node(ray_cell):
                    if ray_dist <= 1.0:
                        new_pc = 1.0
                    else:
                        r0 = max_range / 2.5
                        k = 2.0
                        new_pc = np.clip(1.0 / (1.0 + np.exp(k * (ray_dist - r0))), 0.0, 1.0)

                    current_pc = self.coverage_grid[ray_cell[0], ray_cell[1]]
                    if new_pc > current_pc:
                        self.coverage_grid[ray_cell[0], ray_cell[1]] = new_pc
                        self.world_graph.nodes[ray_cell]['pc'] = new_pc

    def _update_obstacle_belief(self, agent_id: int):
        """
        Update obstacle belief map based on what agent can see.
        Reveals obstacles and free cells within sensor range.
        """
        pos = self.agent_positions[agent_id]

        # Reveal all cells within sensor range
        for r in range(max(0, pos[0] - self.sensor_range),
                      min(self.grid_size, pos[0] + self.sensor_range + 1)):
            for c in range(max(0, pos[1] - self.sensor_range),
                          min(self.grid_size, pos[1] + self.sensor_range + 1)):
                dist = math.sqrt((r - pos[0])**2 + (c - pos[1])**2)

                if dist <= self.sensor_range:
                    # Reveal the true state of this cell
                    self.obstacle_belief[r, c] = self.obstacle_grid[r, c]

    def _calculate_coverage_sum(self):
        """Calculate sum of coverage probabilities."""
        return float(np.sum(self.coverage_grid[self.obstacle_grid == 0]))

    def _calculate_coverage_percentage(self):
        """Calculate percentage of cells above threshold."""
        free_cells = self.obstacle_grid == 0
        covered_cells = (self.coverage_grid >= self.coverage_threshold) & free_cells
        total_free = np.sum(free_cells)

        if total_free == 0:
            return 100.0

        return (np.sum(covered_cells) / total_free) * 100.0

    def _calculate_reward(self, actions, executed_moves):
        """Calculate team reward with shaping."""
        new_coverage = self._calculate_coverage_sum()
        coverage_increase = new_coverage - self.previous_coverage_sum
        coverage_reward = coverage_increase * self.reward_scale_coverage

        self.previous_coverage_sum = new_coverage

        shaping = 0.0

        # Frontier bonus
        frontier_count = 0
        for agent_id in range(self.n_agents):
            if self._is_near_frontier(agent_id, threshold=2):
                frontier_count += 1
        shaping += frontier_count * self.reward_frontier_bonus

        # Spread bonus
        avg_dist = self._average_pairwise_distance()
        target_spread = self.grid_size / 4
        spread_reward = -abs(avg_dist - target_spread) * self.reward_spread_bonus
        shaping += spread_reward

        # Stay penalty
        stay_count = sum(1 for move in executed_moves if move == (0, 0))
        shaping -= stay_count * self.reward_stay_penalty

        # Step penalty
        shaping -= self.reward_step_penalty

        shaping *= self.reward_scale_shaping
        total_reward = coverage_reward + shaping
        total_reward = np.clip(total_reward, -10.0, 50.0)

        info = {
            'reward_coverage': coverage_reward,
            'reward_shaping': shaping,
            'reward_total': total_reward,
            'coverage_increase': coverage_increase,
        }

        return total_reward, info

    def _get_sensor_info(self, agent_id: int):
        """
        Get information about agent's sensor region.
        POMDP: Uses obstacle_belief instead of true obstacle_grid.
        Only considers cells with known state (belief != -1).
        """
        pos = self.agent_positions[agent_id]

        uncovered = 0
        covered = 0
        total = 0

        nearest_frontier_dist = float('inf')
        nearest_frontier_angle = 0.0

        for r in range(max(0, pos[0] - self.sensor_range), min(self.grid_size, pos[0] + self.sensor_range + 1)):
            for c in range(max(0, pos[1] - self.sensor_range), min(self.grid_size, pos[1] + self.sensor_range + 1)):
                dist = math.sqrt((r - pos[0])**2 + (c - pos[1])**2)

                # POMDP: Only count cells that are known to be free (belief == 0)
                if dist <= self.sensor_range and self.obstacle_belief[r, c] == 0:
                    total += 1

                    if self.coverage_grid[r, c] >= self.coverage_threshold:
                        covered += 1
                    else:
                        uncovered += 1

                        if dist < nearest_frontier_dist:
                            nearest_frontier_dist = dist
                            # Corrected: atan2(row_delta, col_delta) for consistent angle calculation
                            nearest_frontier_angle = math.atan2(r - pos[0], c - pos[1])

        return {
            'uncovered_ratio': uncovered / max(total, 1),
            'covered_ratio': covered / max(total, 1),
            'frontier_distance': nearest_frontier_dist,
            'frontier_angle': nearest_frontier_angle,
            'coverage_rate': 0.0
        }

    def _is_near_frontier(self, agent_id: int, threshold: int = 2):
        """Check if agent is near unexplored area."""
        pos = self.agent_positions[agent_id]

        for r in range(max(0, pos[0] - threshold), min(self.grid_size, pos[0] + threshold + 1)):
            for c in range(max(0, pos[1] - threshold), min(self.grid_size, pos[1] + threshold + 1)):
                if self.obstacle_grid[r, c] == 0:
                    if self.coverage_grid[r, c] < self.coverage_threshold:
                        return True

        return False

    def _average_pairwise_distance(self):
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
