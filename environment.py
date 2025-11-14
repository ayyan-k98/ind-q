"""
Multi-Agent RL Coverage Environment
Contains the environment logic for multi-agent coverage planning.
"""

import os
import time
import math
import random
import numpy as np
import networkx as nx
import torch
from typing import Tuple, List, Dict, Set, Optional, Any
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from data_structures import RobotState, WorldState, CoverageMetrics, Room
from agent import MARLCoverageAgent
from utils import ensure_dir, get_latest_model_episode
from visualization import (
    visualize_coverage,
    visualize_agent_local_map,
    visualize_evaluation_timesteps,
    visualize_learning_metrics
)


class MARLCoverageEnvironment:
    """Multi-Agent Reinforcement Learning environment for coverage planning."""

    def __init__(
        self,
        grid_size: int = 20,
        num_agents: int = 4,
        sensor_range: int = 5,
        comm_range: float = 7.0,
        coverage_threshold: float = 0.8,
        completion_threshold_perc: float = 95.0,
        max_episodes: int = 300,
        max_steps_per_episode: int = 100,
        gamma_coverage: float = 15.0,
        step_penalty: float = -0.02,
        orientation_cost_factor: float = 0.02,
        invalid_move_penalty: float = -0.5,
        fov_degrees: float = 120.0,
        device: str = "cpu",
        use_dueling: bool = True,
        agent_config: dict = {},
        tensorboard_dir: str = "./runs/marl_coverage",
        num_rooms_range: Tuple[int, int] = (6, 12),
        room_size_range: Tuple[int, int] = (4, 8),
        cave_fill_prob: float = 0.48,
        cave_smoothing_iterations: int = 5,
        cave_birth_limit: int = 4,
        cave_death_limit: int = 3,
        random_obstacle_density: float = 0.15
    ):
        """
        Initialize MARL Coverage Environment.

        Args:
            grid_size: Size of the grid
            num_agents: Number of agents
            sensor_range: Range of agent sensors
            comm_range: Communication range between agents
            coverage_threshold: Threshold for marking a cell as covered
            completion_threshold_perc: Coverage percentage to end episode
            max_episodes: Maximum training episodes
            max_steps_per_episode: Maximum steps per episode
            gamma_coverage: Reward weight for coverage
            step_penalty: Penalty per step
            orientation_cost_factor: Penalty for orientation change
            invalid_move_penalty: Penalty for invalid moves
            fov_degrees: Field of view in degrees
            device: Device to use (cpu or cuda)
            use_dueling: Whether to use Dueling DQN
            agent_config: Configuration dictionary for agents
            tensorboard_dir: Directory for TensorBoard logs
            num_rooms_range: Range for number of rooms in room map
            room_size_range: Range for room sizes
            cave_fill_prob: Fill probability for cave generation
            cave_smoothing_iterations: Smoothing iterations for cave
            cave_birth_limit: Birth limit for cave cellular automata
            cave_death_limit: Death limit for cave cellular automata
            random_obstacle_density: Density of obstacles in random map
        """
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.sensor_range = sensor_range
        self.comm_range = comm_range
        self.coverage_threshold = coverage_threshold
        self.completion_threshold_perc = completion_threshold_perc
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode

        # Reward components
        self.gamma_coverage = gamma_coverage
        self.step_penalty = step_penalty
        self.orientation_cost_factor = orientation_cost_factor
        self.invalid_move_penalty = invalid_move_penalty

        self.fov_radians = math.radians(fov_degrees)
        self.device = torch.device(device)
        self.use_dueling = use_dueling
        self.agent_config = agent_config

        # Map generation parameters
        self.num_rooms_range = num_rooms_range
        self.room_size_range = room_size_range
        self.cave_fill_prob = cave_fill_prob
        self.cave_smoothing_iterations = cave_smoothing_iterations
        self.cave_birth_limit = cave_birth_limit
        self.cave_death_limit = cave_death_limit
        self.random_obstacle_density = random_obstacle_density

        # Tensorboard Writer
        self.writer = None
        if tensorboard_dir:
            ensure_dir(tensorboard_dir)
            try:
                self.writer = SummaryWriter(tensorboard_dir)
            except Exception as e:
                print(f"Warning: TensorBoard setup failed: {e}")

        # Map generation setup
        self.training_map_generators = {
            'empty': self._generate_empty_map,
            'cave': self._generate_cave_map,
            'room': self._generate_room_map,
            'random': self._generate_random_map
        }
        self.training_map_order = ['room', 'cave', 'random']
        self.current_training_map_index = 0
        self.mode = 'train'

        # Environment State
        self.obstacle_grid: Optional[np.ndarray] = None
        self.world_state: Optional[WorldState] = None
        self.agent_positions: Dict[int, Tuple[int, int]] = {}
        self.agents: List[MARLCoverageAgent] = []

        # Metrics & Time
        self.metrics = CoverageMetrics()
        self.env_time = 0.0
        self.previous_global_coverage = 0.0
        self.current_global_coverage = 0.0
        self.global_coverage_increase = 0.0
        self.current_rewards = {}

        # Caching
        self.raycasting_cache = {}

    # --- Map Generation Methods ---

    def _generate_room_map(self) -> np.ndarray:
        """Generate a map with rooms connected by corridors."""
        grid = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        rooms = []
        num_rooms_target = random.randint(*self.num_rooms_range)
        max_tries = num_rooms_target * 5

        for _ in range(max_tries):
            if len(rooms) >= num_rooms_target:
                break

            h = random.randint(*self.room_size_range)
            w = random.randint(*self.room_size_range)

            if h >= self.grid_size - 2 or w >= self.grid_size - 2:
                continue

            r = random.randint(1, self.grid_size - h - 2)
            c = random.randint(1, self.grid_size - w - 2)
            new_room = Room(r, c, h, w)

            intersects = False
            for other_room in rooms:
                buffered_other = Room(
                    other_room.r1 - 1, other_room.c1 - 1,
                    other_room.height + 2, other_room.width + 2
                )
                if new_room.intersects(buffered_other):
                    intersects = True
                    break

            if not intersects:
                grid[new_room.r1:new_room.r2, new_room.c1:new_room.c2] = 0.0

                if rooms:  # Connect to the previous room
                    prev_room = rooms[-1]
                    pr, pc = prev_room.center
                    cr, cc = new_room.center
                    min_r, max_r = min(pr, cr), max(pr, cr)
                    min_c, max_c = min(pc, cc), max(pc, cc)
                    safe_pr = np.clip(pr, 0, self.grid_size - 1)
                    safe_cc = np.clip(cc, 0, self.grid_size - 1)
                    safe_pc = np.clip(pc, 0, self.grid_size - 1)
                    safe_cr = np.clip(cr, 0, self.grid_size - 1)

                    if random.random() < 0.5:  # H then V
                        grid[safe_pr, min_c:max_c + 1] = 0.0
                        grid[min_r:max_r + 1, safe_cc] = 0.0
                    else:  # V then H
                        grid[min_r:max_r + 1, safe_pc] = 0.0
                        grid[safe_cr, min_c:max_c + 1] = 0.0

                rooms.append(new_room)

        if not rooms:  # Ensure map isn't fully blocked
            center_r, center_c = self.grid_size // 2, self.grid_size // 2
            size = max(1, self.grid_size // 10)
            r1, c1 = max(0, center_r - size // 2), max(0, center_c - size // 2)
            r2, c2 = min(self.grid_size, center_r + (size + 1) // 2), min(self.grid_size, center_c + (size + 1) // 2)
            grid[r1:r2, c1:c2] = 0.0

        # Borders
        grid[0, :] = 1.0
        grid[-1, :] = 1.0
        grid[:, 0] = 1.0
        grid[:, -1] = 1.0

        return grid

    def _generate_cave_map(self) -> np.ndarray:
        """Generate a cave-like map using cellular automata."""
        width, height = self.grid_size, self.grid_size
        grid = np.ones((height, width), dtype=np.float32)
        inner_fill = (np.random.rand(height - 2, width - 2) < self.cave_fill_prob).astype(np.float32)
        grid[1:-1, 1:-1] = inner_fill

        def count_walls(x, y, current_grid):
            sub_grid = current_grid[max(0, y-1):min(height, y+2), max(0, x-1):min(width, x+2)]
            return np.sum(sub_grid) - current_grid[y, x]

        for _ in range(self.cave_smoothing_iterations):
            new_grid = grid.copy()
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    walls = count_walls(x, y, grid)
                    if grid[y, x] == 1:  # If wall
                        if walls < self.cave_death_limit:
                            new_grid[y, x] = 0  # Dies
                    elif walls > self.cave_birth_limit:  # If floor
                        new_grid[y, x] = 1  # Born
            grid = new_grid

        # Borders
        grid[0, :] = 1.0
        grid[-1, :] = 1.0
        grid[:, 0] = 1.0
        grid[:, -1] = 1.0

        return grid

    def _generate_empty_map(self) -> np.ndarray:
        """Generate an empty map with only borders."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        grid[0, :] = 1.0
        grid[-1, :] = 1.0
        grid[:, 0] = 1.0
        grid[:, -1] = 1.0
        return grid

    def _generate_random_map(self) -> np.ndarray:
        """Generate a map with random obstacles."""
        grid = self._generate_empty_map()
        inner_area_size = (self.grid_size - 2) * (self.grid_size - 2)

        if inner_area_size <= 0:
            return grid

        num_obstacles = min(int(self.random_obstacle_density * inner_area_size), inner_area_size)
        if num_obstacles > 0:
            inner_indices = np.random.choice(inner_area_size, num_obstacles, replace=False)
            rows, cols = np.unravel_index(inner_indices, (self.grid_size - 2, self.grid_size - 2))
            grid[rows + 1, cols + 1] = 1.0

        return grid

    # --- Core Environment Logic ---

    def _initialize_world_state(self) -> Optional[WorldState]:
        """Initialize graph and robot states based on obstacle grid."""
        if self.obstacle_grid is None:
            return None

        G = nx.grid_2d_graph(self.grid_size, self.grid_size)
        nodes_to_remove = []

        for node in G.nodes():
            r, c = node
            if self.obstacle_grid[r, c] == 1.0:
                nodes_to_remove.append(node)
            else:
                G.nodes[node]['type'] = 'free'
                G.nodes[node]['pc'] = 0.0

        G.remove_nodes_from(nodes_to_remove)

        robots = []
        placed_positions = set()
        free_positions = list(G.nodes())

        if not free_positions:
            print("Error: No free positions!")
            return None

        for i in range(self.num_agents):
            pos = None
            corner_prefs = [
                (1, 1), (1, self.grid_size - 2),
                (self.grid_size - 2, 1), (self.grid_size - 2, self.grid_size - 2)
            ]
            random.shuffle(corner_prefs)

            for pref_pos in corner_prefs:
                if G.has_node(pref_pos) and pref_pos not in placed_positions:
                    pos = pref_pos
                    break

            if pos is None:
                available_free = [p for p in free_positions if p not in placed_positions]
                if not available_free:
                    pos = random.choice(free_positions)
                else:
                    pos = random.choice(available_free)

            placed_positions.add(pos)
            initial_orientation = random.uniform(0, 2 * math.pi)
            robots.append(RobotState(position=pos, orientation=initial_orientation))

        return WorldState(graph=G, robots=robots)

    def _create_agents(self) -> List[MARLCoverageAgent]:
        """Create agent instances using current world state."""
        if not self.world_state or not self.world_state.robots:
            return []

        agents = []
        for i, robot_state in enumerate(self.world_state.robots):
            if i >= self.num_agents:
                break

            agent = MARLCoverageAgent(
                agent_id=i,
                initial_state=robot_state,
                environment=self,
                grid_size=self.grid_size,
                sensor_range=self.sensor_range,
                comm_range=self.comm_range,
                use_dueling=self.use_dueling,
                device=self.device,
                **self.agent_config
            )
            agents.append(agent)

        return agents

    def reset(self, full_reset=False, mode='train'):
        """Reset environment to initial state."""
        self.mode = mode
        current_time_seed = int(time.time() * 1000) % (2**32)
        np.random.seed(current_time_seed)
        random.seed(current_time_seed)

        map_type = 'random'  # Default for eval
        if self.mode == 'train':
            map_type = self.training_map_order[self.current_training_map_index]

        map_generator = self.training_map_generators[map_type]
        self.obstacle_grid = map_generator()

        self.world_state = self._initialize_world_state()
        if not self.world_state:
            raise RuntimeError("Failed to initialize world state.")

        self.agent_positions = {i: r.position for i, r in enumerate(self.world_state.robots)}

        if not self.agents or full_reset:
            self.agents = self._create_agents()
            if not self.agents:
                raise RuntimeError("Failed to create agents.")

        for i, agent in enumerate(self.agents):
            if i < len(self.world_state.robots):
                agent.update_state(self.world_state.robots[i])
                agent._initialize_local_map()
                agent.known_positions = {}
                agent.communication_history = []
            else:
                print(f"Warning: Agent index {i} out of bounds for world state robots.")

        self.metrics = CoverageMetrics()
        self.env_time = 0.0
        self.previous_global_coverage = 0.0

        for r_state in self.world_state.robots:
            if self.world_state.graph.has_node(r_state.position):
                self.world_state.graph.nodes[r_state.position]['pc'] = 1.0
                self.world_state.graph.nodes[r_state.position]['type'] = 'covered'
                self.previous_global_coverage += 1.0

        self.current_global_coverage = self.previous_global_coverage
        self.global_coverage_increase = 0.0
        self.current_rewards = {i: 0.0 for i in range(self.num_agents)}
        self.raycasting_cache = {}

        initial_states = {i: agent.get_state_tensor() for i, agent in enumerate(self.agents)}
        return initial_states

    def is_in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid boundaries."""
        r, c = pos
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def is_valid_position(self, pos: Tuple[int, int], agent_id: int, current_pos: Tuple[int, int]) -> bool:
        """Check if a target position is valid for movement."""
        if not (0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size):
            return False

        if self.obstacle_grid[pos[0], pos[1]] == 1.0:
            return False

        if not self.world_state or not self.world_state.graph.has_node(pos):
            return False

        for other_id, other_pos in self.agent_positions.items():
            if other_id != agent_id and pos == other_pos:
                return False

        dx = pos[0] - current_pos[0]
        dy = pos[1] - current_pos[1]

        if abs(dx) == 1 and abs(dy) == 1:  # Diagonal move
            adj1 = (current_pos[0] + dx, current_pos[1])
            adj2 = (current_pos[0], current_pos[1] + dy)
            adj1_obs = (not (0 <= adj1[0] < self.grid_size and 0 <= adj1[1] < self.grid_size)) or (self.obstacle_grid[adj1[0], adj1[1]] == 1.0)
            adj2_obs = (not (0 <= adj2[0] < self.grid_size and 0 <= adj2[1] < self.grid_size)) or (self.obstacle_grid[adj2[0], adj2[1]] == 1.0)
            if adj1_obs and adj2_obs:
                return False

        return True

    def get_valid_actions(self, agent_id: int) -> List[Tuple[int, int]]:
        """Get valid actions for an agent."""
        valid_actions = []

        if agent_id not in self.agent_positions or agent_id >= len(self.agents):
            return [(0, 0)]

        current_pos = self.agent_positions[agent_id]
        possible_actions = self.agents[agent_id].actions

        for action in possible_actions:
            target_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
            if self.is_valid_position(target_pos, agent_id, current_pos):
                valid_actions.append(action)

        if not valid_actions:
            return [(0, 0)]

        return valid_actions

    def perform_action(self, agent_id: int, action: Tuple[int, int]) -> Tuple[RobotState, float, bool]:
        """Execute an action for a single agent."""
        if agent_id >= len(self.agents) or agent_id not in self.agent_positions:
            return RobotState((0, 0), 0.0), -1.0, False

        current_robot_state = self.agents[agent_id].state
        old_pos = current_robot_state.position
        old_orientation = current_robot_state.orientation
        target_pos = (old_pos[0] + action[0], old_pos[1] + action[1])

        is_move_valid = self.is_valid_position(target_pos, agent_id, old_pos)

        new_pos = old_pos
        new_orientation = old_orientation
        reward = 0.0
        coverage_increase_area = 0.0

        if is_move_valid:
            new_pos = target_pos
            if action != (0, 0):
                new_orientation = math.atan2(action[1], action[0])

            new_robot_state = RobotState(position=new_pos, orientation=new_orientation)
            self.agents[agent_id].update_state(new_robot_state)
            self.agent_positions[agent_id] = new_pos

            coverage_before = self.calculate_coverage_area()
            self._update_coverage(agent_id)
            coverage_after = self.calculate_coverage_area()
            coverage_increase_area = max(0, coverage_after - coverage_before)

            reward_coverage = self.gamma_coverage * coverage_increase_area
            orientation_change = min(
                abs(new_orientation - old_orientation),
                2 * math.pi - abs(new_orientation - old_orientation)
            )
            orientation_penalty = self.orientation_cost_factor * (orientation_change / math.pi)
            reward_step = self.step_penalty
            reward = reward_coverage - orientation_penalty + reward_step
        else:
            new_robot_state = current_robot_state
            reward = self.invalid_move_penalty

        done = self.check_coverage_complete()
        return new_robot_state, reward, done

    def _update_coverage(self, agent_id: int):
        """Trigger coverage update for the agent and update local map."""
        if not (0 <= agent_id < len(self.agents)):
            print(f"Warning: Invalid agent_id {agent_id} in _update_coverage.")
            return

        agent = self.agents[agent_id]
        agent_state = agent.state

        updated_nodes = self.raycast_coverage_update(agent_state)

        if updated_nodes:
            for node in updated_nodes:
                if self.world_state.graph.has_node(node):
                    data = self.world_state.graph.nodes[node]
                    pc = data.get('pc', 0.0)
                    ntype = data.get('type', 'free')
                    agent.update_local_map(node, pc, ntype)

    def raycast_coverage_update(self, agent_state: RobotState) -> Set[Tuple[int, int]]:
        """Update coverage based on agent's sensor FOV."""
        updated_nodes = set()
        center = np.array(agent_state.position, dtype=float)
        curr_cell_rounded = tuple(np.round(center).astype(int))

        if self.world_state.graph.has_node(curr_cell_rounded):
            self.world_state.graph.nodes[curr_cell_rounded]['pc'] = 1.0
            if 1.0 >= self.coverage_threshold:
                self.world_state.graph.nodes[curr_cell_rounded]['type'] = 'covered'
            updated_nodes.add(curr_cell_rounded)

        immediate_distance_threshold = 1.0
        half_fov = self.fov_radians / 2
        agent_orientation = agent_state.orientation % (2 * math.pi)
        start_angle = agent_orientation - half_fov
        end_angle = agent_orientation + half_fov
        step_size = 0.5
        max_range = float(self.sensor_range)
        num_rays = 32

        for i in range(num_rays):
            fraction = 0.5 if num_rays == 1 else i / (num_rays - 1)
            angle = start_angle + fraction * (end_angle - start_angle)
            ray_dist = 0.0

            while ray_dist <= max_range:
                ray_dist += step_size
                offset = np.array([math.cos(angle), math.sin(angle)]) * ray_dist
                ray_pos = center + offset
                pos_rounded = tuple(np.round(ray_pos).astype(int))

                if not self.is_in_bounds(pos_rounded):
                    break

                if self.obstacle_grid[pos_rounded[0], pos_rounded[1]] == 1.0:
                    break

                if self.world_state.graph.has_node(pos_rounded):
                    if ray_dist <= immediate_distance_threshold:
                        new_pc = 1.0
                    else:
                        r0 = max_range / 2.5
                        k = 2.0
                        new_pc = np.clip(1.0 / (1.0 + np.exp(k * (ray_dist - r0))), 0.0, 1.0)

                    current_pc = self.world_state.graph.nodes[pos_rounded].get('pc', 0.0)
                    if new_pc > current_pc:
                        self.world_state.graph.nodes[pos_rounded]['pc'] = new_pc
                        updated_nodes.add(pos_rounded)
                        if new_pc >= self.coverage_threshold:
                            self.world_state.graph.nodes[pos_rounded]['type'] = 'covered'
                else:
                    break

        return updated_nodes

    def calculate_coverage_area(self) -> float:
        """Calculate sum of coverage values over all nodes."""
        if not self.world_state or not self.world_state.graph:
            return 0.0
        return sum(data.get('pc', 0.0) for node, data in self.world_state.graph.nodes(data=True))

    def calculate_coverage_percentage(self) -> float:
        """Calculate percentage of covered nodes."""
        if not self.world_state or not self.world_state.graph:
            return 0.0

        total_traversable_nodes = self.world_state.graph.number_of_nodes()
        if total_traversable_nodes == 0:
            return 100.0

        covered_count = sum(
            1 for node, data in self.world_state.graph.nodes(data=True)
            if data.get('type') == 'covered' or data.get('pc', 0.0) >= self.coverage_threshold
        )
        return (covered_count / total_traversable_nodes) * 100.0

    def check_coverage_complete(self) -> bool:
        """Check if coverage percentage meets completion threshold."""
        return self.calculate_coverage_percentage() >= self.completion_threshold_perc

    def _resolve_conflicts(self, actions: Dict[int, Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        """Resolve conflicts using ID-based priority."""
        resolved_actions = actions.copy()
        target_positions = defaultdict(list)

        for agent_id, action in actions.items():
            if agent_id in self.agent_positions:
                current_pos = self.agent_positions[agent_id]
                target_pos = (current_pos[0] + action[0], current_pos[1] + action[1])

                if target_pos != current_pos:
                    if self.is_valid_position(target_pos, agent_id, current_pos):
                        target_positions[target_pos].append(agent_id)
                    else:
                        resolved_actions[agent_id] = (0, 0)

        for target_pos, competing_agents in target_positions.items():
            if len(competing_agents) > 1:
                competing_agents.sort()
                winner_agent_id = competing_agents[0]
                for agent_id in competing_agents[1:]:
                    resolved_actions[agent_id] = (0, 0)

        return resolved_actions

    def execute_communications(self):
        """Execute communication between agents within range."""
        comm_events_count = 0
        agents_communicated_this_step = set()

        if len(self.agents) < 2:
            return 0

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if i < len(self.agents) and j < len(self.agents):
                    agent_i, agent_j = self.agents[i], self.agents[j]
                    if agent_i.communicate(agent_j, self.env_time):
                        pair = tuple(sorted((i, j)))
                        if pair not in agents_communicated_this_step:
                            comm_events_count += 1
                            agents_communicated_this_step.add(pair)

        self.metrics.communication_events.append(comm_events_count)
        return comm_events_count

    def step(self, actions: Dict[int, Tuple[int, int]]) -> Dict[int, Tuple]:
        """Execute one environment step."""
        step_start_time = time.time()
        self.previous_global_coverage = self.calculate_coverage_area()
        self.current_rewards = {i: 0.0 for i in range(self.num_agents)}

        for agent_id in range(self.num_agents):
            if agent_id not in actions:
                actions[agent_id] = (0, 0)

        resolved_actions = self._resolve_conflicts(actions)
        results = {}
        episode_done_flag = False

        for agent_id, action in resolved_actions.items():
            if agent_id < len(self.agents):
                new_robot_state, reward, done = self.perform_action(agent_id, action)
                next_state_tensor = self.agents[agent_id].get_state_tensor()
                results[agent_id] = (next_state_tensor, reward, done, {})
                self.current_rewards[agent_id] = reward
                self.metrics.agent_rewards[agent_id].append(reward)
                if done:
                    episode_done_flag = True
            else:
                dummy_state = (
                    torch.zeros(2, self.grid_size, self.grid_size, device=self.device),
                    torch.zeros(2, device=self.device)
                )
                results[agent_id] = (dummy_state, 0.0, False, {'error': 'invalid agent_id'})

        self.execute_communications()

        self.current_global_coverage = self.calculate_coverage_area()
        self.global_coverage_increase = self.current_global_coverage - self.previous_global_coverage
        self.metrics.total_coverage.append(self.current_global_coverage)
        self.metrics.coverage_rate.append(self.global_coverage_increase)

        step_time = time.time() - step_start_time
        self.metrics.execution_times.append(step_time)
        self.env_time += step_time

        final_results = {
            aid: (res[0], res[1], episode_done_flag, res[3])
            for aid, res in results.items()
        }
        return final_results

    def grid_to_matrix(self, graph: Optional[nx.Graph] = None) -> np.ndarray:
        """Convert graph coverage to matrix for visualization."""
        target_graph = graph if graph is not None else self.world_state.graph
        matrix = np.full((self.grid_size, self.grid_size), -1.0, dtype=np.float32)

        if target_graph:
            for node, data in target_graph.nodes(data=True):
                r, c = node
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    matrix[r, c] = data.get('pc', 0.0)

        return matrix

    # --- Training Method ---

    def train(self, episodes_per_type: int = 25, verbose=True, save_interval=50, model_dir="./models"):
        """Train agents, cycling through map types."""
        print("\n--- Starting Training ---")
        ensure_dir(model_dir)
        if self.writer:
            ensure_dir(self.writer.log_dir)

        if episodes_per_type <= 0:
            episodes_per_type = 1

        num_map_types = len(self.training_map_order)
        if num_map_types == 0:
            raise ValueError("Training map order is empty.")

        initial_map_idx = 0
        self.current_training_map_index = initial_map_idx
        states = self.reset(full_reset=True, mode='train')

        if not self.agents:
            raise RuntimeError("Agents not created after initial reset.")

        print(f"Training start. Initial map type: {self.training_map_order[initial_map_idx]}")

        total_steps_across_episodes = 0
        global_optimization_step = 0

        for episode in range(self.max_episodes):
            map_type_index = (episode // episodes_per_type) % num_map_types
            current_map_type = self.training_map_order[map_type_index]

            if self.current_training_map_index != map_type_index:
                if verbose:
                    print(f"\n--- Switching to Map Type: {current_map_type} (Episode {episode}) ---")
                self.current_training_map_index = map_type_index

            states = self.reset(full_reset=False, mode='train')

            episode_trajectories = {i: [self.agents[i].state.position] for i in range(self.num_agents)}
            episode_rewards_sum = {i: 0 for i in range(self.num_agents)}
            episode_steps = 0
            episode_done = False
            ep_start_time = time.time()

            if verbose and episode % 10 == 0:
                print(f"--- Ep {episode}/{self.max_episodes} (Map: {current_map_type}, Eps: {self.agents[0].epsilon:.3f}) ---")

            for step in range(self.max_steps_per_episode):
                actions = {
                    i: agent.select_action(states[i], self.get_valid_actions(i))
                    for i, agent in enumerate(self.agents) if i in states
                }
                results = self.step(actions)

                for agent_id, result_tuple in results.items():
                    if agent_id < len(self.agents):
                        agent = self.agents[agent_id]
                        next_state, reward, done, _ = result_tuple

                        if agent_id in actions and agent_id in states:
                            agent.memory.push(states[agent_id], actions[agent_id], next_state, reward, done)
                            loss = agent.optimize_model()

                            if loss is not None:
                                self.metrics.agent_losses[agent_id].append(loss)
                                if self.writer:
                                    self.writer.add_scalar(f'Loss/Agent_{agent_id}', loss, global_optimization_step)
                                global_optimization_step += 1

                            agent.update_target_network()

                        states[agent_id] = next_state
                        episode_rewards_sum[agent_id] += reward
                        episode_trajectories[agent_id].append(agent.state.position)

                        if done:
                            episode_done = True

                current_total_step = total_steps_across_episodes + episode_steps

                if self.writer:
                    self.writer.add_scalar('Perf/Coverage_Area', self.current_global_coverage, current_total_step)
                    self.writer.add_scalar('Perf/Coverage_Increase', self.global_coverage_increase, current_total_step)
                    if self.current_rewards:
                        self.writer.add_scalar('Perf/Avg_Step_Reward', np.mean(list(self.current_rewards.values())), current_total_step)
                    if self.metrics.communication_events:
                        self.writer.add_scalar('Perf/Comm_Events', self.metrics.communication_events[-1], current_total_step)

                episode_steps += 1
                total_steps_across_episodes += 1

                if episode_done or episode_steps >= self.max_steps_per_episode:
                    break

            # End of Episode
            final_coverage_perc = self.calculate_coverage_percentage()
            avg_ep_reward = np.mean(list(episode_rewards_sum.values())) if episode_rewards_sum else 0.0
            ep_duration = time.time() - ep_start_time

            for agent_id, agent in enumerate(self.agents):
                agent.update_epsilon()
                if self.writer:
                    self.writer.add_scalar(f'Reward/Agent_{agent_id}_EpSum', episode_rewards_sum[agent_id], episode)
                    self.writer.add_scalar(f'Params/Agent_{agent_id}_Epsilon', agent.epsilon, episode)
                self.metrics.epsilon_values[agent_id].append(agent.epsilon)

            if self.writer:
                self.writer.add_scalar('Episode/Steps', episode_steps, episode)
                self.writer.add_scalar('Episode/Coverage_Percent', final_coverage_perc, episode)
                self.writer.add_scalar('Episode/Avg_Reward', avg_ep_reward, episode)
                self.writer.add_scalar('Episode/Duration_Sec', ep_duration, episode)

            if episode > 0 and (episode % save_interval == 0 or episode == self.max_episodes - 1):
                print(f"--- Saving models at episode {episode} ---")
                for i, agent in enumerate(self.agents):
                    agent.save_model(os.path.join(model_dir, f"agent_{i}_ep{episode}.pt"))
                print(f"--- Models saved to {model_dir} ---")

            if verbose and (episode % 50 == 0 or episode == self.max_episodes - 1):
                print(f"Visualizing episode {episode}...")
                try:
                    coverage_matrix = self.grid_to_matrix()
                    visualize_coverage(
                        coverage_matrix,
                        [agent.state for agent in self.agents],
                        episode_trajectories,
                        self.grid_size,
                        self.num_agents,
                        final_coverage_perc,
                        episode=f"{episode} ({current_map_type})",
                        title_suffix=" (Train Final)"
                    )
                    if episode == self.max_episodes - 1:
                        visualize_learning_metrics(self.metrics, self.num_agents)
                except Exception as e:
                    print(f"   Visualization error: {e}")

            if verbose and episode % 10 == 0:
                print(f"   Ep {episode} Summary: Steps={episode_steps}, Cov={final_coverage_perc:.2f}%, AvgRew={avg_ep_reward:.3f}, Time={ep_duration:.2f}s")

        print(f"\n--- Training finished after {self.max_episodes} episodes ---")

        if verbose:
            try:
                print("Generating final training visualizations...")
                coverage_matrix = self.grid_to_matrix()
                visualize_coverage(
                    coverage_matrix,
                    [agent.state for agent in self.agents],
                    episode_trajectories,
                    self.grid_size,
                    self.num_agents,
                    self.calculate_coverage_percentage(),
                    episode=f"Final ({current_map_type})",
                    title_suffix=" (Train Final)"
                )
                visualize_learning_metrics(self.metrics, self.num_agents)
            except Exception as e:
                print(f"   Final visualization error: {e}")

        if self.writer:
            self.writer.close()
            print("TensorBoard writer closed.")

        return self.metrics

    # --- Evaluation Method ---

    def evaluate(
        self,
        num_eval_episodes=20,
        max_steps_per_episode=150,
        model_dir="./models",
        model_episode_tag=None,
        visualize_each_episode=False,
        visualize_timesteps=False,
        save_animations=False,
        animation_dir="./eval_animations"
    ):
        """Evaluate trained agents on random maps."""
        print("\n--- Starting Evaluation Phase ---")

        if not self.agents:
            print("Agents not initialized, attempting reset...")
            self.reset(full_reset=True, mode='evaluate')
            if not self.agents:
                print("Error: Failed to init agents for eval.")
                return None

        if save_animations:
            ensure_dir(animation_dir)

        if model_episode_tag is None:
            latest_ep = get_latest_model_episode(model_dir, agent_id=0)
            if latest_ep != -1:
                model_episode_tag = latest_ep
                print(f"Using latest model tag found: {model_episode_tag}")
            else:
                model_episode_tag = self.max_episodes - 1
                print(f"Using default model tag: {model_episode_tag}")

        models_loaded = 0
        for agent_id, agent in enumerate(self.agents):
            model_path = os.path.join(model_dir, f"agent_{agent_id}_ep{model_episode_tag}.pt")
            if agent.load_model(model_path):
                models_loaded += 1
            agent.policy_net.eval()
            agent.epsilon = 0.0

        print(f"Attempted to load {len(self.agents)} models (Tag: {model_episode_tag}), {models_loaded} reported success.")

        if models_loaded == 0 and len(self.agents) > 0:
            print("Warning: No models loaded successfully.")

        all_episode_coverages = []
        all_episode_steps = []

        for episode in range(num_eval_episodes):
            is_last_episode = (episode == num_eval_episodes - 1)
            episode_label = f"Eval Ep {episode + 1}/{num_eval_episodes}"
            states = self.reset(full_reset=False, mode='evaluate')
            eval_trajectories = {i: [self.agents[i].state.position] for i in range(self.num_agents)}
            episode_done = False
            steps_taken = 0
            episode_coverage_history = []

            if visualize_timesteps and is_last_episode:
                episode_coverage_history.append(self.grid_to_matrix())

            for step in range(max_steps_per_episode):
                actions = {
                    i: agent.select_action(states[i], self.get_valid_actions(i))
                    for i, agent in enumerate(self.agents) if i in states
                }
                results = self.step(actions)

                for agent_id, result_tuple in results.items():
                    if agent_id < len(self.agents):
                        next_state, _, done, _ = result_tuple
                        states[agent_id] = next_state
                        eval_trajectories[agent_id].append(self.agents[agent_id].state.position)
                        if done:
                            episode_done = True

                if visualize_timesteps and is_last_episode:
                    episode_coverage_history.append(self.grid_to_matrix())

                steps_taken += 1

                if episode_done or steps_taken >= max_steps_per_episode:
                    break

            final_coverage_perc = self.calculate_coverage_percentage()
            all_episode_coverages.append(final_coverage_perc)
            all_episode_steps.append(steps_taken)
            print(f"  {episode_label}: Steps={steps_taken}, Final Coverage={final_coverage_perc:.2f}%")

            if visualize_each_episode:
                coverage_matrix = self.grid_to_matrix()
                visualize_coverage(
                    coverage_matrix,
                    [agent.state for agent in self.agents],
                    eval_trajectories,
                    self.grid_size,
                    self.num_agents,
                    final_coverage_perc,
                    episode=episode_label,
                    title_suffix=" (Eval Final)"
                )
                if self.num_agents > 0:
                    local_matrix = self.grid_to_matrix(graph=self.agents[0].local_map)
                    visualize_agent_local_map(
                        local_matrix,
                        self.agents[0].state,
                        0,
                        self.grid_size,
                        self.num_agents,
                        episode=episode_label
                    )

            if visualize_timesteps and is_last_episode:
                print(f"Generating animation for the final evaluation episode ({episode_label})...")
                anim_save_path = os.path.join(animation_dir, f"eval_ep_{episode + 1}_steps.mp4") if save_animations else None
                visualize_evaluation_timesteps(
                    episode_coverage_history,
                    episode_label,
                    self.grid_size,
                    self.obstacle_grid,
                    self.coverage_threshold,
                    save_path=anim_save_path
                )

        # Evaluation Summary
        avg_coverage = np.mean(all_episode_coverages) if all_episode_coverages else 0
        std_coverage = np.std(all_episode_coverages) if all_episode_coverages else 0
        avg_steps = np.mean(all_episode_steps) if all_episode_steps else 0

        print("\n--- Evaluation Summary ---")
        print(f" Model Tag: {model_episode_tag}, Episodes: {num_eval_episodes}")
        print(f" Avg Final Coverage (%): {avg_coverage:.2f} (StdDev: {std_coverage:.2f})")
        print(f" Avg Steps per Episode: {avg_steps:.1f}")
        print("--------------------------\n")

        return {
            "avg_coverage": avg_coverage,
            "std_coverage": std_coverage,
            "avg_steps": avg_steps
        }
