"""
Visualization and Debugging Tools for Coverage Environment

Provides tools for:
- Coverage heatmaps
- Agent trajectories
- Sensor FOV visualization
- Episode replay
- Training metrics
- Action distribution analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


class CoverageVisualizer:
    """Visualization tools for coverage environment debugging."""

    def __init__(self, env):
        """
        Initialize visualizer.

        Args:
            env: CoverageEnv instance
        """
        self.env = env
        self.trajectory_history = []
        self.coverage_history = []
        self.reward_history = []
        self.action_history = []

    def plot_coverage_heatmap(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot coverage heatmap showing coverage intensity.

        Args:
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Coverage heatmap
        im = ax.imshow(self.env.coverage_grid, cmap='YlOrRd', vmin=0, vmax=1)

        # Obstacles
        obstacle_mask = self.env.obstacle_grid == 1
        ax.imshow(np.where(obstacle_mask, 1, np.nan), cmap='gray', alpha=0.5)

        # Agent positions
        for i, pos in enumerate(self.env.agent_positions):
            circle = plt.Circle((pos[1], pos[0]), 0.3, color='blue', alpha=0.7)
            ax.add_patch(circle)
            ax.text(pos[1], pos[0], str(i), ha='center', va='center',
                   color='white', fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coverage Intensity', rotation=270, labelpad=20)

        # Labels
        coverage_pct = self.env._calculate_coverage_percentage()
        ax.set_title(f'Coverage Heatmap (Step {self.env.steps}, Coverage: {coverage_pct:.1f}%)')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_sensor_fov(self, agent_id: int, save_path: Optional[str] = None, show: bool = True):
        """
        Visualize agent's sensor field of view.

        Args:
            agent_id: Agent to visualize
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Base coverage
        ax.imshow(self.env.coverage_grid, cmap='gray', alpha=0.3)

        # Agent position and orientation
        pos = self.env.agent_positions[agent_id]
        orientation = self.env.agent_orientations[agent_id]

        # Draw sensor FOV
        fov_angle = self.env.fov_radians
        start_angle = np.degrees(orientation - fov_angle/2)
        wedge = patches.Wedge(
            (pos[1], pos[0]),
            self.env.sensor_range,
            start_angle - 90,  # Adjust for matplotlib coordinates
            start_angle - 90 + np.degrees(fov_angle),
            color='yellow',
            alpha=0.3,
            label='Sensor FOV'
        )
        ax.add_patch(wedge)

        # Agent
        circle = plt.Circle((pos[1], pos[0]), 0.3, color='blue', alpha=0.9)
        ax.add_patch(circle)

        # Orientation arrow
        dx = np.cos(orientation) * 1.5
        dy = np.sin(orientation) * 1.5
        ax.arrow(pos[1], pos[0], dy, dx, head_width=0.5, head_length=0.5,
                fc='red', ec='red', linewidth=2)

        # Raycasted cells
        for r in range(max(0, pos[0] - self.env.sensor_range),
                      min(self.env.grid_size, pos[0] + self.env.sensor_range + 1)):
            for c in range(max(0, pos[1] - self.env.sensor_range),
                          min(self.env.grid_size, pos[1] + self.env.sensor_range + 1)):
                dist = np.sqrt((r - pos[0])**2 + (c - pos[1])**2)
                if dist <= self.env.sensor_range and self.env.obstacle_grid[r, c] == 0:
                    angle_to_cell = np.arctan2(r - pos[0], c - pos[1])
                    angle_diff = abs(((angle_to_cell - orientation + np.pi) % (2*np.pi)) - np.pi)

                    if angle_diff <= fov_angle / 2:
                        rect = patches.Rectangle(
                            (c - 0.4, r - 0.4), 0.8, 0.8,
                            linewidth=0.5, edgecolor='green', facecolor='none'
                        )
                        ax.add_patch(rect)

        ax.set_xlim(-0.5, self.env.grid_size - 0.5)
        ax.set_ylim(self.env.grid_size - 0.5, -0.5)
        ax.set_title(f'Agent {agent_id} Sensor FOV (Range: {self.env.sensor_range}, FOV: {np.degrees(fov_angle):.0f}°)')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_trajectory(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot agent trajectories over the episode.

        Args:
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        if not self.trajectory_history:
            print("No trajectory history recorded. Call record_step() during episode.")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # Coverage heatmap
        ax.imshow(self.env.coverage_grid, cmap='YlOrRd', alpha=0.4)

        # Obstacles
        obstacle_mask = self.env.obstacle_grid == 1
        ax.imshow(np.where(obstacle_mask, 1, np.nan), cmap='gray', alpha=0.5)

        # Plot trajectories for each agent
        colors = plt.cm.tab10(range(self.env.n_agents))
        for agent_id in range(self.env.n_agents):
            trajectory = [(step[agent_id][1], step[agent_id][0])
                         for step in self.trajectory_history]
            if trajectory:
                traj_array = np.array(trajectory)
                ax.plot(traj_array[:, 0], traj_array[:, 1],
                       'o-', color=colors[agent_id], alpha=0.6,
                       label=f'Agent {agent_id}', linewidth=2, markersize=3)

                # Mark start and end
                ax.plot(traj_array[0, 0], traj_array[0, 1],
                       'o', color=colors[agent_id], markersize=10,
                       markeredgecolor='white', markeredgewidth=2)
                ax.plot(traj_array[-1, 0], traj_array[-1, 1],
                       's', color=colors[agent_id], markersize=10,
                       markeredgecolor='white', markeredgewidth=2)

        ax.set_title(f'Agent Trajectories ({len(self.trajectory_history)} steps)')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.legend()
        ax.set_xlim(-0.5, self.env.grid_size - 0.5)
        ax.set_ylim(self.env.grid_size - 0.5, -0.5)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_coverage_over_time(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot coverage percentage over time.

        Args:
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        if not self.coverage_history:
            print("No coverage history recorded. Call record_step() during episode.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        steps = range(len(self.coverage_history))
        ax.plot(steps, self.coverage_history, 'b-', linewidth=2)
        ax.fill_between(steps, self.coverage_history, alpha=0.3)

        ax.set_xlabel('Step')
        ax.set_ylabel('Coverage %')
        ax.set_title('Coverage Over Time')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_action_distribution(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot action distribution for each agent.

        Args:
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        if not self.action_history:
            print("No action history recorded. Call record_step() during episode.")
            return

        fig, axes = plt.subplots(1, self.env.n_agents, figsize=(4*self.env.n_agents, 4))
        if self.env.n_agents == 1:
            axes = [axes]

        action_names = ['NW', 'N', 'NE', 'W', 'Stay', 'E', 'SW', 'S', 'SE']

        for agent_id in range(self.env.n_agents):
            actions = [step[agent_id] for step in self.action_history]
            action_counts = np.bincount(actions, minlength=9)

            axes[agent_id].bar(range(9), action_counts, color='steelblue')
            axes[agent_id].set_xticks(range(9))
            axes[agent_id].set_xticklabels(action_names, rotation=45)
            axes[agent_id].set_title(f'Agent {agent_id}')
            axes[agent_id].set_ylabel('Count')
            axes[agent_id].grid(True, alpha=0.3, axis='y')

        fig.suptitle('Action Distribution', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_reward_breakdown(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot reward components over time.

        Args:
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        if not self.reward_history:
            print("No reward history recorded. Call record_step() during episode.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        steps = range(len(self.reward_history))
        rewards = [r['total'] for r in self.reward_history]
        coverage_rewards = [r['coverage'] for r in self.reward_history]

        ax.plot(steps, rewards, 'b-', label='Total Reward', linewidth=2)
        ax.plot(steps, coverage_rewards, 'g--', label='Coverage Reward', linewidth=1.5)

        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def create_episode_summary(self, save_dir: str):
        """
        Create comprehensive episode summary with all plots.

        Args:
            save_dir: Directory to save all plots
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating episode summary in {save_dir}/")

        # Coverage heatmap
        self.plot_coverage_heatmap(save_path=save_dir / 'coverage_heatmap.png', show=False)
        print("✓ Coverage heatmap saved")

        # Trajectories
        if self.trajectory_history:
            self.plot_trajectory(save_path=save_dir / 'trajectories.png', show=False)
            print("✓ Trajectories saved")

        # Coverage over time
        if self.coverage_history:
            self.plot_coverage_over_time(save_path=save_dir / 'coverage_over_time.png', show=False)
            print("✓ Coverage over time saved")

        # Action distribution
        if self.action_history:
            self.plot_action_distribution(save_path=save_dir / 'action_distribution.png', show=False)
            print("✓ Action distribution saved")

        # Reward breakdown
        if self.reward_history:
            self.plot_reward_breakdown(save_path=save_dir / 'rewards.png', show=False)
            print("✓ Rewards saved")

        # Sensor FOV for each agent
        for agent_id in range(self.env.n_agents):
            self.plot_sensor_fov(agent_id, save_path=save_dir / f'sensor_fov_agent_{agent_id}.png', show=False)
        print(f"✓ Sensor FOV saved for all {self.env.n_agents} agents")

        # Save summary statistics
        stats = {
            'episode_length': self.env.steps,
            'final_coverage': float(self.env._calculate_coverage_percentage()),
            'total_reward': float(sum(r['total'] for r in self.reward_history)) if self.reward_history else 0,
            'n_agents': self.env.n_agents,
            'grid_size': self.env.grid_size,
        }

        with open(save_dir / 'summary.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print("✓ Summary statistics saved")

        print(f"\n✓ Episode summary complete!")
        print(f"  Coverage: {stats['final_coverage']:.1f}%")
        print(f"  Episode length: {stats['episode_length']} steps")
        print(f"  Total reward: {stats['total_reward']:.2f}")

    def record_step(self, actions: List[int], reward: float, reward_info: Dict):
        """
        Record step for visualization history.

        Args:
            actions: List of actions taken
            reward: Total reward received
            reward_info: Reward breakdown dictionary
        """
        # Record agent positions
        self.trajectory_history.append([pos.copy() for pos in self.env.agent_positions])

        # Record coverage
        coverage_pct = self.env._calculate_coverage_percentage()
        self.coverage_history.append(coverage_pct)

        # Record actions
        self.action_history.append(actions.copy())

        # Record rewards
        self.reward_history.append({
            'total': reward,
            'coverage': reward_info.get('coverage_reward', 0),
        })

    def reset(self):
        """Clear all history for new episode."""
        self.trajectory_history = []
        self.coverage_history = []
        self.reward_history = []
        self.action_history = []


def debug_episode(env, policy_fn, max_steps: int = 200, save_dir: str = 'debug_episode'):
    """
    Run an episode with full visualization and debugging.

    Args:
        env: CoverageEnv instance
        policy_fn: Function that takes (env, agent_id) and returns action
        max_steps: Maximum steps to run
        save_dir: Directory to save debug outputs

    Returns:
        Episode statistics dictionary
    """
    visualizer = CoverageVisualizer(env)

    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    print(f"Starting debug episode (max {max_steps} steps)...")

    while not done and step < max_steps:
        # Get actions from policy
        actions = [policy_fn(env, i) for i in range(env.n_agents)]

        # Step environment
        reward, done, info = env.step(actions)

        # Record for visualization
        visualizer.record_step(actions, reward, info)

        total_reward += reward
        step += 1

        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"Step {step}: Coverage {info['coverage_pct']:.1f}%, Reward: {reward:.2f}")

    print(f"\nEpisode finished!")
    print(f"  Final coverage: {info['coverage_pct']:.1f}%")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Steps: {step}")

    # Generate comprehensive summary
    visualizer.create_episode_summary(save_dir)

    return {
        'coverage': info['coverage_pct'],
        'reward': total_reward,
        'steps': step,
    }


# Example usage functions
def greedy_policy(env, agent_id):
    """Simple greedy policy for testing."""
    avail = env.get_avail_agent_actions(agent_id)
    return avail.index(1) if 1 in avail else 4  # Stay if no valid actions


def random_policy(env, agent_id):
    """Random policy for testing."""
    import random
    avail = env.get_avail_agent_actions(agent_id)
    valid_actions = [i for i, a in enumerate(avail) if a == 1]
    return random.choice(valid_actions) if valid_actions else 4


if __name__ == '__main__':
    # Example: Debug a random episode
    import sys
    sys.path.insert(0, './epymarl/src')
    from envs.coverage import CoverageEnv

    env = CoverageEnv(n_agents=4, grid_size=20)

    print("Running debug episode with random policy...")
    stats = debug_episode(env, random_policy, max_steps=200, save_dir='debug_output')

    print("\n✓ Debug complete! Check debug_output/ directory for visualizations.")
