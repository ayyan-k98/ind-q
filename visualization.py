"""
Visualization utilities for MARL Coverage System
Contains all plotting and animation functions.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Optional, Tuple

from utils import ensure_dir, moving_average


def visualize_coverage(
    coverage_matrix: np.ndarray,
    agent_states: List,
    agent_trajectories: Dict[int, List[Tuple[int, int]]],
    grid_size: int,
    num_agents: int,
    coverage_percentage: float,
    episode: Optional[int] = None,
    title_suffix: str = ""
):
    """
    Plot coverage map and agent trajectories.

    Args:
        coverage_matrix: Matrix representing coverage values
        agent_states: List of agent states (position, orientation)
        agent_trajectories: Dictionary of agent trajectories
        grid_size: Size of the grid
        num_agents: Number of agents
        coverage_percentage: Current coverage percentage
        episode: Episode number (optional)
        title_suffix: Additional title text
    """
    plt.figure(figsize=(11, 10))

    cmap = plt.cm.YlGn.copy()
    cmap.set_under('black')  # Black for obstacles (<0)
    plt.imshow(coverage_matrix.T, cmap=cmap, origin='lower',
               interpolation='nearest', vmin=0.0, vmax=1.0)

    colors = plt.cm.get_cmap('tab10', num_agents)

    for idx, path in agent_trajectories.items():
        if idx < num_agents:
            agent_color = colors(idx)
            if path:
                plt.plot([p[0] for p in path], [p[1] for p in path],
                        '-', linewidth=1, color=agent_color, alpha=0.6)

            if idx < len(agent_states):
                current_pos = agent_states[idx].position
                orientation = agent_states[idx].orientation

                plt.scatter(current_pos[0], current_pos[1], color=agent_color,
                           s=120, marker='o', edgecolors='black', label=f'Agent {idx}')

                arrow_len = 0.8
                plt.arrow(current_pos[0], current_pos[1],
                         arrow_len * math.cos(orientation),
                         arrow_len * math.sin(orientation),
                         head_width=0.3, head_length=0.4, fc=agent_color,
                         ec='black', length_includes_head=True)

    title = "Multi-agent Coverage Map"
    if episode is not None:
        title += f" (Episode {episode}, Coverage: {coverage_percentage:.1f}%)"
    if title_suffix:
        title += f" {title_suffix}"

    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.colorbar(label='Coverage Probability (0-1)', shrink=0.8, extend='min')
    plt.xlim([-0.5, grid_size - 0.5])
    plt.ylim([-0.5, grid_size - 0.5])
    plt.xticks(np.arange(0, grid_size, step=max(1, grid_size // 10)))
    plt.yticks(np.arange(0, grid_size, step=max(1, grid_size // 10)))
    plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='white', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()


def visualize_agent_local_map(
    local_coverage_matrix: np.ndarray,
    agent_state,
    agent_id: int,
    grid_size: int,
    num_agents: int,
    episode: Optional[int] = None
):
    """
    Visualize the local map perspective of a specific agent.

    Args:
        local_coverage_matrix: Agent's local coverage matrix
        agent_state: Agent's current state
        agent_id: ID of the agent
        grid_size: Size of the grid
        num_agents: Total number of agents
        episode: Episode number (optional)
    """
    plt.figure(figsize=(9, 8))

    cmap = plt.cm.YlGn.copy()
    cmap.set_under('black')
    plt.imshow(local_coverage_matrix.T, cmap=cmap, origin='lower',
               interpolation='nearest', vmin=0.0, vmax=1.0)

    agent_pos = agent_state.position
    agent_color = plt.cm.get_cmap('tab10', num_agents)(agent_id)

    plt.scatter(agent_pos[0], agent_pos[1], color=agent_color, s=120,
               marker='o', label=f'Agent {agent_id} Pos', edgecolors='black')

    orientation = agent_state.orientation
    arrow_len = 0.8
    plt.arrow(agent_pos[0], agent_pos[1],
             arrow_len * math.cos(orientation),
             arrow_len * math.sin(orientation),
             head_width=0.3, head_length=0.4, fc=agent_color,
             ec='black', length_includes_head=True)

    title = f"Agent {agent_id}'s Final Local Map Perspective"
    if episode is not None:
        title += f" (End of Episode {episode})"

    plt.title(title)
    plt.colorbar(label='Coverage Probability (0-1)', shrink=0.8, extend='min')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.xlim([-0.5, grid_size - 0.5])
    plt.ylim([-0.5, grid_size - 0.5])
    plt.xticks(np.arange(0, grid_size, step=max(1, grid_size // 10)))
    plt.yticks(np.arange(0, grid_size, step=max(1, grid_size // 10)))
    plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='white', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()


def visualize_evaluation_timesteps(
    coverage_history: List[np.ndarray],
    episode_label: str,
    grid_size: int,
    obstacle_grid: Optional[np.ndarray],
    coverage_threshold: float,
    save_path: Optional[str] = None
):
    """
    Generate an animation showing global coverage evolution.

    Args:
        coverage_history: List of coverage matrices over time
        episode_label: Label for the episode
        grid_size: Size of the grid
        obstacle_grid: Obstacle map
        coverage_threshold: Threshold for considering a cell covered
        save_path: Path to save animation (optional)
    """
    if not coverage_history:
        return

    fig, ax = plt.subplots(figsize=(9, 8))
    num_steps = len(coverage_history)
    initial_matrix = coverage_history[0]

    cmap = plt.cm.YlGn.copy()
    cmap.set_under('black')
    im = ax.imshow(initial_matrix.T, cmap=cmap, origin='lower',
                   interpolation='nearest', vmin=0.0, vmax=1.0)

    colorbar = fig.colorbar(im, ax=ax, label='Coverage Prob (0-1)',
                            shrink=0.8, extend='min')
    title = ax.set_title(f"{episode_label} - Step 0/{num_steps-1}")

    ax.set_xlim([-0.5, grid_size - 0.5])
    ax.set_ylim([-0.5, grid_size - 0.5])
    ax.set_xticks(np.arange(0, grid_size, step=max(1, grid_size // 5)))
    ax.set_yticks(np.arange(0, grid_size, step=max(1, grid_size // 5)))
    ax.grid(True, linestyle=':', linewidth=0.5, color='white', alpha=0.5)

    obstacle_mask = (obstacle_grid == 1.0) if obstacle_grid is not None else np.zeros_like(initial_matrix, dtype=bool)
    traversable_nodes_count = grid_size * grid_size - np.sum(obstacle_mask)

    def update(frame):
        matrix = coverage_history[frame]
        im.set_data(matrix.T)
        covered_count = 0
        if traversable_nodes_count > 0:
            covered_count = np.sum((matrix >= coverage_threshold) & (~obstacle_mask))
            cov_pct = (covered_count / traversable_nodes_count) * 100.0
        else:
            cov_pct = 100.0
        title.set_text(f"{episode_label} - Step {frame}/{num_steps-1} (Coverage: {cov_pct:.1f}%)")
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=num_steps,
                                 interval=150, blit=False, repeat=False)

    if save_path:
        try:
            ensure_dir(os.path.dirname(save_path))
            ani.save(save_path, writer='ffmpeg', fps=10, dpi=100)
            print(f"Eval animation saved: {save_path}")
        except Exception as e:
            print(f"Error saving animation: {e}\nDisplaying instead.")
            plt.show()
    else:
        plt.show()

    plt.close(fig)


def visualize_learning_metrics(metrics, num_agents: int):
    """
    Visualize learning metrics dashboard.

    Args:
        metrics: CoverageMetrics object containing training metrics
        num_agents: Number of agents
    """
    if not metrics.total_coverage and not any(metrics.agent_losses.values()):
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.ravel()
    fig.suptitle("Learning Metrics Dashboard", fontsize=16)

    smooth_window = max(1, len(metrics.total_coverage) // 20) if metrics.total_coverage else 1

    # Total Coverage Area
    ax = axes[0]
    steps = np.arange(len(metrics.total_coverage))
    if len(steps) > 0:
        ax.plot(steps, metrics.total_coverage, 'g-', alpha=0.3, label='Raw')
        smoothed_cov, x_cov = moving_average(metrics.total_coverage, smooth_window)
        ax.plot(x_cov, smoothed_cov, 'g-', linewidth=2, label=f'Smoothed ({smooth_window})')
    ax.set_title('Total Coverage Area (Sum pc)')
    ax.set_xlabel('Environment Step')
    ax.set_ylabel('Area')
    ax.legend()
    ax.grid(True, alpha=0.5)

    # Epsilon Decay
    ax = axes[1]
    num_eps = 0
    colors = plt.cm.viridis(np.linspace(0, 0.8, num_agents))
    for aid, eps in metrics.epsilon_values.items():
        if eps:
            num_eps = len(eps)
            ax.plot(eps, label=f'Agent {aid}', color=colors[aid], alpha=0.8)
    if num_eps > 0:
        ax.set_title('Epsilon Decay')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
    ax.legend()
    ax.grid(True, alpha=0.5)

    # Average Agent Reward
    ax = axes[2]
    avg_rewards = []
    step_counts = [len(r) for r in metrics.agent_rewards.values()]
    if step_counts:
        max_steps = max(step_counts) if step_counts else 0
        for step in range(max_steps):
            step_rewards = [metrics.agent_rewards[aid][step]
                           for aid in metrics.agent_rewards
                           if step < len(metrics.agent_rewards[aid])]
            if step_rewards:
                avg_rewards.append(np.mean(step_rewards))
        if avg_rewards:
            ax.plot(np.arange(len(avg_rewards)), avg_rewards, 'b-', alpha=0.3, label='Raw Avg')
            smoothed_rew, x_rew = moving_average(avg_rewards, smooth_window * 2)
            ax.plot(x_rew, smoothed_rew, 'b-', linewidth=2, label=f'Smoothed ({smooth_window*2})')
    ax.set_title('Average Agent Reward per Step')
    ax.set_xlabel('Environment Step')
    ax.set_ylabel('Avg Reward')
    ax.legend()
    ax.grid(True, alpha=0.5)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)

    # Average Agent Loss
    ax = axes[3]
    avg_losses = []
    opt_step_counts = [len(l) for l in metrics.agent_losses.values()]
    if opt_step_counts:
        max_opt_steps = max(opt_step_counts) if opt_step_counts else 0
        for step in range(max_opt_steps):
            step_losses = [metrics.agent_losses[aid][step]
                          for aid in metrics.agent_losses
                          if step < len(metrics.agent_losses[aid])]
            if step_losses:
                avg_losses.append(np.mean(step_losses))
        if avg_losses:
            ax.plot(np.arange(len(avg_losses)), avg_losses, 'r-', alpha=0.3, label='Raw Avg')
            smoothed_loss, x_loss = moving_average(avg_losses, smooth_window * 2)
            ax.plot(x_loss, smoothed_loss, 'r-', linewidth=2, label=f'Smoothed ({smooth_window*2})')
    ax.set_title('Average Agent Loss per Opt. Step')
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Avg Loss')
    ax.legend()
    ax.grid(True, alpha=0.5)

    # Communication Events
    ax = axes[4]
    if metrics.communication_events:
        ax.plot(np.arange(len(metrics.communication_events)),
               metrics.communication_events, 'm-', alpha=0.6)
    ax.set_title('Communication Events per Step')
    ax.set_xlabel('Environment Step')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.5)

    # Coverage Rate
    ax = axes[5]
    if metrics.coverage_rate:
        ax.plot(np.arange(len(metrics.coverage_rate)),
               metrics.coverage_rate, 'c-', alpha=0.3, label='Raw')
        smoothed_rate, x_rate = moving_average(metrics.coverage_rate, smooth_window)
        ax.plot(x_rate, smoothed_rate, 'c-', linewidth=2, label=f'Smoothed ({smooth_window})')
    ax.set_title('Coverage Rate (Δ Area / Step)')
    ax.set_xlabel('Environment Step')
    ax.set_ylabel('Δ Area')
    ax.legend()
    ax.grid(True, alpha=0.5)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
