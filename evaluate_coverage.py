"""
Evaluation script for Coverage Environment
Includes baselines (random, greedy, single-agent) and visualization
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict, List, Tuple
import json

sys.path.insert(0, './epymarl/src')

from epymarl_coverage_env import CoverageEnvironment


class BaselineAgent:
    """Base class for baseline agents."""

    def select_actions(self, env: CoverageEnvironment) -> List[int]:
        """Select actions for all agents."""
        raise NotImplementedError


class RandomAgent(BaselineAgent):
    """Random baseline: selects random valid actions."""

    def select_actions(self, env: CoverageEnvironment) -> List[int]:
        actions = []
        for agent_id in range(env.n_agents):
            avail = env.get_avail_agent_actions(agent_id)
            valid_actions = [i for i, a in enumerate(avail) if a == 1]
            actions.append(np.random.choice(valid_actions))
        return actions


class GreedyAgent(BaselineAgent):
    """Greedy baseline: moves toward nearest uncovered cell."""

    def select_actions(self, env: CoverageEnvironment) -> List[int]:
        actions = []
        for agent_id in range(env.n_agents):
            action = self._greedy_action(env, agent_id)
            actions.append(action)
        return actions

    def _greedy_action(self, env: CoverageEnvironment, agent_id: int) -> int:
        """Find action that moves toward nearest frontier."""
        pos = env.agent_positions[agent_id]
        avail = env.get_avail_agent_actions(agent_id)

        # Find nearest uncovered cell
        best_dist = float('inf')
        best_target = None

        for r in range(env.grid_size):
            for c in range(env.grid_size):
                if env.obstacle_grid[r, c] == 0:  # Free cell
                    if env.coverage_grid[r, c] < env.coverage_threshold:  # Uncovered
                        dist = abs(r - pos[0]) + abs(c - pos[1])  # Manhattan
                        if dist < best_dist:
                            best_dist = dist
                            best_target = (r, c)

        if best_target is None:
            # All covered, stay
            return 4  # Stay action

        # Find action that moves toward target
        target_r, target_c = best_target
        dr = np.clip(target_r - pos[0], -1, 1)
        dc = np.clip(target_c - pos[1], -1, 1)
        desired_move = (dr, dc)

        # Find corresponding action
        try:
            action_idx = env.actions.index(desired_move)
            if avail[action_idx] == 1:
                return action_idx
        except ValueError:
            pass

        # Fallback: random valid action
        valid_actions = [i for i, a in enumerate(avail) if a == 1]
        return np.random.choice(valid_actions)


def evaluate_baseline(
    baseline_agent: BaselineAgent,
    n_episodes: int = 20,
    env_config: dict = None
) -> Dict:
    """
    Evaluate a baseline agent.

    Returns:
        results: Dictionary with metrics
    """
    if env_config is None:
        env_config = {}

    # Create environment
    env = CoverageEnvironment(**env_config)

    episode_returns = []
    episode_coverages = []
    episode_lengths = []
    episode_times = []

    for ep in range(n_episodes):
        env.reset()
        done = False
        ep_return = 0
        step_count = 0

        start_time = time.time()

        while not done:
            actions = baseline_agent.select_actions(env)
            reward, done, info = env.step(actions)
            ep_return += reward
            step_count += 1

        elapsed = time.time() - start_time

        final_coverage = env._calculate_coverage_percentage()

        episode_returns.append(ep_return)
        episode_coverages.append(final_coverage)
        episode_lengths.append(step_count)
        episode_times.append(elapsed)

        print(f"  Episode {ep+1}/{n_episodes}: Coverage={final_coverage:.1f}%, "
              f"Return={ep_return:.1f}, Steps={step_count}, Time={elapsed:.2f}s")

    results = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_coverage': np.mean(episode_coverages),
        'std_coverage': np.std(episode_coverages),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_time': np.mean(episode_times),
        'std_time': np.std(episode_times),
        'episodes': {
            'returns': episode_returns,
            'coverages': episode_coverages,
            'lengths': episode_lengths,
            'times': episode_times
        }
    }

    return results


def evaluate_single_agent(
    n_episodes: int = 20,
    env_config: dict = None
) -> Dict:
    """Evaluate single agent with greedy policy."""

    if env_config is None:
        env_config = {}

    # Force single agent
    env_config['n_agents'] = 1

    return evaluate_baseline(GreedyAgent(), n_episodes, env_config)


def compare_baselines(env_config: dict = None, n_episodes: int = 20):
    """Compare all baselines."""

    print("=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)
    print()

    results = {}

    # Random baseline
    print("Evaluating Random Baseline...")
    results['random'] = evaluate_baseline(RandomAgent(), n_episodes, env_config)
    print()

    # Greedy baseline
    print("Evaluating Greedy Baseline...")
    results['greedy'] = evaluate_baseline(GreedyAgent(), n_episodes, env_config)
    print()

    # Single agent
    print("Evaluating Single Agent (Greedy)...")
    results['single_agent'] = evaluate_single_agent(n_episodes, env_config)
    print()

    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()

    for name, res in results.items():
        print(f"{name.upper()}:")
        print(f"  Coverage: {res['mean_coverage']:.2f}% ± {res['std_coverage']:.2f}%")
        print(f"  Return:   {res['mean_return']:.2f} ± {res['std_return']:.2f}")
        print(f"  Steps:    {res['mean_length']:.1f} ± {res['std_length']:.1f}")
        print(f"  Time:     {res['mean_time']:.2f}s ± {res['std_time']:.2f}s")
        print()

    return results


def visualize_episode(env: CoverageEnvironment, save_path: str = None):
    """Visualize a single episode with heatmap."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Coverage heatmap
    ax = axes[0]
    coverage_viz = np.copy(env.coverage_grid)
    coverage_viz[env.obstacle_grid == 1] = -0.1  # Mark obstacles

    im = ax.imshow(coverage_viz.T, cmap='YlGn', origin='lower', vmin=0, vmax=1)
    ax.set_title(f'Coverage Map (Step {env.steps})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Coverage Probability')

    # Add agent positions
    for i, pos in enumerate(env.agent_positions):
        ax.plot(pos[0], pos[1], 'ro', markersize=10, label=f'Agent {i}' if i < 3 else '')
        # Add orientation arrow
        dx = 0.5 * np.cos(env.agent_orientations[i])
        dy = 0.5 * np.sin(env.agent_orientations[i])
        ax.arrow(pos[0], pos[1], dx, dy, head_width=0.3, head_length=0.3, fc='r', ec='r')

    if env.n_agents <= 3:
        ax.legend()

    # Obstacle map
    ax = axes[1]
    ax.imshow(env.obstacle_grid.T, cmap='binary', origin='lower')
    ax.set_title('Obstacle Map')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_comparison(results: Dict, save_path: str = None):
    """Create comparison plots for baseline results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = list(results.keys())
    colors = ['blue', 'green', 'orange', 'red']

    # Coverage comparison
    ax = axes[0, 0]
    coverages = [results[m]['mean_coverage'] for m in methods]
    stds = [results[m]['std_coverage'] for m in methods]
    ax.bar(methods, coverages, yerr=stds, color=colors[:len(methods)], alpha=0.7)
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Final Coverage Comparison')
    ax.axhline(y=85, color='r', linestyle='--', label='Target (85%)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Return comparison
    ax = axes[0, 1]
    returns = [results[m]['mean_return'] for m in methods]
    stds = [results[m]['std_return'] for m in methods]
    ax.bar(methods, returns, yerr=stds, color=colors[:len(methods)], alpha=0.7)
    ax.set_ylabel('Episode Return')
    ax.set_title('Episode Return Comparison')
    ax.grid(axis='y', alpha=0.3)

    # Steps comparison
    ax = axes[1, 0]
    steps = [results[m]['mean_length'] for m in methods]
    stds = [results[m]['std_length'] for m in methods]
    ax.bar(methods, steps, yerr=stds, color=colors[:len(methods)], alpha=0.7)
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length Comparison')
    ax.grid(axis='y', alpha=0.3)

    # Time comparison
    ax = axes[1, 1]
    times = [results[m]['mean_time'] for m in methods]
    stds = [results[m]['std_time'] for m in methods]
    ax.bar(methods, times, yerr=stds, color=colors[:len(methods)], alpha=0.7)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Execution Time Comparison')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Coverage Baselines')
    parser.add_argument('--n-episodes', type=int, default=20, help='Number of episodes')
    parser.add_argument('--n-agents', type=int, default=4, help='Number of agents')
    parser.add_argument('--grid-size', type=int, default=20, help='Grid size')
    parser.add_argument('--episode-limit', type=int, default=200, help='Max steps')
    parser.add_argument('--map-type', type=str, default='empty',
                        choices=['empty', 'rooms', 'random'], help='Map type')
    parser.add_argument('--save-dir', type=str, default='./eval_results', help='Save directory')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')

    args = parser.parse_args()

    # Setup
    os.makedirs(args.save_dir, exist_ok=True)

    # Environment config
    env_config = {
        'n_agents': args.n_agents,
        'grid_size': args.grid_size,
        'episode_limit': args.episode_limit,
        'map_type': args.map_type
    }

    # Run evaluation
    results = compare_baselines(env_config, args.n_episodes)

    # Save results
    results_path = os.path.join(args.save_dir, 'baseline_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy types to python types for JSON
        json_results = {}
        for method, res in results.items():
            json_results[method] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in res.items()
                if k != 'episodes'  # Skip raw episode data
            }
        json.dump(json_results, f, indent=2)
    print(f"✓ Results saved to {results_path}")

    # Visualize
    if args.visualize:
        comparison_path = os.path.join(args.save_dir, 'baseline_comparison.png')
        visualize_comparison(results, comparison_path)

        # Visualize one episode
        print("\nVisualizing sample episode...")
        env = CoverageEnvironment(**env_config)
        env.reset()
        agent = GreedyAgent()

        for _ in range(50):  # Run 50 steps
            actions = agent.select_actions(env)
            env.step(actions)

        viz_path = os.path.join(args.save_dir, 'sample_episode.png')
        visualize_episode(env, viz_path)

    print("\n✓ Evaluation complete!")
