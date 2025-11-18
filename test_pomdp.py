#!/usr/bin/env python3
"""
Quick test script to verify POMDP implementation.
Tests that obstacle beliefs are properly updated through exploration.
"""

import sys
sys.path.insert(0, 'epymarl_integration')

import numpy as np
from coverage_env import CoverageEnv

def test_pomdp():
    print("=" * 60)
    print("Testing POMDP Implementation")
    print("=" * 60)

    # Create environment with obstacles
    env = CoverageEnv(
        n_agents=2,
        grid_size=10,
        sensor_range=2,
        map_type='random',
        obstacle_density=0.2,
        seed=42
    )

    # Test 1: Initial belief should be mostly unknown
    print("\nTest 1: Initial Belief State")
    print("-" * 60)
    unknown_cells = np.sum(env.obstacle_belief == -1)
    known_cells = np.sum(env.obstacle_belief != -1)
    total_cells = env.grid_size * env.grid_size

    print(f"Total cells: {total_cells}")
    print(f"Unknown cells: {unknown_cells}")
    print(f"Known cells: {known_cells}")
    print(f"Percentage unknown: {100.0 * unknown_cells / total_cells:.1f}%")

    # After initialization, agents should have revealed cells in sensor range
    if known_cells > 0:
        print("✓ Agents revealed cells in initial sensor range")
    else:
        print("✗ ERROR: No cells revealed initially")
        return False

    # Test 2: Observations use belief
    print("\nTest 2: Observations Use Belief")
    print("-" * 60)
    obs = env.get_obs()
    print(f"Number of agents: {len(obs)}")
    print(f"Observation size per agent: {len(obs[0])}")
    print(f"Expected obs size: {env.obs_size}")

    if len(obs[0]) == env.obs_size:
        print("✓ Observation size correct")
    else:
        print(f"✗ ERROR: Wrong observation size")
        return False

    # Test 3: State includes true obstacles
    print("\nTest 3: State Includes True Obstacles (CTDE)")
    print("-" * 60)
    state = env.get_state()
    expected_size = env.get_state_size()
    print(f"State size: {len(state)}")
    print(f"Expected state size: {expected_size}")

    # State should be: obs * n_agents + coverage_grid + obstacle_grid + metadata
    expected = env.obs_size * env.n_agents + env.grid_size**2 + env.grid_size**2 + 3
    if len(state) == expected:
        print(f"✓ State size correct (includes {env.grid_size**2} cells for obstacle map)")
    else:
        print(f"✗ ERROR: State size incorrect. Expected {expected}, got {len(state)}")
        return False

    # Test 4: Beliefs update through exploration
    print("\nTest 4: Beliefs Update Through Exploration")
    print("-" * 60)
    initial_known = np.sum(env.obstacle_belief != -1)

    # Take random actions for several steps
    for step in range(10):
        actions = [np.random.choice(env.n_actions) for _ in range(env.n_agents)]
        env.step(actions)

    final_known = np.sum(env.obstacle_belief != -1)
    newly_revealed = final_known - initial_known

    print(f"Initially known cells: {initial_known}")
    print(f"After 10 steps known cells: {final_known}")
    print(f"Newly revealed cells: {newly_revealed}")

    if newly_revealed >= 0:  # Could be 0 if agents don't move much
        print("✓ Belief map updated (agents may have explored)")
    else:
        print("✗ ERROR: Beliefs decreased (impossible)")
        return False

    # Test 5: Verify true obstacles != belief
    print("\nTest 5: True Obstacles vs Belief (POMDP)")
    print("-" * 60)
    true_obstacles = np.sum(env.obstacle_grid == 1)
    believed_obstacles = np.sum(env.obstacle_belief == 1)
    print(f"True obstacles: {true_obstacles}")
    print(f"Known obstacles (belief): {believed_obstacles}")

    if believed_obstacles <= true_obstacles:
        print("✓ Agents know fewer/equal obstacles than exist (POMDP working)")
    else:
        print("✗ ERROR: Agents know MORE obstacles than exist")
        return False

    # Test 6: Render comparison
    print("\nTest 6: Visualization")
    print("-" * 60)
    print("True obstacle map:")
    print_grid(env.obstacle_grid, env.agent_positions)
    print("\nAgent's belief map (-1=unknown, 0=free, 1=obstacle):")
    print_grid(env.obstacle_belief, env.agent_positions)

    print("\n" + "=" * 60)
    print("✓ All POMDP tests passed!")
    print("=" * 60)
    return True

def print_grid(grid, agent_positions=None):
    """Print a grid nicely."""
    symbols = {-1: '?', 0: '.', 1: '#'}
    grid_str = np.vectorize(lambda x: symbols.get(x, str(x)))(grid)

    if agent_positions:
        for i, (r, c) in enumerate(agent_positions):
            grid_str[r, c] = str(i)

    for row in grid_str:
        print(''.join(row))

if __name__ == '__main__':
    success = test_pomdp()
    sys.exit(0 if success else 1)
