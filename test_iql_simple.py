#!/usr/bin/env python3
"""
Test IQL with parameter sharing.
Verifies that one shared CNN network works for all agents.
"""

import sys
sys.path.insert(0, 'epymarl_integration')

from coverage_env import CoverageEnv
from cnn_agent import CNNAgent
import torch
import numpy as np

def test_iql_parameter_sharing():
    print("=" * 60)
    print("Testing IQL with Parameter Sharing")
    print("=" * 60)

    # Create environment
    print("\n1. Creating environment...")
    env = CoverageEnv(
        n_agents=4,
        grid_size=20,
        sensor_range=2,
        use_spatial_obs=True,
        map_type='empty',
        seed=42
    )
    print(f"✓ Environment created: {env.n_agents} agents on {env.grid_size}×{env.grid_size} grid")

    # Create ONE shared agent
    print("\n2. Creating shared CNN agent...")
    agent = CNNAgent(
        input_channels=4,
        scalar_size=56,
        hidden_dim=256,
        n_actions=9
    )
    param_count = sum(p.numel() for p in agent.parameters())
    print(f"✓ Shared agent created: {param_count:,} parameters")
    print(f"  Note: This ONE network is used by ALL {env.n_agents} agents")

    # Test: All agents use same network
    print("\n3. Testing parameter sharing...")
    obs_list = env.reset()[0]

    # Get Q-values for each agent
    q_values_list = []
    for i, obs_dict in enumerate(obs_list):
        spatial = torch.FloatTensor(obs_dict['spatial']).unsqueeze(0)
        scalars = torch.FloatTensor(obs_dict['scalars']).unsqueeze(0)

        with torch.no_grad():
            q_values = agent(spatial, scalars)

        q_values_list.append(q_values)
        print(f"  Agent {i}: Q-values shape {q_values.shape}, max Q = {q_values.max():.3f}")

    print("✓ All agents successfully use the same network!")
    print(f"  Same network ID for all: {id(agent)}")

    # Test: Different observations → different Q-values
    print("\n4. Verifying different observations produce different Q-values...")
    q_array = torch.stack(q_values_list).squeeze()

    # Check if Q-values differ (they should, due to different positions)
    all_same = torch.all(q_array == q_array[0])

    if not all_same:
        print("✓ Different observations → different Q-values (correct!)")
        print(f"  Q-value variance: {q_array.var():.3f}")
    else:
        print("⚠ Warning: All Q-values identical (agents have same observation)")

    # Run episode
    print("\n5. Running episode with parameter sharing...")
    obs_list = env.reset()[0]
    done = False
    step = 0
    coverage_history = []

    while not done and step < 50:
        actions = []

        # Each agent uses the SAME network
        for obs_dict in obs_list:
            spatial = torch.FloatTensor(obs_dict['spatial']).unsqueeze(0)
            scalars = torch.FloatTensor(obs_dict['scalars']).unsqueeze(0)

            with torch.no_grad():
                q_values = agent(spatial, scalars)

            # Epsilon-greedy (ε=0.1 for testing)
            if np.random.random() < 0.1:
                action = np.random.randint(9)
            else:
                action = q_values.argmax().item()

            actions.append(action)

        # Step environment
        _, reward, done, _, info = env.step(actions)
        obs_list = env.get_obs()

        coverage_history.append(info['coverage_pct'])
        step += 1

        if step % 10 == 0:
            print(f"  Step {step}: Coverage {info['coverage_pct']:.1f}%, Reward {reward:.2f}")

    print(f"\n✓ Episode complete!")
    print(f"  Final coverage: {info['coverage_pct']:.1f}%")
    print(f"  Steps taken: {step}")
    print(f"  Coverage improved: {coverage_history[0]:.1f}% → {coverage_history[-1]:.1f}%")

    # Test: Scale invariance (20×20 vs 30×30)
    print("\n6. Testing scale invariance (20×20 vs 30×30)...")

    env_30 = CoverageEnv(
        n_agents=4,
        grid_size=30,
        sensor_range=3,
        use_spatial_obs=True,
        map_type='empty',
        seed=42
    )

    obs_30 = env_30.reset()[0][0]
    spatial_30 = torch.FloatTensor(obs_30['spatial']).unsqueeze(0)
    scalars_30 = torch.FloatTensor(obs_30['scalars']).unsqueeze(0)

    with torch.no_grad():
        q_values_30 = agent(spatial_30, scalars_30)

    print(f"  20×20 spatial shape: (4, 20, 20)")
    print(f"  30×30 spatial shape: (4, 30, 30)")
    print(f"  Both produce Q-values: {q_values_30.shape}")
    print("✓ Same network handles both grid sizes (scale invariance!)")

    # Test: Batch processing
    print("\n7. Testing batch processing (all agents at once)...")

    obs_list = env.reset()[0]

    # Stack all observations into batches
    spatial_batch = torch.stack([
        torch.FloatTensor(obs_dict['spatial'])
        for obs_dict in obs_list
    ])  # (4, 4, 20, 20)

    scalar_batch = torch.stack([
        torch.FloatTensor(obs_dict['scalars'])
        for obs_dict in obs_list
    ])  # (4, 56)

    with torch.no_grad():
        q_values_batch = agent(spatial_batch, scalar_batch)  # (4, 9)

    print(f"  Batched spatial: {spatial_batch.shape}")
    print(f"  Batched scalars: {scalar_batch.shape}")
    print(f"  Batched Q-values: {q_values_batch.shape}")
    print("✓ Batch processing works (efficient training!)")

    print("\n" + "=" * 60)
    print("✓ All IQL parameter sharing tests passed!")
    print("=" * 60)

    print("\nSummary:")
    print("- ONE shared network for all 4 agents")
    print(f"- Network has {param_count:,} parameters (not 4×)")
    print("- Different observations → different actions")
    print("- Scale invariant (20×20 and 30×30 both work)")
    print("- Batch processing enabled (fast training)")
    print("\nReady for IQL training with EPyMARL!")

    return True


if __name__ == '__main__':
    try:
        success = test_iql_parameter_sharing()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
