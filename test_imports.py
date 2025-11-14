"""
Test script to verify all imports work correctly
"""

print("Testing imports...")

try:
    print("  - Importing config...")
    import config
    print("    ✓ config imported successfully")

    print("  - Importing data_structures...")
    from data_structures import RobotState, WorldState, CoverageMetrics, CommunicationEvent, Room
    print("    ✓ data_structures imported successfully")

    print("  - Importing utils...")
    from utils import ensure_dir, set_seeds, moving_average, get_latest_model_episode
    print("    ✓ utils imported successfully")

    print("  - Importing replay_memory...")
    from replay_memory import ReplayMemory
    print("    ✓ replay_memory imported successfully")

    print("  - Importing networks...")
    from networks import ConvDQN, DuelingConvDQN
    print("    ✓ networks imported successfully")

    print("  - Importing visualization...")
    from visualization import (
        visualize_coverage,
        visualize_agent_local_map,
        visualize_evaluation_timesteps,
        visualize_learning_metrics
    )
    print("    ✓ visualization imported successfully")

    print("  - Importing agent...")
    from agent import MARLCoverageAgent
    print("    ✓ agent imported successfully")

    print("  - Importing environment...")
    from environment import MARLCoverageEnvironment
    print("    ✓ environment imported successfully")

    print("\n✓ All imports successful!")
    print("\nTesting basic functionality...")

    # Test creating a simple environment
    print("  - Creating environment instance...")
    env = MARLCoverageEnvironment(
        grid_size=10,
        num_agents=2,
        max_episodes=2,
        max_steps_per_episode=10,
        device="cpu"
    )
    print("    ✓ Environment created successfully")

    # Test reset
    print("  - Testing environment reset...")
    states = env.reset(full_reset=True, mode='train')
    print(f"    ✓ Environment reset successful (got {len(states)} agent states)")

    # Test that agents were created
    print("  - Checking agents...")
    print(f"    ✓ Created {len(env.agents)} agents")

    # Test a single step
    print("  - Testing single step...")
    actions = {i: (0, 0) for i in range(len(env.agents))}
    results = env.step(actions)
    print(f"    ✓ Step executed successfully (got {len(results)} results)")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe modular structure is working correctly.")
    print("You can now run 'python main.py' to start training.")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\nPlease fix the error above before proceeding.")
