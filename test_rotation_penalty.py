#!/usr/bin/env python3
"""
Test rotation penalty calculation.
Verifies that direction changes are penalized correctly.
"""

import sys
sys.path.insert(0, 'epymarl_integration')

from coverage_env import CoverageEnv
import numpy as np

def test_rotation_penalty():
    print("=" * 60)
    print("Testing Rotation Penalty")
    print("=" * 60)

    # Create environment with rotation penalty
    env = CoverageEnv(
        n_agents=2,
        grid_size=10,
        sensor_range=2,
        reward_rotation_penalty=0.1,  # 0.1 penalty per normalized rotation
        seed=42
    )

    print(f"\nEnvironment created:")
    print(f"  Agents: {env.n_agents}")
    print(f"  Rotation penalty weight: {env.reward_rotation_penalty}")

    # Action mappings
    action_names = {
        0: "NW (315°)",
        1: "N (0°)",
        2: "NE (45°)",
        3: "W (270°)",
        4: "Stay",
        5: "E (90°)",
        6: "SW (225°)",
        7: "S (180°)",
        8: "SE (135°)",
    }

    # Test 1: No rotation (same direction)
    print("\nTest 1: No Rotation (N → N)")
    print("-" * 60)

    env.reset()
    env.previous_actions = [1, 1]  # Both agents moving N
    actions = [1, 1]  # Both continue N

    penalty = env._calculate_rotation_penalty(actions)
    print(f"  Previous: Agent 0=N, Agent 1=N")
    print(f"  Current:  Agent 0=N, Agent 1=N")
    print(f"  Angular diff: 0° for both")
    print(f"  Penalty: {penalty:.4f}")

    expected = 0.0  # No rotation
    assert abs(penalty - expected) < 0.001, f"Expected {expected}, got {penalty}"
    print(f"  ✓ Correct: {penalty:.4f} == {expected:.4f}")

    # Test 2: 45° turn (adjacent direction)
    print("\nTest 2: 45° Turn (N → NE)")
    print("-" * 60)

    env.previous_actions = [1, 1]  # Both moving N (0°)
    actions = [2, 2]  # Both turn to NE (45°)

    penalty = env._calculate_rotation_penalty(actions)
    print(f"  Previous: Agent 0=N (0°), Agent 1=N (0°)")
    print(f"  Current:  Agent 0=NE (45°), Agent 1=NE (45°)")
    print(f"  Angular diff: 45° for both")
    print(f"  Normalized: 45/180 = 0.25 each")
    print(f"  Penalty: {penalty:.4f}")

    # Expected: (0.25 + 0.25) * 0.1 = 0.05
    expected = 0.05
    assert abs(penalty - expected) < 0.001, f"Expected {expected}, got {penalty}"
    print(f"  ✓ Correct: {penalty:.4f} == {expected:.4f}")

    # Test 3: 90° turn
    print("\nTest 3: 90° Turn (N → E)")
    print("-" * 60)

    env.previous_actions = [1, 1]  # Both moving N (0°)
    actions = [5, 5]  # Both turn to E (90°)

    penalty = env._calculate_rotation_penalty(actions)
    print(f"  Previous: Agent 0=N (0°), Agent 1=N (0°)")
    print(f"  Current:  Agent 0=E (90°), Agent 1=E (90°)")
    print(f"  Angular diff: 90° for both")
    print(f"  Normalized: 90/180 = 0.5 each")
    print(f"  Penalty: {penalty:.4f}")

    # Expected: (0.5 + 0.5) * 0.1 = 0.1
    expected = 0.1
    assert abs(penalty - expected) < 0.001, f"Expected {expected}, got {penalty}"
    print(f"  ✓ Correct: {penalty:.4f} == {expected:.4f}")

    # Test 4: 180° turn (complete reversal)
    print("\nTest 4: 180° Turn (N → S)")
    print("-" * 60)

    env.previous_actions = [1, 1]  # Both moving N (0°)
    actions = [7, 7]  # Both turn to S (180°)

    penalty = env._calculate_rotation_penalty(actions)
    print(f"  Previous: Agent 0=N (0°), Agent 1=N (0°)")
    print(f"  Current:  Agent 0=S (180°), Agent 1=S (180°)")
    print(f"  Angular diff: 180° for both")
    print(f"  Normalized: 180/180 = 1.0 each")
    print(f"  Penalty: {penalty:.4f}")

    # Expected: (1.0 + 1.0) * 0.1 = 0.2
    expected = 0.2
    assert abs(penalty - expected) < 0.001, f"Expected {expected}, got {penalty}"
    print(f"  ✓ Correct: {penalty:.4f} == {expected:.4f}")

    # Test 5: Stay action (no penalty)
    print("\nTest 5: Stay Action (no penalty)")
    print("-" * 60)

    env.previous_actions = [1, 1]  # Both moving N
    actions = [4, 4]  # Both stay

    penalty = env._calculate_rotation_penalty(actions)
    print(f"  Previous: Agent 0=N, Agent 1=N")
    print(f"  Current:  Agent 0=Stay, Agent 1=Stay")
    print(f"  Penalty: {penalty:.4f}")

    expected = 0.0  # Stay has no angle, so no penalty
    assert abs(penalty - expected) < 0.001, f"Expected {expected}, got {penalty}"
    print(f"  ✓ Correct: {penalty:.4f} == {expected:.4f}")

    # Test 6: Mixed actions
    print("\nTest 6: Mixed Actions")
    print("-" * 60)

    env.previous_actions = [1, 5]  # Agent 0: N (0°), Agent 1: E (90°)
    actions = [7, 3]  # Agent 0: S (180°), Agent 1: W (270°)

    penalty = env._calculate_rotation_penalty(actions)
    print(f"  Previous: Agent 0=N (0°), Agent 1=E (90°)")
    print(f"  Current:  Agent 0=S (180°), Agent 1=W (270°)")
    print(f"  Agent 0 diff: 180° → normalized=1.0")
    print(f"  Agent 1 diff: 180° → normalized=1.0")
    print(f"  Penalty: {penalty:.4f}")

    # Expected: (1.0 + 1.0) * 0.1 = 0.2
    expected = 0.2
    assert abs(penalty - expected) < 0.001, f"Expected {expected}, got {penalty}"
    print(f"  ✓ Correct: {penalty:.4f} == {expected:.4f}")

    # Test 7: Wrap-around (315° → 45° = 90° not 270°)
    print("\nTest 7: Wrap-Around (NW → NE)")
    print("-" * 60)

    env.previous_actions = [0, 0]  # Both NW (315°)
    actions = [2, 2]  # Both NE (45°)

    penalty = env._calculate_rotation_penalty(actions)
    print(f"  Previous: Agent 0=NW (315°), Agent 1=NW (315°)")
    print(f"  Current:  Agent 0=NE (45°), Agent 1=NE (45°)")
    print(f"  Raw diff: |45-315| = 270°")
    print(f"  Wrap-around: 360-270 = 90°")
    print(f"  Normalized: 90/180 = 0.5 each")
    print(f"  Penalty: {penalty:.4f}")

    # Expected: (0.5 + 0.5) * 0.1 = 0.1
    expected = 0.1
    assert abs(penalty - expected) < 0.001, f"Expected {expected}, got {penalty}"
    print(f"  ✓ Correct: {penalty:.4f} == {expected:.4f}")

    # Test 8: Integration test (full episode)
    print("\nTest 8: Integration Test (Full Episode)")
    print("-" * 60)

    env.reset()
    total_rotation_penalty = 0.0

    # Simulate zigzag movement (alternating directions)
    action_sequence = [
        [1, 1],  # N, N
        [5, 5],  # E, E (90° turn)
        [1, 1],  # N, N (90° turn)
        [5, 5],  # E, E (90° turn)
        [1, 1],  # N, N (90° turn)
    ]

    for step, actions in enumerate(action_sequence):
        _, reward, _, _, info = env.step(actions)

        # Get penalty from info dict (calculated during step)
        penalty = info['rotation_penalty']
        total_rotation_penalty += penalty
        print(f"  Step {step}: Actions {actions}, Penalty {penalty:.4f}")

    print(f"\n  Total rotation penalty: {total_rotation_penalty:.4f}")
    print(f"  Expected: ~{0.1 * 4:.4f} (4 turns × 90° × 2 agents)")

    # Should be 4 steps with 90° turns for 2 agents
    # (0.5 + 0.5) * 0.1 = 0.1 per step × 4 steps = 0.4
    expected_range = (0.35, 0.45)
    assert expected_range[0] <= total_rotation_penalty <= expected_range[1], \
        f"Expected in range {expected_range}, got {total_rotation_penalty}"
    print(f"  ✓ Correct: within expected range")

    print("\n" + "=" * 60)
    print("✓ All rotation penalty tests passed!")
    print("=" * 60)

    print("\nSummary:")
    print("- No rotation: 0.0 penalty")
    print("- 45° turn: 0.25 normalized penalty")
    print("- 90° turn: 0.50 normalized penalty")
    print("- 180° turn: 1.0 normalized penalty (max)")
    print("- Stay action: 0.0 penalty (no angle)")
    print("- Wrap-around handled correctly")
    print("\nRotation penalty encourages smooth, consistent movement!")

    return True


if __name__ == '__main__':
    try:
        success = test_rotation_penalty()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
