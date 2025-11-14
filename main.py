"""
Main entry point for MARL Coverage System
Runs training and evaluation using modular components.
"""

from environment import MARLCoverageEnvironment
from utils import ensure_dir, set_seeds
import config


def main():
    """Main execution function."""
    # Set random seeds for reproducibility
    set_seeds(config.RANDOM_SEED)

    # Setup directories
    ensure_dir(config.MODEL_DIR)
    ensure_dir(config.RUN_DIR)
    if config.SAVE_EVAL_ANIMATIONS:
        ensure_dir(config.ANIMATION_DIR)

    print(f"Using device: {config.DEVICE}")
    print(f"Agent Config: {config.AGENT_CONFIG}")

    # Create environment
    env = MARLCoverageEnvironment(
        grid_size=config.GRID_SIZE,
        num_agents=config.NUM_AGENTS,
        sensor_range=config.SENSOR_RANGE,
        comm_range=config.COMM_RANGE,
        coverage_threshold=config.COVERAGE_THRESHOLD,
        completion_threshold_perc=config.COMPLETION_THRESHOLD,
        max_episodes=config.MAX_TRAINING_EPISODES,
        max_steps_per_episode=config.MAX_STEPS_PER_EPISODE,
        gamma_coverage=config.GAMMA_COVERAGE,
        step_penalty=config.STEP_PENALTY,
        orientation_cost_factor=config.ORIENTATION_COST_FACTOR,
        invalid_move_penalty=config.INVALID_MOVE_PENALTY,
        fov_degrees=config.FOV_DEGREES,
        use_dueling=config.USE_DUELING_DQN,
        device=config.DEVICE,
        tensorboard_dir=config.RUN_DIR,
        agent_config=config.AGENT_CONFIG,
        num_rooms_range=config.NUM_ROOMS_RANGE,
        room_size_range=config.ROOM_SIZE_RANGE,
        cave_fill_prob=config.CAVE_FILL_PROB,
        cave_smoothing_iterations=config.CAVE_SMOOTHING_ITERATIONS,
        cave_birth_limit=config.CAVE_BIRTH_LIMIT,
        cave_death_limit=config.CAVE_DEATH_LIMIT,
        random_obstacle_density=config.RANDOM_OBSTACLE_DENSITY
    )

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    env.train(
        episodes_per_type=config.EPISODES_PER_MAP_TYPE,
        verbose=True,
        save_interval=config.SAVE_INTERVAL,
        model_dir=config.MODEL_DIR
    )
    print("\nTraining complete.")

    # Evaluate
    print("\n" + "=" * 60)
    print("STARTING EVALUATION")
    print("=" * 60)
    eval_results = env.evaluate(
        num_eval_episodes=config.NUM_EVAL_EPISODES,
        max_steps_per_episode=config.MAX_EVAL_STEPS,
        model_dir=config.MODEL_DIR,
        visualize_each_episode=config.VISUALIZE_EVAL_FINAL,
        visualize_timesteps=config.VISUALIZE_EVAL_STEPS,
        save_animations=config.SAVE_EVAL_ANIMATIONS,
        animation_dir=config.ANIMATION_DIR
    )

    if eval_results:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION RESULTS")
        print("=" * 60)
        for key, value in eval_results.items():
            print(f"  {key}: {value:.2f}")
        print("=" * 60)
    else:
        print("Evaluation failed or produced no results.")

    print("\nScript finished successfully.")


if __name__ == "__main__":
    main()
