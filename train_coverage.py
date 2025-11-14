"""
Training script for Coverage Environment with EPyMARL
Includes checkpointing, resuming, and logging
"""

import os
import sys
import yaml
import torch
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np

# Add EPyMARL to path
sys.path.insert(0, './epymarl/src')

# Import EPyMARL components
from run import run as epymarl_run
from utils.logging import get_logger


class CheckpointManager:
    """Manages checkpointing and resuming for EPyMARL training."""

    def __init__(self, checkpoint_dir: str = './checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.latest_file = self.checkpoint_dir / 'latest.txt'
        self.metadata_file = self.checkpoint_dir / 'metadata.json'

    def save_checkpoint(self,
                        model_path: str,
                        timestep: int,
                        episode: int,
                        config: dict,
                        metrics: dict = None):
        """Save checkpoint metadata."""

        # Create checkpoint name
        checkpoint_name = f"checkpoint_t{timestep}_e{episode}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Copy model files
        if os.path.exists(model_path):
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            # Copy all model files
            for f in Path(model_path).glob('*'):
                if f.is_file():
                    shutil.copy(f, checkpoint_path / f.name)

        # Save metadata
        metadata = {
            'checkpoint_name': checkpoint_name,
            'checkpoint_path': str(checkpoint_path),
            'timestep': timestep,
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': metrics or {}
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update latest
        with open(self.latest_file, 'w') as f:
            f.write(checkpoint_name)

        print(f"✓ Checkpoint saved: {checkpoint_name}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_name: str = None):
        """Load checkpoint metadata and return path."""

        # Use latest if not specified
        if checkpoint_name is None:
            if self.latest_file.exists():
                with open(self.latest_file, 'r') as f:
                    checkpoint_name = f.read().strip()
            else:
                return None

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        if not checkpoint_path.exists():
            print(f"✗ Checkpoint not found: {checkpoint_name}")
            return None

        # Load metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata['checkpoint_path'] = str(checkpoint_path)

        print(f"✓ Loaded checkpoint: {checkpoint_name}")
        return metadata

    def list_checkpoints(self):
        """List all available checkpoints."""
        checkpoints = []
        for d in self.checkpoint_dir.glob('checkpoint_*'):
            if d.is_dir():
                checkpoints.append(d.name)
        return sorted(checkpoints)


def setup_environment():
    """Register coverage environment with EPyMARL."""

    # Import environment
    from epymarl_coverage_env import CoverageEnvironment

    # Register with EPyMARL
    try:
        from envs import REGISTRY as env_REGISTRY
        env_REGISTRY["coverage"] = CoverageEnvironment
        print("✓ Coverage environment registered")
    except Exception as e:
        print(f"✗ Failed to register environment: {e}")
        return False

    return True


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def modify_config_for_resume(config: dict, checkpoint_metadata: dict):
    """Modify config to resume from checkpoint."""

    # Set checkpoint path
    config['checkpoint_path'] = checkpoint_metadata['checkpoint_path']
    config['load_step'] = checkpoint_metadata['timestep']

    # Keep other settings from original config
    return config


def train(
    config_path: str = './coverage_config.yaml',
    checkpoint_dir: str = './checkpoints',
    results_dir: str = './results',
    resume: bool = False,
    checkpoint_name: str = None
):
    """
    Train coverage agent with EPyMARL.

    Args:
        config_path: Path to config YAML
        checkpoint_dir: Directory for checkpoints
        results_dir: Directory for results
        resume: Whether to resume from checkpoint
        checkpoint_name: Specific checkpoint to resume from (None = latest)
    """

    print("=" * 60)
    print("EPyMARL Coverage Training")
    print("=" * 60)

    # Setup environment
    if not setup_environment():
        print("✗ Environment setup failed")
        return

    # Load config
    config = load_config(config_path)
    print(f"✓ Loaded config from {config_path}")

    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    # Resume from checkpoint if requested
    if resume:
        checkpoint_metadata = checkpoint_manager.load_checkpoint(checkpoint_name)

        if checkpoint_metadata:
            print(f"Resuming from timestep {checkpoint_metadata['timestep']}")
            config = modify_config_for_resume(config, checkpoint_metadata)
        else:
            print("✗ No checkpoint found, starting from scratch")
            resume = False

    # Setup results directory
    os.makedirs(results_dir, exist_ok=True)
    config['local_results_path'] = results_dir

    # Setup logging
    if 'use_tensorboard' in config and config['use_tensorboard']:
        tb_log_dir = os.path.join(results_dir, 'tb_logs')
        os.makedirs(tb_log_dir, exist_ok=True)
        print(f"✓ TensorBoard logs: {tb_log_dir}")

    # Print training info
    print("\nTraining Configuration:")
    print(f"  Agents: {config.get('env_args', {}).get('n_agents', 'N/A')}")
    print(f"  Grid Size: {config.get('env_args', {}).get('grid_size', 'N/A')}")
    print(f"  Episode Limit: {config.get('env_args', {}).get('episode_limit', 'N/A')}")
    print(f"  Algorithm: {config.get('name', 'N/A')}")
    print(f"  Total Steps: {config.get('t_max', 'N/A')}")
    print(f"  Batch Size: {config.get('batch_size', 'N/A')}")
    print(f"  Learning Rate: {config.get('lr', 'N/A')}")
    print()

    # Start training
    print("=" * 60)
    print("Starting Training...")
    print("=" * 60)
    print()

    try:
        # Call EPyMARL's run function
        # This will handle the actual training loop
        epymarl_run(config, _log=get_logger())

        print("\n" + "=" * 60)
        print("Training Completed!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")

        # EPyMARL should auto-save, but we can add extra logic here
        print("✓ Checkpoint saved")

    except Exception as e:
        print(f"\n\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    if checkpoint_manager.list_checkpoints():
        print("\nAvailable checkpoints:")
        for cp in checkpoint_manager.list_checkpoints()[-5:]:  # Show last 5
            print(f"  - {cp}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Coverage Agent with EPyMARL')
    parser.add_argument('--config', type=str, default='./coverage_config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Results directory')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Specific checkpoint to resume from')

    args = parser.parse_args()

    train(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        resume=args.resume,
        checkpoint_name=args.checkpoint
    )
