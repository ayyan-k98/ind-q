# MARL Coverage System

A modular Multi-Agent Reinforcement Learning system for coverage planning using Deep Q-Networks (DQN).

## Project Structure

```
ind-q/
├── config.py              # Configuration parameters and hyperparameters
├── data_structures.py     # Data classes (RobotState, WorldState, etc.)
├── utils.py              # Helper functions (directory management, seeds, etc.)
├── replay_memory.py      # Experience replay buffer for DQN
├── networks.py           # Neural network architectures (ConvDQN, DuelingConvDQN)
├── agent.py              # MARL Coverage Agent implementation
├── visualization.py      # Visualization and plotting functions
├── environment.py        # MARL Coverage Environment (training, evaluation, map generation)
├── main.py               # Main entry point for training and evaluation
├── test_imports.py       # Test script to verify module imports
├── requirements.txt      # Python dependencies
└── __init__.py          # Package initialization
```

## Module Description

### 1. `config.py`
Contains all configuration parameters:
- Environment settings (grid size, agents, sensors)
- Training parameters (episodes, steps, learning rates)
- Reward configuration
- Network architecture settings
- Directory paths

### 2. `data_structures.py`
Defines data classes:
- `RobotState`: Agent position and orientation
- `WorldState`: Global graph and robot states
- `CoverageMetrics`: Training/evaluation metrics
- `CommunicationEvent`: Agent communication events
- `Room`: Helper class for map generation

### 3. `utils.py`
Utility functions:
- `ensure_dir()`: Safe directory creation
- `set_seeds()`: Reproducibility seed setting
- `moving_average()`: Data smoothing for visualization
- `get_latest_model_episode()`: Find latest saved model

### 4. `replay_memory.py`
Experience replay buffer implementation for DQN training.

### 5. `networks.py`
Neural network architectures:
- `ConvDQN`: Convolutional DQN for spatial processing
- `DuelingConvDQN`: Dueling architecture for better value estimation

### 6. `agent.py`
Complete agent implementation:
- Action selection (epsilon-greedy)
- Local map management
- Communication with other agents
- Model optimization and training
- Model saving/loading

### 7. `visualization.py`
Visualization functions:
- Coverage map plotting
- Agent trajectory visualization
- Local map perspective
- Learning metrics dashboard
- Animation generation for evaluation

### 8. `environment.py`
Main environment class:
- Map generation (room, cave, random, empty)
- Multi-agent coordination
- Coverage calculation
- Training loop
- Evaluation loop
- TensorBoard logging

### 9. `main.py`
Entry point that orchestrates training and evaluation using all modules.

## Installation

1. Install Python 3.8 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete training and evaluation pipeline:
```bash
python main.py
```

### Customizing Configuration

Edit `config.py` to change:
- Grid size and number of agents
- Training episodes and steps
- Reward parameters
- Network architecture
- Save directories

Example:
```python
# In config.py
GRID_SIZE = 30
NUM_AGENTS = 6
MAX_TRAINING_EPISODES = 200
```

### Using Individual Modules

You can import and use modules independently:

```python
from environment import MARLCoverageEnvironment
from agent import MARLCoverageAgent
import config

# Create custom environment
env = MARLCoverageEnvironment(
    grid_size=20,
    num_agents=4,
    device="cuda"
)

# Train
env.train(
    episodes_per_type=50,
    model_dir="./my_models"
)

# Evaluate
results = env.evaluate(
    num_eval_episodes=10,
    model_dir="./my_models"
)
```

### Testing

Verify all modules import correctly:
```bash
python test_imports.py
```

## Output

### Training Outputs
- **Models**: Saved in `MODEL_DIR` (default: `./models_coverage_dqn_final/`)
- **TensorBoard Logs**: Saved in `RUN_DIR` (default: `./runs/coverage_dqn_experiment_final/`)
- **Visualizations**: Displayed during training (configurable)

### Evaluation Outputs
- **Animations**: Saved in `ANIMATION_DIR` (default: `./eval_animations_dqn_final/`)
- **Metrics**: Printed to console (average coverage, steps, etc.)

## Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir=./runs/coverage_dqn_experiment_final
```

## Key Features

1. **Modular Design**: Each component is isolated and can be modified independently
2. **Configurable**: All parameters centralized in `config.py`
3. **Multiple Map Types**: Room, cave, and random map generation
4. **Communication**: Agents share coverage information
5. **Visualization**: Comprehensive plotting and animation capabilities
6. **TensorBoard Integration**: Real-time training monitoring
7. **Model Persistence**: Save and load trained models

## Extending the System

### Adding a New Map Generator

1. Add function to `environment.py`:
```python
def _generate_custom_map(self) -> np.ndarray:
    # Your map generation logic
    return grid
```

2. Register in `__init__`:
```python
self.training_map_generators['custom'] = self._generate_custom_map
```

3. Add to training order in `config.py` or environment init.

### Adding New Metrics

1. Add field to `CoverageMetrics` in `data_structures.py`
2. Update collection in `environment.py`
3. Add visualization in `visualization.py`

### Custom Network Architecture

1. Create new network class in `networks.py`
2. Update agent initialization in `agent.py`
3. Set flag in `config.py`

## Troubleshooting

### Import Errors
Run `python test_imports.py` to identify missing dependencies.

### CUDA Out of Memory
Reduce batch size in `config.py`:
```python
AGENT_CONFIG['batch_size'] = 64  # or lower
```

### Training Too Slow
- Reduce grid size
- Decrease number of agents
- Reduce max steps per episode
- Use GPU if available

## Citation

If you use this code in your research, please cite appropriately.

## License

[Specify your license here]
