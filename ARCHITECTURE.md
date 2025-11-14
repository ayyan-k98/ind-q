# System Architecture

## Module Dependency Graph

```
main.py
   │
   ├─── config.py (configuration parameters)
   │
   └─── environment.py
           │
           ├─── agent.py
           │      │
           │      ├─── networks.py (ConvDQN, DuelingConvDQN)
           │      ├─── replay_memory.py (ReplayMemory)
           │      ├─── data_structures.py (RobotState, CommunicationEvent)
           │      └─── utils.py (ensure_dir)
           │
           ├─── visualization.py
           │      └─── utils.py (moving_average)
           │
           ├─── data_structures.py (WorldState, CoverageMetrics, Room)
           └─── utils.py (ensure_dir, get_latest_model_episode)
```

## Data Flow

### Training Phase
```
1. main.py initializes config parameters
2. Creates MARLCoverageEnvironment
3. Environment creates agents (MARLCoverageAgent)
4. Each agent initializes:
   - Policy network (ConvDQN or DuelingConvDQN)
   - Target network
   - Replay memory

Training Loop:
5. Environment.reset() generates map and initializes world
6. For each episode:
   a. Agents observe state (get_state_tensor)
   b. Agents select actions (epsilon-greedy)
   c. Environment.step() executes actions:
      - Resolves conflicts
      - Updates coverage via raycasting
      - Calculates rewards
      - Updates agent positions
   d. Agents store transitions in replay memory
   e. Agents optimize policy network
   f. Agents update target network
   g. Communication between nearby agents
7. Save models periodically
8. Log metrics to TensorBoard
9. Visualize results
```

### Evaluation Phase
```
1. Load trained models
2. Set epsilon = 0 (greedy policy)
3. For each evaluation episode:
   a. Reset environment with random map
   b. Run episode with trained policy
   c. Record coverage metrics
   d. Optionally visualize trajectories
   e. Generate animations for last episode
4. Report average performance
```

## Class Relationships

### MARLCoverageEnvironment
**Contains:**
- List of MARLCoverageAgent instances
- WorldState (graph + robot positions)
- Obstacle grid
- CoverageMetrics
- TensorBoard writer

**Responsibilities:**
- Map generation (room, cave, random)
- Multi-agent coordination
- Conflict resolution
- Coverage calculation via raycasting
- Training and evaluation loops
- Metrics collection

### MARLCoverageAgent
**Contains:**
- Policy network (ConvDQN/DuelingConvDQN)
- Target network
- Replay memory
- Local map (NetworkX graph)
- Communication history

**Responsibilities:**
- Action selection (epsilon-greedy)
- State representation (get_state_tensor)
- Model optimization
- Local map updates
- Communication with other agents
- Model persistence

### Networks (ConvDQN, DuelingConvDQN)
**Inputs:**
- Grid tensor [batch, channels, height, width]
- Feature tensor [batch, feature_dim]

**Outputs:**
- Q-values [batch, num_actions]

**Architecture:**
- Convolutional layers for spatial processing
- Fully connected layers for decision making
- (Dueling only) Separate value and advantage streams

## Communication Protocol

```
Agent A ←→ Agent B
   │         │
   ├─ Check distance (within comm_range?)
   │         │
   ├─ Exchange map info
   │    ├─ For each known node:
   │    │    ├─ Compare coverage probability (pc)
   │    │    └─ Update with max(local_pc, other_pc)
   │    │
   │    └─ Update node types (covered/free)
   │
   └─ Exchange position info
```

## Reward Structure

```
Total Reward = Coverage Reward - Orientation Penalty - Step Penalty

Where:
- Coverage Reward = γ_coverage × ΔCoverage
- Orientation Penalty = α_orientation × (Δθ / π)
- Step Penalty = constant negative value

Special Cases:
- Invalid Move: Large negative penalty
- Episode Complete: Normal reward (no bonus)
```

## State Representation

### Global State (Environment)
- Obstacle grid: [H, W] (0=free, 1=obstacle)
- Coverage grid: [H, W] (0 to 1 probability)
- Agent positions: List[(r, c)]
- Agent orientations: List[θ]

### Local State (Agent)
- Local map: NetworkX graph with coverage info
- Coverage grid: [H, W] derived from local map
- Obstacle grid: [H, W] from environment
- Own position: (r, c)
- Own orientation: θ
- Known other agent positions: {id: (r, c)}

### Network Input
- Grid stack: [2, H, W]
  - Channel 0: Coverage probability
  - Channel 1: Obstacles
- Features: [2]
  - sin(θ)
  - cos(θ)

## Map Generation Strategies

### Room Map
1. Generate random rooms (rectangles)
2. Ensure rooms don't overlap (with buffer)
3. Connect rooms with corridors (L-shaped paths)
4. Add border walls

### Cave Map
1. Initialize with random fill
2. Apply cellular automata:
   - Birth rule: floor becomes wall if >birth_limit neighbors are walls
   - Death rule: wall becomes floor if <death_limit neighbors are walls
3. Iterate multiple times for smoothing
4. Add border walls

### Random Map
1. Start with empty grid
2. Randomly place obstacles in interior
3. Add border walls

### Empty Map
1. All cells free
2. Border walls only

## Key Algorithms

### Raycasting Coverage Update
```
For each ray in FOV:
    1. Start at agent position
    2. Step along ray direction
    3. For each cell along ray:
       a. Check if in bounds
       b. Check if obstacle → break
       c. Calculate coverage probability:
          - Close cells: pc = 1.0
          - Far cells: pc = sigmoid decay
       d. Update cell if new pc > current pc
       e. Mark as 'covered' if pc >= threshold
```

### Conflict Resolution
```
For each target cell with multiple agents:
    1. Sort agents by ID
    2. Lowest ID wins (moves to cell)
    3. Others stay in place (action = (0,0))
```

### Epsilon Decay
```
After each step:
    ε = max(ε_end, ε × ε_decay)
```

## Performance Considerations

1. **Raycasting Cache**: Currently created but could be used to cache ray computations
2. **Spatial Index**: KDTree setup exists but not actively used
3. **Batch Processing**: Agents process steps sequentially (could parallelize)
4. **Memory Management**: Tensors detached and moved to CPU for replay memory
5. **Target Network Updates**: Soft updates every step vs hard updates periodically

## Extension Points

### Adding New Features
1. **New reward components**: Modify `perform_action()` in environment.py
2. **New map types**: Add generator method and register in `training_map_generators`
3. **New network architectures**: Create in networks.py, update agent initialization
4. **New metrics**: Add to CoverageMetrics, collect in environment, visualize
5. **Different action spaces**: Modify agent.actions and environment validation

### Integration with External Systems
- Replace map generation with real map loader
- Add ROS interface for real robot deployment
- Export trained policies to ONNX/TorchScript
- Add HTTP API for online inference
