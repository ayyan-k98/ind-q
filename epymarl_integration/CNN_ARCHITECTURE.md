# CNN + Coordinate Normalization Architecture

## Motivation

**Scale Invariance Problem:**
- Training on 20×20 grid, testing on 30×30 grid fails with flat observations
- Absolute positions break: agent at (10,10) on 20×20 vs (15,15) on 30×30
- Need architecture that generalizes across grid sizes

**Solution:**
1. **Coordinate Normalization**: All positions normalized by grid_size
2. **CNN Spatial Processing**: Extract features from coverage/belief grids
3. **Adaptive Pooling**: Handle variable grid sizes

## Observation Structure

### Current (Flat Vector - 64 dims)
```python
obs = [
    pos_x/grid_size, pos_y/grid_size,           # [2] Own position (normalized)
    sin(orientation), cos(orientation),         # [2] Orientation
    timestep/episode_limit,                     # [1] Time
    uncovered_ratio, covered_ratio,             # [2] Sensor info
    frontier_dist/grid_size, sin(angle), cos(angle),  # [3] Frontier
    ... other agents (40 dims) ...,             # [40] Relative positions
    ... global stats (3 dims) ...               # [3] Coverage stats
]
```

**Problem:** No spatial structure, can't use CNNs

### New (Spatial + Scalar)

**Spatial Grids** (C × H × W channels):
```python
spatial = {
    'coverage_grid':      (1 × H × W)  # Coverage values 0-1
    'obstacle_belief':    (1 × H × W)  # -1=unknown, 0=free, 1=obstacle (normalized)
    'agent_positions':    (1 × H × W)  # All agents as gaussian kernels
    'own_position':       (1 × H × W)  # This agent's position highlight
}
# Total: 4 channels × H × W
```

**Scalar Features** (kept separate):
```python
scalars = [
    sin(orientation), cos(orientation),         # [2] Orientation (rotation invariant)
    timestep/episode_limit,                     # [1] Normalized time

    # Other agents (coordinate normalized)
    for each agent:
        dx/grid_size, dy/grid_size,             # Relative position (normalized)
        distance/grid_size,                     # Distance (normalized)
        sin(angle), cos(angle),                 # Relative angle (rotation invariant)

    # Global statistics (already normalized)
    coverage_percentage/100,                    # [1]
    coverage_sum/(grid_size²),                  # [1]
    coverage_delta,                             # [1]
]
# Total: ~50 scalars
```

## Network Architecture

### CNN Feature Extractor
```python
class CoverageCNN(nn.Module):
    def __init__(self, grid_size, n_agents):
        # Input: 4 channels × H × W
        self.conv1 = Conv2d(4, 32, kernel_size=3, padding=1)  # → 32 × H × W
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1) # → 64 × H × W
        self.conv3 = Conv2d(64, 64, kernel_size=3, padding=1) # → 64 × H × W

        # Adaptive pooling to fixed size (handles variable grid sizes!)
        self.adaptive_pool = AdaptiveAvgPool2d((4, 4))       # → 64 × 4 × 4

        # Flatten
        self.spatial_features = 64 * 4 * 4 = 1024

    def forward(self, spatial_obs):
        x = F.relu(self.conv1(spatial_obs))    # (B, 32, H, W)
        x = F.relu(self.conv2(x))              # (B, 64, H, W)
        x = F.relu(self.conv3(x))              # (B, 64, H, W)
        x = self.adaptive_pool(x)              # (B, 64, 4, 4)
        x = x.view(-1, 1024)                   # (B, 1024)
        return x
```

### Combined Agent Network
```python
class CNNAgent(nn.Module):
    def __init__(self, grid_size, n_agents, n_actions):
        self.cnn = CoverageCNN(grid_size, n_agents)

        # Combine spatial features + scalars
        self.fc1 = Linear(1024 + 50, 256)     # Spatial + scalars
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, n_actions)

    def forward(self, spatial_obs, scalar_obs):
        spatial_features = self.cnn(spatial_obs)      # (B, 1024)
        combined = torch.cat([spatial_features, scalar_obs], dim=1)  # (B, 1074)

        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values
```

## Scale Invariance Properties

### Why This Works

**1. Coordinate Normalization:**
```python
# Training on 20×20:
pos = (10, 10)
normalized = (10/20, 10/20) = (0.5, 0.5)  # Center

# Testing on 30×30:
pos = (15, 15)
normalized = (15/30, 15/30) = (0.5, 0.5)  # Still center!
```

**2. Convolutional Features:**
- CNNs learn local patterns (walls, frontiers, agent clusters)
- Local patterns are grid-size invariant
- Edge detection works on any size grid

**3. Adaptive Pooling:**
```python
# Training: 20×20 → pool to 4×4
# Testing: 30×30 → pool to 4×4
# Same fixed-size output regardless of input size!
```

## Coordinate Normalization Details

### All Normalized Features
```python
# Positions (always divide by grid_size)
pos_x_norm = pos_x / grid_size          # [0, 1]
pos_y_norm = pos_y / grid_size          # [0, 1]

# Distances (always divide by grid_size)
distance_norm = distance / grid_size    # [0, √2]

# Relative positions
dx_norm = (other_x - self_x) / grid_size   # [-1, 1]
dy_norm = (other_y - self_y) / grid_size   # [-1, 1]

# Frontier distance
frontier_norm = frontier_dist / grid_size  # [0, ~2]

# Angles (already rotation invariant)
sin(angle), cos(angle)  # [-1, 1]

# Coverage (already normalized)
coverage_pct / 100                         # [0, 1]
coverage_sum / (grid_size * grid_size)     # [0, 1]
```

### Never Use Absolute Values
❌ Bad (not scale invariant):
```python
pos_x  # Absolute position (breaks on different grid sizes)
distance  # Absolute distance
grid_size  # Reveals training size
```

✅ Good (scale invariant):
```python
pos_x / grid_size     # Relative position
distance / grid_size  # Normalized distance
1.0                   # Grid size implicit in normalization
```

## Implementation Steps

### Step 1: Modify Environment Observations
```python
def get_obs_agent_spatial(self, agent_id):
    """Return spatial observation for CNN."""
    # Spatial grids
    spatial = np.stack([
        self.coverage_grid,
        (self.obstacle_belief + 1) / 2,  # Normalize -1/0/1 to 0/0.5/1
        self._get_agent_position_grid(),
        self._get_own_position_grid(agent_id),
    ], axis=0)  # (4, H, W)

    # Scalar features (all normalized)
    scalars = self._get_scalar_features(agent_id)

    return {'spatial': spatial, 'scalars': scalars}
```

### Step 2: Create CNN Agent in EPyMARL
- Subclass `BasicMAC` with CNN agent
- Handle spatial observations in agent forward pass
- Keep QMIX mixer unchanged (operates on Q-values)

### Step 3: Configuration
```yaml
# coverage_cnn.yaml
agent: "cnn"  # Use CNN agent instead of RNN
agent_cnn_channels: [32, 64, 64]
agent_cnn_pooled_size: [4, 4]
agent_hidden_dim: 256
```

### Step 4: Scale Invariance Testing
```bash
# Train on 20×20
python src/main.py --config=qmix_cnn --env-config=coverage_small

# Test on 30×30 (zero-shot transfer)
python src/main.py --config=qmix_cnn --env-config=coverage_large \
    --checkpoint_path=models/coverage_small --evaluate=True
```

## Expected Performance

### Baseline (Flat MLP)
- Train 20×20: 85-95% coverage
- Test 30×30: **40-60% coverage** (fails to generalize)

### CNN + Coordinate Normalization
- Train 20×20: 85-95% coverage
- Test 30×30: **75-90% coverage** (graceful degradation)

**Note:** Perfect transfer is impossible (30×30 has 2.25× area, different obstacle patterns), but CNN should maintain reasonable performance.

## Advantages

### 1. Scale Invariance
- Works across grid sizes
- Coordinate normalization prevents absolute position encoding
- Adaptive pooling handles variable input dimensions

### 2. Spatial Reasoning
- CNN captures local patterns
- Coverage frontiers
- Obstacle configurations
- Agent clustering

### 3. Sample Efficiency
- Convolutional parameter sharing
- Fewer parameters than fully connected
- Learns reusable spatial features

### 4. Interpretability
- Can visualize conv filters (what patterns agents look for)
- Attention maps show where agents focus
- Feature maps reveal learned representations

## Next Steps

1. ✅ Design observation structure (this document)
2. ⏳ Implement `get_obs_agent_spatial()` in coverage_env.py
3. ⏳ Create CNN agent architecture for EPyMARL
4. ⏳ Integrate with QMIX (modify MAC)
5. ⏳ Add configuration files
6. ⏳ Train and test scale invariance
7. ⏳ Visualize learned features

## References

- **Spatial Invariance**: "Spatial Invariance in Multi-Agent Reinforcement Learning" (Anonymous submission concept)
- **Coordinate Normalization**: Common practice in robotics/navigation
- **Adaptive Pooling**: PyTorch AdaptiveAvgPool2d documentation
- **QMIX**: Rashid et al., 2018 (value decomposition independent of agent representation)
