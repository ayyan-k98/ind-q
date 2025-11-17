# Visualization and Debugging Guide

Comprehensive tools for visualizing and debugging the coverage environment.

## Quick Start

```python
# Install matplotlib if not already installed
pip install matplotlib

# Import visualization tools
from visualization import CoverageVisualizer, debug_episode, random_policy
from epymarl.src.envs.coverage import CoverageEnv

# Create environment
env = CoverageEnv(n_agents=4, grid_size=20)

# Run debug episode
stats = debug_episode(env, random_policy, max_steps=200, save_dir='debug_output')
```

This generates:
- Coverage heatmap
- Agent trajectories
- Coverage over time graph
- Action distribution plots
- Reward breakdown
- Sensor FOV for each agent
- Summary statistics JSON

---

## Visualization Tools

### 1. Coverage Heatmap

Shows coverage intensity across the grid.

```python
vis = CoverageVisualizer(env)

# After running some steps
env.reset()
for _ in range(50):
    actions = [0, 1, 2, 3]  # Example actions
    env.step(actions)

# Plot heatmap
vis.plot_coverage_heatmap(save_path='coverage.png')
```

**Features:**
- Color-coded coverage intensity (0.0 to 1.0)
- Agent positions marked with blue circles
- Obstacles shown in gray
- Current coverage percentage in title

---

### 2. Sensor Field of View (FOV)

Visualize what each agent can see with their sensors.

```python
vis = CoverageVisualizer(env)

# Visualize agent 0's sensor
vis.plot_sensor_fov(agent_id=0, save_path='sensor_fov.png')
```

**Features:**
- Yellow wedge shows sensor FOV
- Green boxes show cells being raycasted
- Red arrow shows agent orientation
- Sensor range and FOV angle in title

**Use cases:**
- Debug raycasting logic
- Verify FOV coverage
- Check sensor overlap between agents
- Understand coverage patterns

---

### 3. Agent Trajectories

Plot paths taken by all agents over an episode.

```python
vis = CoverageVisualizer(env)

# Run episode and record
env.reset()
for _ in range(100):
    actions = [random.randint(0, 8) for _ in range(env.n_agents)]
    reward, done, info = env.step(actions)
    vis.record_step(actions, reward, info)
    if done:
        break

# Plot trajectories
vis.plot_trajectory(save_path='trajectories.png')
```

**Features:**
- Different color for each agent
- Circles mark start positions
- Squares mark end positions
- Coverage heatmap in background
- Obstacles shown

**Use cases:**
- Verify agent exploration patterns
- Check for redundant coverage
- Identify coordination issues
- Debug movement behavior

---

### 4. Coverage Over Time

Track coverage percentage throughout the episode.

```python
vis = CoverageVisualizer(env)

# Record episode (same as above)
env.reset()
for _ in range(100):
    actions = [...]
    reward, done, info = env.step(actions)
    vis.record_step(actions, reward, info)
    if done:
        break

# Plot coverage growth
vis.plot_coverage_over_time(save_path='coverage_time.png')
```

**Features:**
- Line graph showing coverage % vs steps
- Shaded area under curve
- Grid for easy reading
- 0-100% Y-axis scale

**Use cases:**
- Monitor coverage efficiency
- Identify plateaus or slowdowns
- Compare different policies
- Verify expected coverage growth

---

### 5. Action Distribution

Analyze which actions agents are taking.

```python
vis = CoverageVisualizer(env)

# Record episode with actions
env.reset()
for _ in range(100):
    actions = [...]
    reward, done, info = env.step(actions)
    vis.record_step(actions, reward, info)
    if done:
        break

# Plot action distribution
vis.plot_action_distribution(save_path='actions.png')
```

**Features:**
- Bar chart for each agent
- 9 actions: NW, N, NE, W, Stay, E, SW, S, SE
- Count of each action type

**Use cases:**
- Detect biases in policy (e.g., always moving right)
- Check action masking effectiveness
- Verify agents aren't stuck (too many "Stay")
- Compare exploration strategies

---

### 6. Reward Breakdown

Visualize reward components over time.

```python
vis = CoverageVisualizer(env)

# Record episode
env.reset()
for _ in range(100):
    actions = [...]
    reward, done, info = env.step(actions)
    vis.record_step(actions, reward, info)
    if done:
        break

# Plot rewards
vis.plot_reward_breakdown(save_path='rewards.png')
```

**Features:**
- Total reward (blue line)
- Coverage reward (green dashed line)
- Shows reward trends over episode

**Use cases:**
- Debug reward calculation
- Identify reward sparsity issues
- Verify reward shaping effectiveness
- Track learning progress

---

## Complete Episode Summary

Generate all visualizations at once.

```python
vis = CoverageVisualizer(env)

# Run episode
env.reset()
for _ in range(200):
    actions = [policy(env, i) for i in range(env.n_agents)]
    reward, done, info = env.step(actions)
    vis.record_step(actions, reward, info)
    if done:
        break

# Create comprehensive summary
vis.create_episode_summary(save_dir='episode_summary')
```

This creates a directory with:
```
episode_summary/
├── coverage_heatmap.png
├── trajectories.png
├── coverage_over_time.png
├── action_distribution.png
├── rewards.png
├── sensor_fov_agent_0.png
├── sensor_fov_agent_1.png
├── sensor_fov_agent_2.png
├── sensor_fov_agent_3.png
└── summary.json
```

**summary.json** contains:
```json
{
  "episode_length": 185,
  "final_coverage": 87.3,
  "total_reward": 245.6,
  "n_agents": 4,
  "grid_size": 20
}
```

---

## Debug Episode Function

High-level function for full debugging workflow.

```python
from visualization import debug_episode, greedy_policy, random_policy

# Run debug episode with random policy
stats = debug_episode(
    env,
    policy_fn=random_policy,
    max_steps=200,
    save_dir='debug_random'
)

# Or use greedy policy
stats = debug_episode(
    env,
    policy_fn=greedy_policy,
    max_steps=200,
    save_dir='debug_greedy'
)

print(f"Coverage: {stats['coverage']:.1f}%")
print(f"Reward: {stats['reward']:.2f}")
print(f"Steps: {stats['steps']}")
```

**Custom policy:**
```python
def my_policy(env, agent_id):
    """Your custom policy."""
    # Get available actions
    avail = env.get_avail_agent_actions(agent_id)

    # Your logic here
    action = ...

    return action

stats = debug_episode(env, my_policy, save_dir='debug_custom')
```

---

## Use Cases

### 1. Debug Coverage Issues

```python
# Check if agents are actually covering cells
env = CoverageEnv()
vis = CoverageVisualizer(env)

env.reset()
for i in range(10):
    actions = [2] * env.n_agents  # All move NE
    env.step(actions)

# Should show increasing coverage
vis.plot_coverage_heatmap()
```

### 2. Verify Sensor Logic

```python
# Test sensor FOV and raycasting
env = CoverageEnv()
vis = CoverageVisualizer(env)

env.reset()

# Check each agent's sensor
for agent_id in range(env.n_agents):
    vis.plot_sensor_fov(agent_id)
```

### 3. Compare Policies

```python
# Test random vs greedy
env = CoverageEnv()

stats_random = debug_episode(env, random_policy, save_dir='compare/random')
stats_greedy = debug_episode(env, greedy_policy, save_dir='compare/greedy')

print(f"Random: {stats_random['coverage']:.1f}%")
print(f"Greedy: {stats_greedy['coverage']:.1f}%")
```

### 4. Analyze Trained Model

```python
# Load trained QMIX model and visualize
def qmix_policy(env, agent_id):
    # Load your trained model
    # Get action from model
    return action

stats = debug_episode(env, qmix_policy, save_dir='trained_model_debug')
```

### 5. Check Coordination

```python
# Verify agents aren't overlapping too much
env = CoverageEnv(n_agents=4)
vis = CoverageVisualizer(env)

env.reset()
for _ in range(100):
    actions = [policy(env, i) for i in range(env.n_agents)]
    reward, done, info = env.step(actions)
    vis.record_step(actions, reward, info)

# Trajectories will show if agents are exploring different areas
vis.plot_trajectory()
```

---

## Integration with EPyMARL

Add visualization to EPyMARL training:

```python
# In epymarl/src/runners/episode_runner.py

from visualization import CoverageVisualizer

class EpisodeRunner:
    def __init__(self, ...):
        # ... existing code ...
        self.visualizer = CoverageVisualizer(self.env) if args.visualize else None

    def run(self, ...):
        # ... existing code ...

        # After episode ends
        if self.visualizer and episode_num % 100 == 0:
            self.visualizer.create_episode_summary(
                save_dir=f'visualizations/episode_{episode_num}'
            )
            self.visualizer.reset()
```

---

## Troubleshooting

### "No trajectory history recorded"

Make sure to call `vis.record_step()` after each environment step:

```python
vis = CoverageVisualizer(env)
env.reset()

for _ in range(100):
    actions = [...]
    reward, done, info = env.step(actions)
    vis.record_step(actions, reward, info)  # <-- Don't forget this!
```

### Plots not showing

If running in script, add `plt.show()` or use `show=True`:

```python
vis.plot_coverage_heatmap(show=True)
```

In Jupyter, plots should show automatically.

### Out of memory with large episodes

Limit recording frequency:

```python
if step % 10 == 0:  # Record every 10 steps
    vis.record_step(actions, reward, info)
```

---

## API Reference

### CoverageVisualizer

**Methods:**
- `plot_coverage_heatmap(save_path=None, show=True)` - Plot coverage intensity
- `plot_sensor_fov(agent_id, save_path=None, show=True)` - Visualize sensor FOV
- `plot_trajectory(save_path=None, show=True)` - Plot agent paths
- `plot_coverage_over_time(save_path=None, show=True)` - Coverage growth graph
- `plot_action_distribution(save_path=None, show=True)` - Action histogram
- `plot_reward_breakdown(save_path=None, show=True)` - Reward components
- `create_episode_summary(save_dir)` - Generate all visualizations
- `record_step(actions, reward, reward_info)` - Record step for history
- `reset()` - Clear all history

### Utility Functions

- `debug_episode(env, policy_fn, max_steps, save_dir)` - Run full debug episode
- `greedy_policy(env, agent_id)` - Simple greedy baseline
- `random_policy(env, agent_id)` - Random action baseline

---

## Examples

See `visualization.py` for complete examples. Run standalone:

```bash
python visualization.py
```

This runs a debug episode and saves all visualizations to `debug_output/`.

---

**Happy debugging!** 🎨
