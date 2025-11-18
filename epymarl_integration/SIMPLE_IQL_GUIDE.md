# Simple IQL + CNN Guide

## What is IQL?

**IQL (Independent Q-Learning)** is the simplest multi-agent RL approach:

- Each agent learns its own Q-function
- **Parameter sharing**: All agents use the **same network weights**
- No coordination (independent learning)
- Much simpler than QMIX (no mixing network)

### Parameter Sharing

```
Agent 1 → CNN Network (shared) → Q-values for Agent 1
Agent 2 → CNN Network (shared) → Q-values for Agent 2
Agent 3 → CNN Network (shared) → Q-values for Agent 3
Agent 4 → CNN Network (shared) → Q-values for Agent 4

All agents use the SAME network!
```

**Benefits:**
- ✅ Simple to implement
- ✅ Sample efficient (4 agents = 4× data for same network)
- ✅ Works well for homogeneous agents
- ✅ Scale invariant with CNN + coordinate normalization

**Limitations:**
- ❌ No explicit coordination
- ❌ Agents treated as part of environment (non-stationarity)
- ❌ May converge to suboptimal equilibria

## Architecture

### Simple IQL Flow

```
For each agent i:
  1. Get observation: obs_i = env.get_obs_agent(i)
  2. Forward through CNN: q_values_i = cnn_agent(obs_i)
  3. Select action: a_i = argmax(q_values_i) or ε-greedy
  4. Execute all actions: env.step([a_1, a_2, a_3, a_4])
  5. Update: Q(obs_i, a_i) ← r + γ max Q(obs'_i, a')
```

**Parameter sharing means:**
- Same CNN weights for all agents
- Updates from all agents improve the shared network
- 4 agents = 4× faster learning!

### Network Architecture

```
Observation (agent i):
  Spatial: (4, H, W)  - coverage, belief, agents, own_position
  Scalars: (56,)      - orientation, other agents, stats
    ↓
CNN Feature Extractor (SHARED across all agents):
  Conv(4→32, k=3) + ReLU
  Conv(32→64, k=3) + ReLU
  Conv(64→64, k=3) + ReLU
  AdaptivePool((4,4))
    ↓
  Spatial features: (1024,)
    ↓
Concatenate with scalars: (1080,)
    ↓
MLP (SHARED):
  FC(1080→256) + ReLU
  FC(256→256) + ReLU
  FC(256→9)  → Q-values for agent i
```

## Configuration Files

### Algorithm: `iql_cnn.yaml`

```yaml
mac: "cnn_mac"

# CNN Architecture
cnn_hidden_dim: 256
cnn_channels: [32, 64, 64]
cnn_pooled_size: [4, 4]

# IQL (no mixer)
learner: "q_learner"

# Hyperparameters
gamma: 0.99
lr: 0.0005
batch_size: 32
buffer_size: 5000

# Exploration
epsilon_start: 1.0
epsilon_finish: 0.05
epsilon_anneal_time: 50000
```

### Environment: `coverage_small.yaml`

```yaml
env: coverage

env_args:
  grid_size: 20
  n_agents: 4
  sensor_range: 2
  use_spatial_obs: True  # IMPORTANT: Enable CNN observations
  map_type: random

  # Reward scaling
  reward_scale_coverage: 0.5
  reward_scale_shaping: 0.1
```

## Training

### Install EPyMARL

```bash
# Clone EPyMARL
git clone https://github.com/uoe-agents/epymarl.git
cd epymarl
pip install -e .
```

### Copy Files

```bash
# From ind-q/epymarl_integration/ directory

# Copy CNN agent
cp cnn_agent.py ../epymarl/src/modules/agents/

# Copy CNN MAC
cp cnn_mac.py ../epymarl/src/controllers/

# Copy configs
cp iql_cnn.yaml ../epymarl/src/config/algs/
cp coverage_small.yaml ../epymarl/src/config/envs/
cp coverage_large.yaml ../epymarl/src/config/envs/

# Copy coverage environment
cp coverage_env.py ../epymarl/src/envs/coverage/
cp coverage__init__.py ../epymarl/src/envs/coverage/__init__.py
cp multiagentenv.py ../epymarl/src/envs/
```

### Register Components

**1. Register CNN MAC** - Edit `epymarl/src/controllers/__init__.py`:

```python
from .basic_controller import BasicMAC
from .cnn_controller import CNNMAC  # Add this

REGISTRY = {}
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["cnn_mac"] = CNNMAC  # Add this
```

**2. Register CNN Agent** - Edit `epymarl/src/modules/agents/__init__.py`:

```python
from .rnn_agent import RNNAgent
from .cnn_agent import CNNAgent  # Add this

REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["cnn"] = CNNAgent  # Add this
```

**3. Register Coverage Environment** - Edit `epymarl/src/envs/__init__.py`:

```python
from .multiagentenv import MultiAgentEnv
from .coverage import CoverageEnv  # Add this

REGISTRY = {}
REGISTRY["coverage"] = CoverageEnv  # Add this

def env_REGISTRY(env_name):
    if env_name in REGISTRY:
        return REGISTRY[env_name]
    else:
        raise ValueError(f"Unknown environment: {env_name}")
```

### Run Training

```bash
cd epymarl

# Train IQL with CNN on 20×20 grid
python src/main.py \
    --config=iql_cnn \
    --env-config=coverage_small \
    with use_tensorboard=True

# Monitor with TensorBoard
tensorboard --logdir results/tb_logs
```

**Expected training time:** 4-6 hours on GPU for 1M timesteps

**Expected performance:** 85-95% coverage on 20×20 grid

### Test Scale Invariance

After training on 20×20, test on 30×30:

```bash
# Find your checkpoint
ls results/models/

# Test on 30×30 grid (zero-shot transfer)
python src/main.py \
    --config=iql_cnn \
    --env-config=coverage_large \
    --checkpoint_path=results/models/<run>/<timestep> \
    --evaluate=True \
    with test_nepisode=20
```

**Expected performance:** 70-85% coverage on 30×30 grid (graceful degradation)

## Results Interpretation

### Training Curves (TensorBoard)

**Monitor these metrics:**

1. **Episode Return** - Should increase over time
   - Early: 2-5 (random exploration)
   - Mid: 10-20 (learning coordination)
   - Late: 20-30 (good coverage)

2. **Coverage Percentage** - Main metric
   - Early: 40-60% (random)
   - Mid: 70-85% (learning)
   - Late: 85-95% (converged)

3. **Episode Length** - Should decrease
   - Early: 200 steps (hitting episode limit)
   - Late: 70-100 steps (efficient coverage)

4. **Loss** - Should stabilize
   - Should NOT be NaN (if NaN, reduce learning rate)
   - Typical range: 0.1-1.0

5. **Grad Norm** - Should be reasonable
   - Typical: 1-10
   - If >50: gradients exploding (add clipping)

### Expected Timeline

| Timesteps | Coverage | Notes |
|-----------|----------|-------|
| 0-100K | 40-60% | Exploration phase |
| 100K-300K | 60-75% | Learning coordination |
| 300K-600K | 75-85% | Refinement |
| 600K-1M | 85-95% | Converged |

## Debugging

### Issue 1: NaN Loss

**Symptom:** Loss becomes NaN during training

**Solutions:**
```yaml
# In coverage_small.yaml
env_args:
  reward_scale_coverage: 0.5  # Reduce if still NaN
  reward_scale_shaping: 0.1

# In iql_cnn.yaml
lr: 0.0001  # Reduce learning rate
grad_norm_clip: 10.0  # Add gradient clipping
```

### Issue 2: Low Coverage (<70%)

**Symptom:** Agents don't learn to cover the grid

**Solutions:**
```yaml
# Increase exploration
epsilon_anneal_time: 100000  # Explore longer

# Increase sensor range (easier task)
env_args:
  sensor_range: 3  # Was 2

# Reduce map difficulty
env_args:
  map_type: empty  # Was 'random'
```

### Issue 3: Agents Cluster Together

**Symptom:** All agents stay in same area

**Solutions:**
```yaml
# Increase spread bonus
env_args:
  reward_spread_bonus: 0.5  # Was 0.1

# Decrease step penalty (encourage exploration)
env_args:
  reward_step_penalty: 0.05  # Was 0.1
```

### Issue 4: Slow Training

**Symptom:** Training takes >12 hours

**Solutions:**
```yaml
# Reduce CNN size
cnn_channels: [16, 32, 32]  # Was [32, 64, 64]
cnn_hidden_dim: 128  # Was 256

# Reduce batch size
batch_size: 16  # Was 32

# Reduce timesteps
t_max: 500000  # Was 1000000 (for quick testing)
```

## Comparison: IQL vs QMIX

| Aspect | IQL | QMIX |
|--------|-----|------|
| **Complexity** | ✅ Simple | ❌ Complex (mixer network) |
| **Parameter Sharing** | ✅ Yes | ✅ Yes |
| **Coordination** | ❌ Implicit | ✅ Explicit (mixing) |
| **Training Speed** | ✅ Fast | ⏸️ Slower (more parameters) |
| **Performance** | ⏸️ Good | ✅ Better (with coordination) |
| **Debugging** | ✅ Easy | ❌ Harder |

**Recommendation:** Start with IQL, then try QMIX if you need better coordination.

## Next Steps

After getting IQL working:

1. **Visualize behavior** - See how agents explore
2. **Test scale invariance** - 20×20 → 30×30 transfer
3. **Compare baselines** - IQL with MLP (no CNN)
4. **Ablation studies** - Remove coordinate normalization
5. **Try QMIX** - Better coordination (qmix_cnn.yaml)

## Parameter Sharing Details

### How Parameter Sharing Works

**Without parameter sharing:**
```python
agent_1 = CNNAgent()  # Network 1 (separate)
agent_2 = CNNAgent()  # Network 2 (separate)
agent_3 = CNNAgent()  # Network 3 (separate)
agent_4 = CNNAgent()  # Network 4 (separate)

# 4× parameters, 4× memory, slower learning
```

**With parameter sharing (IQL):**
```python
shared_agent = CNNAgent()  # ONE network for all

# All agents use the same network:
q_values_1 = shared_agent(obs_1)
q_values_2 = shared_agent(obs_2)
q_values_3 = shared_agent(obs_3)
q_values_4 = shared_agent(obs_4)

# 1× parameters, 4× data, faster learning!
```

### Why Parameter Sharing Works

**For homogeneous agents** (all agents are identical):
- Same capabilities (sensor range, actions)
- Same objective (maximize coverage)
- Observations include agent-specific info (own position, other agents)
- Network learns "what should I do given my position?"

**Key:** Each agent gets a **different observation** (own position highlighted), so they act differently even with shared network!

### Observation Includes Agent Identity

```python
# Agent 1's observation:
spatial[3] = [1 at position (5,5), 0 elsewhere]  # Own position grid

# Agent 2's observation:
spatial[3] = [1 at position (8,12), 0 elsewhere]  # Own position grid

# Same network, different observations → different actions!
```

## Simple Testing Script

Create `test_iql.py`:

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, 'epymarl_integration')

from coverage_env import CoverageEnv
from cnn_agent import CNNAgent
import torch
import numpy as np

# Create environment
env = CoverageEnv(
    n_agents=4,
    grid_size=20,
    sensor_range=2,
    use_spatial_obs=True
)

# Create shared agent
agent = CNNAgent(
    input_channels=4,
    scalar_size=56,
    hidden_dim=256,
    n_actions=9
)

print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

# Run one episode
obs_list = env.reset()[0]
done = False
step = 0

while not done and step < 50:
    actions = []

    # Each agent uses the SAME network
    for obs_dict in obs_list:
        spatial = torch.FloatTensor(obs_dict['spatial']).unsqueeze(0)
        scalars = torch.FloatTensor(obs_dict['scalars']).unsqueeze(0)

        # Forward through shared agent
        with torch.no_grad():
            q_values = agent(spatial, scalars)

        # Greedy action
        action = q_values.argmax().item()
        actions.append(action)

    # Step environment
    _, reward, done, _, info = env.step(actions)
    obs_list = env.get_obs()

    step += 1
    print(f"Step {step}: Coverage {info['coverage_pct']:.1f}%")

print(f"\nFinal coverage: {info['coverage_pct']:.1f}%")
print("✓ IQL with parameter sharing works!")
```

Run: `python test_iql.py`

## Summary

**IQL + CNN is the simplest approach:**
- ✅ One shared network for all agents
- ✅ Scale invariant (coordinate normalization + adaptive pooling)
- ✅ POMDP compatible (agents explore unknown map)
- ✅ Easy to implement and debug
- ✅ Fast training with parameter sharing

**Start here, then consider QMIX if you need explicit coordination!**
