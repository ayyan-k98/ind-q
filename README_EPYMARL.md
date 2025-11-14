# EPyMARL Multi-Agent Coverage System

Complete implementation of multi-agent coverage using EPyMARL + QMIX algorithm.

## 🎯 Project Goals

- **Target Performance:** 85%+ coverage
- **Beat Baselines:** Outperform Random, Greedy, and Single-Agent
- **Faster Inference:** More efficient than greedy search
- **Multi-Agent Coordination:** Demonstrate benefits of cooperation

---

## 📁 Project Structure

```
ind-q/
├── epymarl_coverage_env.py       # Custom coverage environment (EPyMARL compatible)
├── coverage_config.yaml           # Training configuration for QMIX
├── train_coverage.py              # Training script with checkpointing
├── evaluate_coverage.py           # Evaluation with baselines
├── EPyMARL_Coverage_Training.ipynb # Complete Colab notebook
├── colab_setup.py                 # Automated Colab setup
└── README_EPYMARL.md              # This file
```

---

## 🚀 Quick Start (Colab)

### Option 1: Use Notebook (Recommended)

1. Upload `EPyMARL_Coverage_Training.ipynb` to Colab
2. Run cells sequentially
3. All dependencies installed automatically
4. Checkpoints saved to Google Drive

### Option 2: Manual Setup

```python
# 1. Run setup
!python colab_setup.py

# 2. Test environment
from epymarl_coverage_env import CoverageEnvironment
env = CoverageEnvironment(n_agents=4, grid_size=20)
print(env.get_env_info())

# 3. Evaluate baselines
!python evaluate_coverage.py --n-episodes 20 --visualize

# 4. Train
!python train_coverage.py --config coverage_config.yaml

# 5. Resume (if interrupted)
!python train_coverage.py --config coverage_config.yaml --resume
```

---

## 🏗️ Architecture

### Environment (`epymarl_coverage_env.py`)

**Observation Space** (64D vector per agent):
- Own state: position, orientation, time [5]
- Sensor region info: uncovered ratio, frontiers [6]
- Other agents: relative positions [40]
- Global stats: coverage percentage [3]
- Padding [10]

**Action Space** (9 discrete):
- 8 directions (N, NE, E, SE, S, SW, W, NW) + Stay
- **Action Masking:** Invalid actions filtered out

**Reward** (Team + Shaping):
- Primary: Coverage increase × 100
- Shaping: Frontier bonus, spread bonus, movement penalties
- Total range: [-10, 50] per step

**State** (For QMIX mixer):
- All agent observations concatenated
- Global coverage grid
- Metadata (time, coverage %, n_agents)

### Algorithm (QMIX)

- **Centralized Training, Decentralized Execution (CTDE)**
- **Mixer Network:** Combines individual Q-values monotonically
- **Credit Assignment:** Learns coordination patterns
- **Configuration:** See `coverage_config.yaml`

---

## 🔧 Configuration

Key parameters in `coverage_config.yaml`:

```yaml
# Environment
env_args:
  n_agents: 4
  grid_size: 20
  episode_limit: 200
  map_type: empty  # or 'rooms', 'random'

# Training
t_max: 2000000  # 2M steps (~2-4 hours on Colab)
batch_size: 32
lr: 0.0005
epsilon_anneal_time: 50000

# Network
hidden_dim: 64
mixing_embed_dim: 32

# Rewards
reward_scale_coverage: 100.0
reward_frontier_bonus: 0.5
reward_spread_bonus: 0.1
```

---

## 📊 Baselines

Three baselines for comparison:

### 1. Random Agent
- Selects random valid actions
- Expected: ~30-40% coverage

### 2. Greedy Agent (Multi-Agent)
- Each agent moves toward nearest uncovered cell
- Expected: ~60-75% coverage
- Fast but no coordination

### 3. Single Agent (Greedy)
- One agent with greedy policy
- Expected: ~70-80% coverage
- Slower but systematic

**Evaluate baselines:**
```bash
python evaluate_coverage.py --n-episodes 20 --visualize
```

---

## 🎓 Training

### Start Training
```bash
python train_coverage.py \
    --config coverage_config.yaml \
    --checkpoint-dir ./checkpoints \
    --results-dir ./results
```

### Resume Training
```bash
python train_coverage.py \
    --config coverage_config.yaml \
    --resume
```

### Monitor with TensorBoard
```bash
tensorboard --logdir ./results/tb_logs
```

**Training Progress:**
- **0-500k steps:** Agents learn basic movement, coverage increases slowly
- **500k-1M steps:** Coordination emerges, coverage accelerates
- **1M-2M steps:** Fine-tuning, approaching optimal policy
- **Expected final:** 85-95% coverage

---

## 📈 Evaluation

### Evaluate Trained Agent

```python
# TODO: Implement trained agent evaluation
# For now, baselines provide benchmarks
```

### Compare Results

Target metrics:
- **Coverage:** 85%+ (vs 60-75% greedy, 70-80% single)
- **Return:** 80+ (vs ~50 greedy)
- **Steps:** <180 (vs ~200 greedy)
- **Inference Time:** <0.5s (vs ~1s greedy)

---

## 🐛 Troubleshooting

### Installation Issues

**EPyMARL not found:**
```bash
cd epymarl && pip install -e . && cd ..
```

**Gym version error:**
```bash
pip install gym==0.21.0
```

### Training Issues

**CUDA out of memory:**
- Reduce `batch_size` to 16 in config
- Use CPU: `use_cuda: False`

**Training slow:**
- Reduce `t_max` to 1000000
- Use smaller grid: `grid_size: 15`
- Fewer agents: `n_agents: 2`

**Checkpointing fails:**
- Check Google Drive permissions
- Ensure `checkpoint_dir` exists
- Use local directory instead

### Environment Issues

**Action masking not working:**
- Check `get_avail_agent_actions()` returns valid list
- Debug with `env.render()` to see agent positions

**Coverage not increasing:**
- Check raycasting logic in `_raycast_update()`
- Verify `coverage_grid` updates correctly
- Print `info` dict to see `coverage_increase`

**Agents colliding:**
- Check `_resolve_conflicts()` is called
- Verify `_is_valid_position()` checks other agents

---

## 📝 Key Implementation Details

### 1. Action Masking (CRITICAL)
```python
def get_avail_agent_actions(self, agent_id):
    # Returns [1, 0, 1, ..., 1]
    # 1 = valid, 0 = invalid
    avail = [0] * self.n_actions
    for i, (dr, dc) in enumerate(self.actions):
        target = (pos[0] + dr, pos[1] + dc)
        if self._is_valid_position(target, agent_id):
            avail[i] = 1
    avail[4] = 1  # Always allow stay
    return avail
```

**Why it matters:** Without masking, agents waste 30-40% of training on invalid actions.

### 2. Raycasting Coverage
```python
def _raycast_update(self, agent_id):
    # Cast rays within FOV
    # Update coverage_grid with sigmoid decay
    # Monotonic: coverage never decreases
```

**Coverage Probability:**
- Close cells (≤1): `pc = 1.0`
- Far cells: `pc = sigmoid(distance)`
- Always take `max(current_pc, new_pc)`

### 3. Reward Calculation
```python
def _calculate_reward(self, actions, executed_moves):
    # Primary: coverage increase × 100
    coverage_reward = (new_cov - old_cov) * 100

    # Shaping: small bonuses/penalties
    shaping = frontier_bonus + spread_bonus - stay_penalty - step_penalty

    return coverage_reward + shaping
```

**Reward Scale:**
- Good step: +5 to +15
- Neutral step: -0.5 to +0.5
- Bad step: -2 to 0
- Episode total: +70 to +105

### 4. Observation Construction
```python
def get_obs_agent(self, agent_id):
    obs = []
    # 1. Own state [5]
    obs.extend([x/grid, y/grid, sin(θ), cos(θ), time])

    # 2. Sensor info [6]
    obs.extend([uncovered_ratio, covered_ratio, frontier_dist, ...])

    # 3. Other agents [40]
    for other in others:
        obs.extend([dx, dy, dist, angle])

    # 4. Global stats [3]
    obs.extend([coverage%, coverage_sum, rate])

    # Pad to 64
    return np.array(obs, dtype=np.float32)
```

**Fixed Size:** Always 64D, works with any number of agents (up to 11).

---

## 🔬 Expected Results

Based on similar multi-agent coverage tasks:

### Training Curves

**Coverage over Time:**
```
  0k steps: ~20% (random exploration)
500k steps: ~60% (basic coordination)
  1M steps: ~80% (good coordination)
  2M steps: ~90% (near-optimal)
```

**Episode Return over Time:**
```
  0k steps: ~30 (inefficient)
500k steps: ~60 (improving)
  1M steps: ~80 (good)
  2M steps: ~90 (optimal)
```

### Final Performance

**Target (Trained QMIX):**
- Coverage: **85-95%**
- Return: **80-95**
- Steps: **150-180**
- Inference: **<0.5s**

**Greedy Baseline:**
- Coverage: **60-75%**
- Return: **50-60**
- Steps: **180-200**
- Inference: **~1s**

**Single Agent:**
- Coverage: **70-80%**
- Return: **60-70**
- Steps: **180-200**
- Inference: **~0.5s**

---

## 🎯 Success Criteria

✅ **Coverage:** 85%+ (beat all baselines)
✅ **Multi-Agent Advantage:** Clearly faster/better than single agent
✅ **Efficiency:** Fewer steps than greedy
✅ **Inference Speed:** Faster than greedy (<0.5s vs ~1s)

---

## 📚 References

- **EPyMARL:** https://github.com/uoe-agents/epymarl
- **QMIX Paper:** Rashid et al., "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning" (ICML 2018)
- **PyMARL:** https://github.com/oxwhirl/pymarl

---

## 🤝 Contributing

This is a research/class project. For improvements:

1. Test changes locally first
2. Document any config changes
3. Update baselines if environment changes
4. Keep checkpointing compatible

---

## 📧 Contact

For questions or issues, see GitHub repository.

---

**Good luck with training! 🚀**

Remember:
1. Start with baselines (Section 4)
2. Monitor TensorBoard during training
3. Resume if interrupted (happens on Colab)
4. Compare final results to baselines
5. Celebrate 85%+ coverage! 🎉
