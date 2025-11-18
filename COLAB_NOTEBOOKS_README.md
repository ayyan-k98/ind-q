# Google Colab Training Notebooks

This directory contains two comprehensive Google Colab notebooks for training multi-agent reinforcement learning algorithms on the coverage task.

## 📓 Available Notebooks

### 1. `colab_iql_flat.ipynb` - IQL with Flat Observations
**Algorithm:** Independent Q-Learning (IQL)
**Observations:** 64-dimensional flat vectors
**Training Time:** ~2-3 hours on Colab GPU

**Features:**
- Simple baseline algorithm
- Fast training
- Parameter sharing across agents
- Implicit coordination through environment

**Best for:**
- Quick experiments
- Baseline comparisons
- Understanding basic multi-agent RL

---

### 2. `colab_qmix_flat.ipynb` - QMIX with Flat Observations
**Algorithm:** QMIX (Q-Mixing Networks)
**Observations:** 64-dimensional flat vectors
**Training Time:** ~3-4 hours on Colab GPU

**Features:**
- Value decomposition with mixing network
- Explicit multi-agent coordination
- Centralized training, decentralized execution (CTDE)
- Monotonic value function factorization

**Best for:**
- Advanced coordination
- Better performance on complex tasks
- Research on value decomposition methods

---

## 🚀 Quick Start

### Option 1: Upload to Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Select `colab_iql_flat.ipynb` or `colab_qmix_flat.ipynb`
4. Run all cells sequentially

### Option 2: Open from GitHub

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Open notebook → GitHub**
3. Enter repository URL: `https://github.com/ayyan-k98/ind-q`
4. Select the notebook you want to use

### Option 3: Direct Links

**IQL Notebook:**
```
https://colab.research.google.com/github/ayyan-k98/ind-q/blob/main/colab_iql_flat.ipynb
```

**QMIX Notebook:**
```
https://colab.research.google.com/github/ayyan-k98/ind-q/blob/main/colab_qmix_flat.ipynb
```

---

## 📊 What Each Notebook Includes

Both notebooks provide comprehensive training and monitoring capabilities:

### 1. **Setup and Installation**
- GPU availability check
- Dependency installation
- Repository cloning
- EPyMARL integration

### 2. **Real-Time Monitoring**
- Training curves (returns, coverage, epsilon)
- Loss tracking (total, mixer, agent losses for QMIX)
- Episode length and efficiency metrics
- Live statistics dashboard

### 3. **Visualization Tools**
- Coverage heatmaps
- Agent trajectories
- Obstacle maps
- Coordination analysis (QMIX only)
- Performance distributions

### 4. **TensorBoard Integration**
- Real-time training metrics
- Detailed logs
- Experiment tracking

### 5. **Evaluation and Analysis**
- Model loading and evaluation
- Performance analysis over multiple episodes
- Statistical summaries
- Comparison tools (when both algorithms trained)

### 6. **Model Management**
- Automatic checkpoint saving
- Model loading for continued training
- Results export and download

---

## 📈 Monitoring During Training

Both notebooks include a custom monitoring class that tracks:

### IQL Monitoring
- Episode returns
- Coverage percentage
- Exploration rate (epsilon)
- Training loss
- Episode length
- Coverage efficiency

### QMIX Monitoring (additional)
- Mixer network loss
- Individual agent loss
- Inter-agent distances (coordination metric)
- Agent spacing variance

---

## 🎯 Expected Performance

### IQL + Flat Observations
- **Coverage:** 70-85% on 20×20 grid
- **Episode Length:** 150-200 steps
- **Training Time:** 2-3 hours (1M timesteps)

### QMIX + Flat Observations
- **Coverage:** 75-90% on 20×20 grid
- **Episode Length:** 140-180 steps
- **Training Time:** 3-4 hours (1M timesteps)
- **Better coordination** through value decomposition

---

## 📁 Output Files

After training, each notebook generates:

### Saved Models
- Checkpoints every 100k timesteps
- Located in `results/models/`

### Visualizations
- Training curves (PNG)
- Performance analysis plots
- Coverage heatmaps
- All saved in `results/plots/`

### Metrics
- CSV files with all training metrics
- TensorBoard logs
- Sacred experiment logs

### Downloadable Results
- Zip file containing all results
- Automatic download at the end

---

## 🔧 Customization

### Modify Training Parameters

In each notebook, find the `CONFIG` cell and adjust:

```python
CONFIG = {
    'n_episodes': 5000,        # Number of episodes
    'batch_size': 32,          # Batch size for learning
    'learning_rate': 0.0005,   # Learning rate
    'epsilon_start': 1.0,      # Initial exploration
    'epsilon_end': 0.05,       # Final exploration
    'log_interval': 10,        # Logging frequency
    # ... more parameters
}
```

### Change Environment Settings

```python
env = CoverageEnv(
    grid_size=20,              # Grid dimensions (20×20 or 30×30)
    n_agents=4,                # Number of agents (2-10)
    max_steps=200,             # Episode length
    obstacle_density=0.1       # Obstacle percentage
)
```

---

## 🎨 Visualization Examples

### Coverage Heatmap
Shows which areas have been covered by agents, with agent positions overlaid.

### Agent Trajectories
Displays the path each agent took through the environment.

### Coordination Analysis (QMIX)
Plots inter-agent distances over time to show spreading behavior.

### Training Curves
- Returns over episodes
- Coverage percentage over episodes
- Loss curves (with smoothing)
- Exploration rate decay

---

## 💡 Tips for Best Results

### 1. **Use GPU Runtime**
- Go to **Runtime → Change runtime type**
- Select **GPU** as hardware accelerator
- Verify with `!nvidia-smi`

### 2. **Monitor Memory**
- Check memory usage periodically
- Clear outputs if notebook becomes slow
- Restart runtime if needed

### 3. **Save Checkpoints**
- Models are saved automatically
- Download checkpoints periodically
- Continue training from checkpoints if interrupted

### 4. **Adjust Visualization Frequency**
- Reduce `plot_interval` if training is slow
- Increase for more frequent updates
- Balance between performance and monitoring

### 5. **Use TensorBoard**
- Better for long training runs
- Less memory intensive than matplotlib
- Persistent across notebook restarts

---

## 🔬 Research Applications

These notebooks are ideal for:

1. **Algorithm Comparison**
   - IQL vs QMIX performance
   - Flat vs CNN observations (future notebooks)
   - Different coordination mechanisms

2. **Hyperparameter Tuning**
   - Learning rates
   - Exploration strategies
   - Network architectures

3. **Environment Analysis**
   - Different grid sizes
   - Varying number of agents
   - Obstacle configurations

4. **Transfer Learning**
   - Train on small grids
   - Test on larger grids
   - Evaluate generalization

---

## 📚 Additional Resources

- **EPyMARL Documentation:** [https://github.com/uoe-agents/epymarl](https://github.com/uoe-agents/epymarl)
- **QMIX Paper:** [Rashid et al., 2018](https://arxiv.org/abs/1803.11485)
- **Project Documentation:** See `README.md` and `IMPLEMENTATION_SUMMARY.md`
- **Architecture Guide:** See `ARCHITECTURE_COMPARISON.md`

---

## ⚠️ Troubleshooting

### Issue: GPU not available
**Solution:** Change runtime type to GPU (Runtime → Change runtime type → GPU)

### Issue: Out of memory
**Solution:**
- Reduce batch size
- Clear cell outputs
- Restart runtime

### Issue: Training too slow
**Solution:**
- Verify GPU is being used
- Reduce logging frequency
- Check system resources

### Issue: Checkpoints not saving
**Solution:**
- Check disk space
- Verify write permissions
- Check checkpoint path

### Issue: Visualizations not updating
**Solution:**
- Clear output and re-run cell
- Reduce plot frequency
- Check matplotlib backend

---

## 🤝 Contributing

If you improve these notebooks:
1. Test thoroughly on Google Colab
2. Document any new features
3. Update this README
4. Submit a pull request

---

## 📝 Citation

If you use these notebooks in your research, please cite:

```bibtex
@misc{ind-q-notebooks,
  title={Multi-Agent Coverage Training Notebooks},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/ayyan-k98/ind-q}
}
```

---

## 📧 Support

For issues or questions:
- Open an issue on GitHub
- Check existing documentation
- Review TensorBoard logs for training insights

---

**Happy Training! 🚀**
