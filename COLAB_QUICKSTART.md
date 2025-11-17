# 🚀 Colab/Kaggle Quick Start Guide

Run EPyMARL Coverage training on Google Colab or Kaggle with automatic checkpointing and resume functionality.

## Option 1: Direct Colab Link (Fastest)

**Click here to open in Colab:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayyan-k98/ind-q/blob/main/EPyMARL_Coverage_Colab.ipynb)

Then just **Run All** cells!

---

## Option 2: Manual Setup (5 steps)

### Step 1: Open Colab/Kaggle
- **Colab:** https://colab.research.google.com/
- **Kaggle:** https://www.kaggle.com/code

### Step 2: Upload Notebook
Upload `EPyMARL_Coverage_Colab.ipynb` from this repository.

### Step 3: Enable GPU (Recommended)
- **Colab:** Runtime → Change runtime type → GPU
- **Kaggle:** Settings → Accelerator → GPU

### Step 4: Run All Cells
Click **Runtime → Run all** (Colab) or **Run All** (Kaggle)

### Step 5: Monitor Training
- TensorBoard will show live training progress
- Training auto-saves to Google Drive every 100K steps
- If session times out, just restart and run again - it will resume automatically!

---

## 📊 What to Expect

### Training Timeline
| Timesteps | Coverage | Time (GPU) | Time (CPU) |
|-----------|----------|------------|------------|
| 500K      | 60-70%   | 1.5 hours  | 4 hours    |
| 1M        | 75-85%   | 3 hours    | 8 hours    |
| 2M        | 85-95%   | 6 hours    | 16 hours   |

### Typical Workflow
1. **Session 1 (4-5 hours):** Train from 0 → 1M timesteps
2. ⏰ *Session times out, checkpoint auto-saved to Drive*
3. **Session 2 (4-5 hours):** Resume from 1M → 2M timesteps
4. ✅ **Done!** Model achieves 85-95% coverage

---

## 🔄 Resume Training After Timeout

**Don't worry if your session times out!** The notebook automatically:
- ✅ Saves checkpoints to Google Drive every 100K steps
- ✅ Detects existing checkpoints on restart
- ✅ Resumes training from where it left off

**To resume:** Just run the notebook again - it handles everything automatically.

---

## 📈 Monitor Progress

### During Training
You'll see output like:
```
t_env: 500000 / 2000000
Episode: 2500  reward: 52.3  coverage: 78.5%
Episode: 2501  reward: 55.1  coverage: 81.2%
...
```

### TensorBoard
Live plots show:
- Episode rewards over time
- Coverage percentage trend
- Win rate (episodes reaching 95% coverage)
- Training loss

### Check Progress Manually
Run the "Check Training Progress" cell to see:
```
Training Progress:
  Latest checkpoint: checkpoint_1000000
  Timesteps: 1,000,000 / 2,000,000
  Progress: 50.0%
```

---

## 🎯 Expected Results

After 2M timesteps, your model should achieve:
- **Coverage:** 85-95% (target: beat 60-75% greedy baseline)
- **Episode length:** 150-200 steps (20×20 grid)
- **Inference speed:** <0.5s per episode (faster than ~1s greedy)

---

## 💾 Save & Download Results

### Checkpoints Location
- **Colab:** `/content/drive/MyDrive/EPyMARL_Coverage/`
- **Kaggle:** `/kaggle/working/EPyMARL_Coverage/`

### Download Results
Run the final cell to download:
- Trained models
- TensorBoard logs
- Evaluation results

---

## ⚙️ Configuration Options

### Train Longer
```python
# In cell 8, change t_max:
--t_max=5000000  # 5M instead of 2M
```

### More Agents
```python
# Add to training command:
--env-config.n_agents=8
```

### Larger Grid
```python
# Add to training command:
--env-config.grid_size=30
--env-config.episode_limit=300
```

### Add Obstacles
```python
# Add to training command:
--env-config.map_type=rooms
```

---

## 🐛 Troubleshooting

### "No module named 'smaclite'" or Import Errors

**Cause:** smaclite is installed by EPyMARL, not available on PyPI

**Solution:**
1. Make sure cell 3 ran completely (EPyMARL installation)
2. Restart runtime and run all cells from the beginning
3. If still failing, manually run:
   ```python
   !cd epymarl && pip install -e . && cd ..
   ```

### "CUDA out of memory"
→ Change to CPU: `--use_cuda=False` in cell 8

### "Session disconnected"
→ Normal! Just restart and run cell 8 again - training will resume

### Training not improving
→ Check TensorBoard - might need to train longer (3-4M timesteps for harder maps)

---

## 📞 Need Help?

1. Check [ERROR_CHECK_REPORT.md](ERROR_CHECK_REPORT.md) for validation results
2. Review [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for details
3. See [README_EPYMARL.md](README_EPYMARL.md) for advanced usage

---

## ✅ Quick Checklist

- [ ] Open notebook in Colab/Kaggle
- [ ] Enable GPU (Runtime → Change runtime type)
- [ ] Run all cells (Runtime → Run all)
- [ ] Check TensorBoard shows training curves
- [ ] Wait 4-5 hours for first checkpoint
- [ ] If timeout, restart and run again
- [ ] Repeat until 2M timesteps reached
- [ ] Evaluate final model (last cell)
- [ ] Download results

**That's it! Happy training! 🎉**
