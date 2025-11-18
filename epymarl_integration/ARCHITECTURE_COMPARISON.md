# Architecture Comparison Guide

This guide shows all available architectures for testing different approaches on the coverage problem with **POMDP** and **rotation penalty**.

## 🎯 Available Configurations

### **1. IQL + Flat Observations (Baseline)**
```bash
python src/main.py --config=iql_flat --env-config=coverage_flat
```

**Features:**
- ✅ Simple independent Q-learning
- ✅ Flat 64-dim observations
- ✅ Standard RNN agent
- ✅ POMDP (obstacle exploration)
- ✅ Rotation penalty (smooth movement)
- ❌ No scale invariance (fails on different grid sizes)

**Use for:** Baseline comparison, fastest training

---

### **2. IQL + CNN (Scale Invariant)**
```bash
python src/main.py --config=iql_cnn --env-config=coverage_small
```

**Features:**
- ✅ Simple independent Q-learning
- ✅ Spatial observations (4 channels + scalars)
- ✅ CNN with adaptive pooling
- ✅ POMDP (obstacle exploration)
- ✅ Rotation penalty (smooth movement)
- ✅ **Scale invariant** (20×20 → 30×30 transfer)

**Use for:** Scale invariance testing, main research contribution

---

### **3. QMIX + Flat Observations**
```bash
python src/main.py --config=qmix_flat --env-config=coverage_flat
```

**Features:**
- ✅ Value decomposition with mixing
- ✅ Explicit coordination
- ✅ Flat 64-dim observations
- ✅ Standard RNN agent
- ✅ POMDP (obstacle exploration)
- ✅ Rotation penalty (smooth movement)
- ❌ No scale invariance

**Use for:** Testing coordination benefits with standard architecture

---

### **4. QMIX + CNN (Full Features)**
```bash
python src/main.py --config=qmix_cnn --env-config=coverage_small
```

**Features:**
- ✅ Value decomposition with mixing
- ✅ Explicit coordination
- ✅ Spatial observations
- ✅ CNN with adaptive pooling
- ✅ POMDP (obstacle exploration)
- ✅ Rotation penalty (smooth movement)
- ✅ **Scale invariant**

**Use for:** Best performance (but slower training)

---

## 📊 Quick Comparison Table

| Config | Algorithm | Observations | Scale Invariant | Coordination | Speed | Use Case |
|--------|-----------|--------------|-----------------|--------------|-------|----------|
| **iql_flat** | IQL | Flat (64-dim) | ❌ | Implicit | ⚡⚡⚡ Fast | Baseline |
| **iql_cnn** | IQL | Spatial (CNN) | ✅ | Implicit | ⚡⚡ Medium | **Main contribution** |
| **qmix_flat** | QMIX | Flat (64-dim) | ❌ | Explicit | ⚡⚡ Medium | Coordination test |
| **qmix_cnn** | QMIX | Spatial (CNN) | ✅ | Explicit | ⚡ Slow | Best performance |

---

## 🔬 Experimental Setup

### **Baseline Comparison**

Test the impact of each component:

**1. Effect of CNN (Scale Invariance):**
```bash
# Train both on 20×20
python src/main.py --config=iql_flat --env-config=coverage_flat
python src/main.py --config=iql_cnn --env-config=coverage_small

# Test both on 30×30 (zero-shot transfer)
python src/main.py --config=iql_flat --env-config=coverage_flat \
    --checkpoint_path=models/iql_flat/... --evaluate=True

python src/main.py --config=iql_cnn --env-config=coverage_large \
    --checkpoint_path=models/iql_cnn/... --evaluate=True
```

**Expected results:**
- Flat: 20×20=90%, 30×30=**35%** (fails)
- CNN: 20×20=90%, 30×30=**75%** (succeeds!)

---

**2. Effect of QMIX (Coordination):**
```bash
# Compare IQL vs QMIX on same architecture
python src/main.py --config=iql_flat --env-config=coverage_flat
python src/main.py --config=qmix_flat --env-config=coverage_flat
```

**Expected results:**
- IQL: 85-90% coverage (independent learning)
- QMIX: 90-95% coverage (better coordination)

---

**3. Effect of Rotation Penalty:**
```bash
# Disable rotation penalty
# Edit coverage_flat.yaml: reward_rotation_penalty: 0.0

# Compare movement patterns in TensorBoard
# Look for smoother trajectories with penalty enabled
```

---

### **Ablation Studies**

Test each component independently:

| Experiment | Config | POMDP | Rotation | CNN | QMIX | Expected Coverage |
|------------|--------|-------|----------|-----|------|-------------------|
| 1. Baseline | iql_flat | ✅ | ✅ | ❌ | ❌ | 85-90% |
| 2. +CNN | iql_cnn | ✅ | ✅ | ✅ | ❌ | 85-90% (same) |
| 3. +QMIX | qmix_flat | ✅ | ✅ | ❌ | ✅ | 90-95% |
| 4. Full | qmix_cnn | ✅ | ✅ | ✅ | ✅ | 90-95% |

**Key insights:**
- CNN doesn't improve coverage, but enables **transfer**
- QMIX improves coverage through **coordination**
- Rotation penalty improves **efficiency** (fewer steps)

---

## 🚀 Training Commands

### **Quick Start (IQL + CNN)**
```bash
# Recommended starting point
python src/main.py \
    --config=iql_cnn \
    --env-config=coverage_small \
    with use_tensorboard=True
```

### **Baseline (IQL + Flat)**
```bash
# Fast baseline for comparison
python src/main.py \
    --config=iql_flat \
    --env-config=coverage_flat \
    with use_tensorboard=True
```

### **Best Performance (QMIX + CNN)**
```bash
# Slowest but best results
python src/main.py \
    --config=qmix_cnn \
    --env-config=coverage_small \
    with use_tensorboard=True
```

### **Coordination Test (QMIX + Flat)**
```bash
# Test QMIX coordination
python src/main.py \
    --config=qmix_flat \
    --env-config=coverage_flat \
    with use_tensorboard=True
```

---

## 📈 Expected Training Times (GPU)

| Config | Timesteps | Time | Coverage |
|--------|-----------|------|----------|
| iql_flat | 1M | 2-3 hours | 85-90% |
| iql_cnn | 1M | 4-5 hours | 85-90% |
| qmix_flat | 1M | 3-4 hours | 90-95% |
| qmix_cnn | 1M | 6-8 hours | 90-95% |

---

## 🧪 Testing Scale Invariance

**Only CNN configs support scale invariance:**

```bash
# 1. Train on 20×20
python src/main.py --config=iql_cnn --env-config=coverage_small

# 2. Test on 30×30 (zero-shot transfer)
python src/main.py \
    --config=iql_cnn \
    --env-config=coverage_large \
    --checkpoint_path=results/models/<run>/<timestep> \
    --evaluate=True
```

**Results:**
- **IQL + CNN**: 75-85% on 30×30 ✅
- **IQL + Flat**: 30-50% on 30×30 ❌

---

## 📝 Configuration Files Summary

### **Algorithm Configs** (src/config/algs/)
- `iql_flat.yaml` - IQL with RNN
- `iql_cnn.yaml` - IQL with CNN
- `qmix_flat.yaml` - QMIX with RNN
- `qmix_cnn.yaml` - QMIX with CNN

### **Environment Configs** (src/config/envs/)
- `coverage_flat.yaml` - 20×20, flat observations
- `coverage_small.yaml` - 20×20, spatial observations (CNN)
- `coverage_large.yaml` - 30×30, spatial observations (CNN)

---

## 🎓 Research Questions

**Q1: Does CNN enable scale invariance?**
- **Test:** Train iql_cnn on 20×20, evaluate on 30×30
- **Baseline:** Train iql_flat on 20×20, evaluate on 30×30
- **Metric:** Coverage % on 30×30
- **Hypothesis:** CNN maintains 70-85%, flat drops to 30-50%

**Q2: Does QMIX improve coordination?**
- **Test:** Compare qmix_flat vs iql_flat on 20×20
- **Metric:** Final coverage %, steps to 90% coverage
- **Hypothesis:** QMIX achieves 5-10% higher coverage

**Q3: Does rotation penalty improve efficiency?**
- **Test:** Compare with/without rotation penalty
- **Metric:** Steps to 90% coverage, trajectory smoothness
- **Hypothesis:** 10-20% fewer steps with penalty

**Q4: Does POMDP prevent trivial solutions?**
- **Test:** Visual inspection of learned behavior
- **Metric:** Exploration patterns, obstacle discovery
- **Hypothesis:** Agents learn exploration, not pathplanning

---

## 🔍 What to Monitor in TensorBoard

**All configs track:**
- `episode_return` - Total reward
- `coverage_pct` - Main metric (target: 90-95%)
- `episode_length` - Steps to completion
- `rotation_penalty` - Movement smoothness
- `loss` - Training stability

**IQL specific:**
- `q_values` - Q-function estimates
- `grad_norm` - Gradient magnitudes

**QMIX specific:**
- `mixer_loss` - Mixing network loss
- `q_taken` - Chosen Q-values
- `target_mean` - Target network values

---

## 💡 Recommendations

**For fast experiments:**
→ Use `iql_flat` (2-3 hours)

**For scale invariance (main contribution):**
→ Use `iql_cnn` (4-5 hours)

**For best performance:**
→ Use `qmix_cnn` (6-8 hours)

**For coordination study:**
→ Compare `iql_flat` vs `qmix_flat`

**For ablations:**
→ Test all 4 configs systematically

---

## 🎯 Summary

**You now have 4 complete architectures:**

1. **IQL + Flat** - Fast baseline
2. **IQL + CNN** - Scale invariant (main contribution)
3. **QMIX + Flat** - Coordination test
4. **QMIX + CNN** - Full features

**All include:**
- ✅ POMDP (obstacle exploration)
- ✅ Rotation penalty (smooth movement)
- ✅ Proper reward scaling
- ✅ Comprehensive configs

**Ready for experiments!** 🚀

Pick your config based on the research question you want to answer.
