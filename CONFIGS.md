# RL Scheduler Optimization Configurations

This file contains pre-tuned configurations for different training scenarios.

## Configuration 1: Conservative Training (Stable)

Best for: Initial testing, limited computational resources

```bash
python main.py --train \
  --episodes 150 \
  --learning-rate 3e-4 \
  --gamma 0.98 \
  --memory-size 30000 \
  --batch-size 64 \
  --target-update-freq 1500 \
  --eval-interval 10 \
  --use-dueling
```

**Characteristics**:
- Slower training but more stable
- Lower memory footprint
- Good for debugging and validation

---

## Configuration 2: Balanced Training (Recommended)

Best for: Production, default choice

```bash
python main.py --train \
  --episodes 300 \
  --learning-rate 5e-4 \
  --gamma 0.99 \
  --memory-size 50000 \
  --batch-size 128 \
  --target-update-freq 1000 \
  --eval-interval 15 \
  --use-dueling
```

**Characteristics**:
- Balance between speed and stability
- Good generalization
- Recommended starting point

---

## Configuration 3: Aggressive Training (Fast)

Best for: Exploration, quick iterations, powerful hardware

```bash
python main.py --train \
  --episodes 500 \
  --learning-rate 1e-3 \
  --gamma 0.995 \
  --memory-size 100000 \
  --batch-size 256 \
  --target-update-freq 500 \
  --eval-interval 20 \
  --use-dueling
```

**Characteristics**:
- Faster convergence
- Requires more memory/GPU
- May have higher variance

---

## Configuration 4: Large-Scale Training

Best for: Large datasets, serious optimization

```bash
python main.py --train \
  --episodes 1000 \
  --learning-rate 3e-4 \
  --gamma 0.999 \
  --memory-size 200000 \
  --batch-size 256 \
  --target-update-freq 2000 \
  --eval-interval 50 \
  --episode-task-count 1024 \
  --use-dueling
```

**Characteristics**:
- Extended training for complex patterns
- Very high quality final policy
- Requires significant resources

---

## Hyperparameter Tuning Guide

### Learning Rate (lr)
- **Too High** (1e-2): Unstable, diverging rewards
- **Good Range** (1e-4 to 1e-3): 
  - Start with 5e-4 for Dueling DQN
  - Decrease if training is unstable
- **Too Low** (1e-5): Very slow learning

### Gamma (Discount Factor)
- **Closer to 1.0**: More long-term planning
  - 0.99 or 0.995: Excellent for scheduling (prefer far-future optimization)
- **Closer to 0**: More myopic decisions
  - 0.95 or lower: Use if immediate task completion important

### Memory Size
- **Rule of thumb**: 5000-10x batch size
- **Minimum**: batch_size * 10
- **Recommended**: batch_size * 400-800
- **GPU memory constraint**: Reduce batch_size before memory_size

### Batch Size
- **Smaller** (32-64): More frequent updates, noisier gradients
- **Medium** (128): Good balance
- **Larger** (256-512): Smoother gradients, slower convergence

### Target Update Frequency
- **Too Frequent** (<500): Chasing moving target, unstable
- **Good Range** (1000-2000): Allows learning before target changes
- **Too Infrequent** (>5000): Target lagging too far behind

---

## Performance Benchmarking Script

```python
import subprocess
import json
import time

configs = {
    "conservative": {
        "learning_rate": 3e-4,
        "gamma": 0.98,
        "batch_size": 64,
    },
    "balanced": {
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "batch_size": 128,
    },
    "aggressive": {
        "learning_rate": 1e-3,
        "gamma": 0.995,
        "batch_size": 256,
    },
}

results = {}

for config_name, params in configs.items():
    print(f"\n{'='*60}")
    print(f"Training: {config_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    cmd = [
        "python", "main.py",
        "--train",
        "--episodes", "200",
        "--learning-rate", str(params["learning_rate"]),
        "--gamma", str(params["gamma"]),
        "--batch-size", str(params["batch_size"]),
        "--use-dueling",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    results[config_name] = {
        "time": elapsed,
        "success": result.returncode == 0,
    }
    
    print(f"Training completed in {elapsed:.1f} seconds")

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
for config, result in results.items():
    print(f"{config}: {result['time']:.1f}s - {'✓' if result['success'] else '✗'}")
```

---

## Common Issues & Solutions

### Issue 1: Training Reward Decreases
**Possible Causes**:
- Learning rate too high → Reduce to 3e-4
- Batch size too small → Increase to 128+
- Network too small → Use Dueling architecture

**Fix**:
```bash
python main.py --train --episodes 200 --learning-rate 3e-4 --batch-size 128 --use-dueling
```

### Issue 2: Model Not Improving on Test Set
**Possible Causes**:
- Overfitting to training data → Use larger eval_interval
- Reward function not aligned with test metrics → Review reward design
- Not enough training → Increase episodes

**Fix**:
```bash
python main.py --train --episodes 500 --eval-interval 20 --use-dueling
```

### Issue 3: Training Too Slow
**Possible Causes**:
- Batch size too large → Reduce to 64 or 128
- Target update too frequent → Increase to 1500
- Learning rate too low → Increase to 1e-3

**Fix**:
```bash
python main.py --train --batch-size 128 --target-update-freq 1500 --learning-rate 1e-3 --use-dueling
```

### Issue 4: GPU Out of Memory
**Possible Causes**:
- Batch size too large
- Memory size too large
- Network layers too big

**Fix**:
```bash
python main.py --train --batch-size 64 --memory-size 30000 --episodes 200 --use-dueling
```

---

## Monitoring & Analysis

### Key Metrics to Log

1. **Episode Reward**: Should trend upward
```
E1: +34.2, E2: +38.1, E3: +41.5, ... (good)
E1: +40.1, E2: +38.2, E3: +35.4, ... (bad - decreasing)
```

2. **Loss**: Should trend downward
```
E1: 0.45, E2: 0.38, E3: 0.28, ... (good)
E1: 0.45, E2: 0.52, E3: 0.48, ... (unstable)
```

3. **Validation Reward**: Should show convergence
```
V20: 42.1, V40: 48.3, V60: 50.2, V80: 50.5, V100: 50.6 (converged)
V20: 42.1, V40: 60.2, V60: 35.5, V80: 55.3, V100: 42.8 (unstable)
```

### Plotting Training Curves

```python
import matplotlib.pyplot as plt
import numpy as np

# After training, plot in Jupyter:
rewards = agent.train_history['rewards']
losses = agent.train_history['losses']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(rewards)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('Training Reward per Episode')
ax1.grid(True)

ax2.plot(losses)
ax2.set_xlabel('Update Step')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

---

## Recommended Workflow

### Phase 1: Validation (1-2 hours)
1. Use **Configuration 1 (Conservative)**
2. Train on 20% of data
3. Check: Reward increases, no NaNs, reasonable test performance

### Phase 2: Optimization (2-4 hours)
1. Use **Configuration 2 (Balanced)**
2. Train on 100% of training data
3. Validate frequently
4. Adjust hyperparameters based on metrics

### Phase 3: Production (4-24 hours)
1. Use **Configuration 3 or 4**
2. Train to convergence
3. Test against all baselines
4. Save best model

---

## Quick Start Commands

```bash
# Quick test (5 min)
python main.py --train --episodes 50 --max-queue-size 8 --episode-task-count 256

# Default training (1-2 hours)
python main.py --train --episodes 300 --use-dueling

# Serious training (4+ hours)
python main.py --train --episodes 1000 --learning-rate 3e-4 --memory-size 100000 --use-dueling

# Compare all schedulers (test trained model)
python main.py --load-model --model models/rl_agent.pth

# Skip RL, test traditional only
python main.py --skip-rl
```

