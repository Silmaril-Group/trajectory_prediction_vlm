# Experimental Reward & Stability Features

These features are implemented but experimental. Use them via CLI overrides.

---

## 0. Stability Hardening (always-on, automatic)

The trainer has bounds and safeguards to prevent training collapse:

### Std floor + advantage clamp

```python
STD_FLOOR = 0.1     # never normalize by tiny std
ADV_CLAMP = 5.0     # advantages clamped to [-5, +5]

advantages = ((rewards - mean) / max(std, STD_FLOOR)).clamp(-ADV_CLAMP, ADV_CLAMP)
```

**Why:** prevents 10⁷ advantage explosion when std ≈ 0. The previous `+1e-8` floor allowed advantages of ~1e7 when all rewards were nearly identical, causing massive gradient blow-ups.

### Skip update on zero signal

```python
if std_r < 1e-6:
    return None  # caller skips backward
```

When all G generations get the same reward (no signal), the trainer **skips the optimizer update entirely** instead of zeroing advantages. This prevents the "permanent freeze" failure mode where zero-advantage updates accumulate noise.

The training loop checks for `loss is None` and skips backward + optimizer step.

### Schulman k3 KL estimator

The previous KL formula `(log_probs - ref_log_probs).mean()` is the **mean log-ratio**, which can be negative and isn't actually KL divergence.

Replaced with the **k3 estimator** (Schulman):
```python
diff = ref_log_probs - log_probs
kl = (diff.exp() - 1 - diff).mean()   # always ≥ 0, low variance
```

**Why:** the old formula could produce negative "KL" values that anti-anchored the model (pushed it AWAY from the reference). The k3 estimator is always ≥ 0 and a proper KL approximation.

### REINFORCE policy loss (num_iterations=1)

With `num_iterations=1`, PPO clipping is dead — the ratio `exp(log_probs - log_probs.detach())` is always 1. The previous code computed a meaningless clipped surrogate.

Replaced with direct REINFORCE:
```python
policy_loss = -(log_probs * advantage).mean()
```

**Why:** cleaner, removes dead code path, behaves identically when `num_iterations=1`.

### Recommended training stability config

```yaml
training:
  learning_rate: 5.0e-6      # lower than 2e-5 — safer for LoRA on VLMs
  max_grad_norm: 0.3         # tighter than 1.0 — prevents loss spikes
  warmup_ratio: 0.05

grpo:
  beta: 0.01                 # small KL anchor (k3 estimator)
  entropy_coeff: 0.02
  entropy_floor: 0.5         # emergency brake
  entropy_floor_coeff: 1.0
```

---

## 1. Simplified Proximity (distance to hitbox center)

**What changed:** Previously there were 3 separate bonuses:
- `proximity_bonus` (misses only, distance to duck top-left)
- `edge_bonus` (hits only, distance from screen center)
- `hitbox_center_bonus` (hits only, distance to hitbox center)

**Now:** Single `proximity_bonus` that measures distance from shot to the **duck hitbox center**. Works for BOTH hits and misses. Closer to hitbox center = higher bonus.

```
duck_center = duck_position + (sprite_size / 2) in normalized coords
distance = euclidean_distance(shot, duck_center)
proximity_bonus = proximity_bonus_max × e^(-decay × distance)
```

**Why simpler is better:** One continuous gradient from anywhere on screen toward the duck center. No separate hit/miss logic. The model always gets signal pointing toward the duck.

**Config:**
```yaml
reward:
  proximity_bonus: 0.5       # max bonus at zero distance
  proximity_decay: 3.0       # how fast bonus drops with distance
```

Same config keys as before — just the computation changed internally.

---

## 2. Format Reward Decay

**Problem:** Once the model learns tool call format (~200 steps), `format_weight=0.3` adds a constant +0.3 to ALL generations. GRPO advantages are computed relative to group mean, so a constant adds zero signal. It's wasted.

**Solution:** Decay format_weight to 0 over N steps.

```
format_weight(step) = initial_weight × max(0, 1 - step / decay_steps)
```

| Step | format_weight (decay_steps=1000) |
|------|----------------------------------|
| 0 | 0.30 (full — model learning format) |
| 250 | 0.225 |
| 500 | 0.15 |
| 750 | 0.075 |
| 1000+ | 0.00 (pure accuracy signal) |

**Config:**
```yaml
reward:
  format_weight: 0.3
  format_decay_steps: 1000    # 0 = no decay (constant)
```

**CLI:**
```bash
--override reward.format_decay_steps=1000
```

**When to use:** When the model learns format quickly (LFM2-VL-3B, Qwen3-VL-8B) and you want training to focus entirely on aiming after the first ~1000 steps.

---

## 3. Reward Normalization Modes

Three modes for computing GRPO advantages:

### Mode: `group` (default)

Standard GRPO. Normalize rewards within the current step's generations only.

```
advantages = (rewards - mean(rewards)) / std(rewards)
```

**Pros:** Simple, proven.
**Cons:** No memory of past performance. Model at 50% hit rate gets same advantage scale as model at 5%.

### Mode: `moving_avg`

⚠️ **Use with caution** — caused training collapse in early experiments. Now bounded with std floor + clamp, but `group` is the safer default.

Compare rewards against an exponential moving average (EMA) of recent performance.

```
baseline = (1 - α) × baseline + α × mean(current_rewards)
std_used = max(centered.std(), group.std(), 0.1)   # std floor
advantages = ((rewards - baseline) / std_used).clamp(-5, 5)
```

**How it works:**
- Baseline tracks the model's recent average reward
- A reward above baseline gets positive advantage
- As the model improves, baseline rises — the model must keep improving to get positive advantages
- Prevents plateau: the model can't rest on past performance

**Config:**
```yaml
reward:
  reward_normalization: "moving_avg"
  moving_avg_alpha: 0.01      # EMA smoothing (smaller = longer memory)
```

| alpha | Memory | Description |
|-------|--------|-------------|
| 0.1 | ~10 steps | Fast adaptation, noisy |
| 0.01 | ~100 steps | Smooth baseline, standard |
| 0.001 | ~1000 steps | Very stable, slow to adapt |

**When to use:** When hit rate plateaus at a fixed level. The moving average creates pressure to keep improving.

**Failure mode it had before bounds were added:** when baseline drifted close to current rewards, `centered.std()` could approach zero, producing massive advantages (10⁷+) and exploding gradients. The std floor (0.1) and clamp (±5) prevent this.

### Mode: `per_component`

Normalize accuracy and format rewards separately, then combine.

```
norm_accuracy = (accuracy - mean) / std    # across generations
norm_format = (format - mean) / std        # across generations

advantage = accuracy_weight × norm_accuracy + format_weight × norm_format
```

**How it works:**
- Each reward component is normalized to zero mean, unit variance independently
- A tiny proximity difference (0.01) gets the same importance as a big format difference (1.0)
- Prevents one component from dominating the gradient

**Example without per_component:**
```
Gen 0: accuracy=-0.10, format=1.0  →  total=0.90  →  advantage ≈ 0
Gen 1: accuracy=-0.15, format=1.0  →  total=0.85  →  advantage ≈ 0
(format dominates, accuracy difference invisible)
```

**Example with per_component:**
```
Gen 0: norm_acc=+1.0, norm_fmt=0.0  →  advantage = +1.0  (closer to duck!)
Gen 1: norm_acc=-1.0, norm_fmt=0.0  →  advantage = -1.0  (further from duck)
(accuracy difference fully visible)
```

**Config:**
```yaml
reward:
  reward_normalization: "per_component"
```

**When to use:** When format reward drowns out proximity signal. Especially useful early in training when some gens produce valid tool calls and some don't.

**Note:** Requires `reward_breakdowns` and `format_rewards` to be passed in the batch dict. Currently only works with per_component when these are available.

---

## CLI Examples

```bash
# Format decay: format reward fades to 0 over 1000 steps
--override reward.format_decay_steps=1000

# Moving average normalization
--override reward.reward_normalization=moving_avg
--override reward.moving_avg_alpha=0.01

# Per-component normalization
--override reward.reward_normalization=per_component

# Combine format decay with moving average
--override reward.format_decay_steps=1000 \
--override reward.reward_normalization=moving_avg

# Full example
accelerate launch --num_processes=2 --mixed_precision=bf16 \
    train.py --config configs/liquidai_3b_config.yaml --custom \
    --override training.max_steps=25000 \
    --override grpo.curriculum_phase2_step=15000 \
    --override reward.format_decay_steps=1000 \
    --override reward.reward_normalization=moving_avg \
    --override logging.wandb_run_name="lfm2-3b-moving-avg"
```

---

## What to Watch in W&B

| Metric | What it tells you |
|--------|-------------------|
| `train/mean_reward` | With moving_avg: should trend upward as baseline pushes performance |
| `train/std_reward` | With per_component: should be more consistent (no format dominance) |
| `train/advantages_std` | Higher = GRPO has more signal. Should be >0.5 for meaningful learning |
| `train/hit_rate` | The only metric that matters — are ducks being hit? |

---

## Status

| Feature | Implemented | Tested | Notes |
|---------|-------------|--------|-------|
| Std floor + advantage clamp | Yes | Yes | Always on, prevents 10⁷ blow-ups |
| Skip update on zero signal | Yes | Yes | Prevents permanent freeze |
| Schulman k3 KL estimator | Yes | Yes | Replaces incorrect mean log-ratio |
| REINFORCE policy loss | Yes | Yes | Cleaner with num_iterations=1 |
| Simplified proximity (hitbox center) | Yes | No | Replaces edge_bonus + hitbox_center_bonus |
| Format decay | Yes | No | `format_decay_steps` config |
| Moving average normalization | Yes | Caused collapse | Now bounded; use `group` by default |
| Per-component normalization | Yes | Partial | Needs format_rewards in batch |
