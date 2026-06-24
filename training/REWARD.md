# Reward System — Duck Hunt GRPO Training

The total reward the model receives for each generation is a weighted combination of two independent reward functions:

```
total_reward = accuracy_weight * accuracy_reward + format_weight * format_reward
```

Default weights: `accuracy_weight = 1.0`, `format_weight = 0.3`.

## 1. Accuracy Reward

Computed in `src/reward.py`. This is the core game reward — did the shot hit a duck?

### Decision flow

```
Model output
    │
    ├─ Can't parse tool call?          → -1.0  (invalid_action)
    │
    ├─ No ducks flying at obs time?    → -0.5  (shoot_nothing)
    │
    ├─ Ducks escaped during wait?      → -0.7  (shoot_dead_duck)
    │   (had_target at obs, not at shot)
    │
    ├─ Hit both ducks?                 → +2.5 - horizon_penalty + edge_bonus + center_bonus  (double_kill)
    │
    ├─ Hit one duck?                   → +1.0 - horizon_penalty + edge_bonus + center_bonus  (hit)
    │
    └─ Missed?                         → -0.3 - horizon_penalty_miss + proximity_bonus  (miss)
```

### 1.1 Invalid action: `-1.0`

The model produced output that couldn't be parsed into `shoot(x, y, horizon)`. This is the worst possible reward. Examples:

- `"The ducks move in a pattern..."` (text instead of tool call)
- `"[Shoot emoji]"` (no parseable parameters)

**Why -1.0**: must be strictly worse than any valid shot (even a miss at -0.3) so the model always prefers producing a parseable tool call.

### 1.2 Shoot nothing: `-0.5`

The model fired when no ducks were flying at observation time. The model should learn to recognize "no target" states.

### 1.3 Shoot dead duck: `-0.7`

Ducks were flying when the model saw the frames, but by the time the shot lands (`latency_frames + horizon` frames later), all ducks have escaped or fallen. This means the model chose too large a horizon.

**Why -0.7**: worse than a miss (-0.3) because the model wasted a shot on something that was guaranteed to fail. Penalizes excessive horizons. But not as bad as invalid output (-1.0) because the model at least produced a valid tool call.

```
Example:
  - Duck is near top edge, moving up
  - Model predicts horizon=25
  - After 25 frames, duck has escaped off screen
  - Result: shoot_dead_duck = -0.7

  If model had used horizon=5, duck would still be on screen → chance to hit
```

### 1.4 Hit rewards: `+1.0` and `+2.5`

Base reward for hitting one duck (+1.0) or both ducks simultaneously (+2.5). The double kill bonus is 2.5x a single hit because it's much harder — both ducks must be at the same pixel location.

### 1.5 Horizon penalty

**Applied on both hits and misses** with different strengths. Penalizes large horizons to encourage the model to shoot sooner rather than using high horizon as a lottery ticket.

#### On hits:

```
penalty = lambda_horizon * (horizon / max_horizon)
       = 0.4 * (horizon / 30)
```

| Horizon | Penalty | Net single hit | Net double kill |
|---------|---------|---------------|-----------------|
| 0 | 0.000 | +1.000 | +2.500 |
| 5 | 0.067 | +0.933 | +2.433 |
| 10 | 0.133 | +0.867 | +2.367 |
| 15 | 0.200 | +0.800 | +2.300 |
| 20 | 0.267 | +0.733 | +2.233 |
| 30 | 0.400 | +0.600 | +2.100 |

**Why 0.4**: the old value (0.1) was too weak — a hit with h=30 scored +0.9, giving almost no incentive to reduce horizon. At 0.4, a hit with h=30 scores +0.6, creating real pressure to aim with shorter horizons while still keeping any hit better than a miss.

#### On misses:

```
penalty = lambda_horizon_miss * (horizon / max_horizon)
       = 0.2 * (horizon / 30)
```

| Horizon | Penalty | Net miss (before proximity) |
|---------|---------|---------------------------|
| 0 | 0.000 | -0.300 |
| 5 | 0.033 | -0.333 |
| 10 | 0.067 | -0.367 |
| 15 | 0.100 | -0.400 |
| 20 | 0.133 | -0.433 |
| 30 | 0.200 | -0.500 |

**Why penalize horizon on misses**: without this, all misses with the same proximity get identical rewards regardless of horizon. A miss with h=24 is objectively worse than h=2 — the model waited longer and still missed, making the prediction harder. This creates gradient signal to reduce horizon even when the model is mostly missing (e.g. 15% hit rate), which is when horizon optimization matters most.

**Why 0.2 (lower than hits)**: misses are already negative, so we don't need as strong a penalty. The miss with h=30 goes from -0.3 to -0.5, which is meaningful but doesn't approach `shoot_dead_duck` (-0.7) territory.

### 1.6 Miss: `-0.3 - horizon_penalty_miss`

The shot was valid, ducks were flying, but the shot didn't hit any duck's hitbox. The base miss reward is -0.3, further penalized by horizon (see section 1.5).

**Why -0.3 base (not -1.0)**: a miss with a valid tool call is much better than invalid output. The model at least produced the right format and attempted to aim. The proximity bonus (below) provides gradient signal to improve aim.

**Effective miss range**: from -0.3 (h=0) to -0.5 (h=30), before proximity bonus.

### 1.7 Proximity bonus (target-aware)

**Only on misses. Only measures distance to FLYING ducks.**

If one duck is falling/dead and the other is still flying, the proximity bonus only rewards aiming near the flying duck. Aiming at a dead duck gives no proximity signal — the model learns to ignore it and target the alive one.

```
proximity = proximity_bonus * exp(-proximity_decay * min_distance_to_flying_duck)
          = 0.3 * exp(-5.0 * d)
```

| Distance (norm) | Approx pixels | Proximity bonus | Total miss reward |
|----------------|---------------|----------------|-------------------|
| 0.00 | 0 px | +0.300 | +0.000 |
| 0.05 | ~40 px | +0.234 | -0.066 |
| 0.10 | ~80 px | +0.182 | -0.118 |
| 0.20 | ~160 px | +0.110 | -0.190 |
| 0.30 | ~240 px | +0.067 | -0.233 |
| 0.50 | ~400 px | +0.025 | -0.275 |
| 1.00 | ~800 px | +0.002 | -0.298 |

**Target-aware behavior by duck state:**

| Duck A state | Duck B state | Proximity measures distance to |
|---|---|---|
| flying | flying | nearest of A or B |
| flying | falling/escaped | A only |
| falling/escaped | flying | B only |
| falling/escaped | falling/escaped | no proximity bonus (both dead) |

**Why this matters**: without proximity bonus, all misses get the same -0.3 regardless of whether the shot was 10px or 400px from the duck. GRPO needs reward variance across generations to compute advantages. If 5 out of 6 generations miss by different amounts, the proximity bonus creates different rewards for each, giving the optimizer a gradient toward the closer miss.

**The decay rate (5.0)** is tuned so that:
- Within ~50px (one duck-width): meaningful bonus (+0.2)
- Beyond ~200px: negligible bonus (<0.1)
- At max distance (diagonal): effectively zero

## 2. Format Strictness (per model family)

Each model family has a strict parser in `src/formats.py`. Outputs that don't match the model's native format are rejected as **invalid** (accuracy = -1.0, format = 0.0).

**LiquidAI** accepts only pythonic format:
- `<|tool_call_start|>[shoot(x=0.5, y=0.3, horizon=8)]<|tool_call_end|>` — primary
- `shoot(x=0.5, y=0.3, horizon=8)` — plain pythonic (no special tokens)

Mistral-style JSON (`[{"name": "shoot", "arguments": {...}}]`) is **intentionally rejected** even though it contains valid coordinates. Without this, the model gets rewarded for using the wrong format, removing the learning signal to produce its native tool call syntax.

**Mistral** accepts its native format plus JSON/KV fallbacks (Mistral models occasionally vary their JSON structure).

## 3. Format Reward

Computed in `src/dataset.py → make_format_reward_function()`. Scores the structural quality of the output.

### Decision flow

```
Model output
    │
    ├─ Can't parse tool call?          → 0.0
    │
    ├─ Valid call, clean (≤5 extra chars)?  → 1.0
    │
    └─ Valid call + extra text?        → max(0.3, 1.0 - extra_chars * 0.01)
```

### 3.1 No tool call: `0.0`

The model didn't produce any parseable function call. Different from accuracy's -1.0 because format reward is scaled by `format_weight = 0.3`, so the total contribution is `0.3 * 0.0 = 0.0`.

### 3.2 Clean tool call: `1.0`

The model produced a valid tool call with no extra text:
```
<|tool_call_start|>[shoot(x=0.5, y=0.3, horizon=8)]<|tool_call_end|>
```

This is the maximum format reward. Special tokens (`<|pad|>`, `<|im_end|>`) don't count as extra text.

### 3.3 Verbose tool call: `0.3–0.9`

The model produced a valid tool call but also generated explanatory text:
```
The ducks are moving to the right. <|tool_call_start|>[shoot(x=0.5, y=0.3, horizon=8)]<|tool_call_end|>
```

Penalty scales linearly with extra character count:

| Extra chars | Format reward | Example |
|------------|--------------|---------|
| 0–5 | 1.0 | Just the tool call |
| 20 | 0.8 | Short prefix |
| 50 | 0.5 | One sentence of explanation |
| 70+ | 0.3 (floor) | Paragraph of reasoning |

**Why penalize verbosity**: extra tokens waste compute and slow inference. The model should learn to respond with just the tool call. The floor at 0.3 ensures that a verbose-but-valid response is still better than no tool call (0.0).

## 4. Hotspot Penalty (target-aware anti-exploit)

Implemented in `src/trainer.py`. Prevents the model from exploiting a fixed position (e.g. always shooting screen center) while allowing legitimate aiming at ducks that happen to be near center.

### When it activates

The hotspot penalty **only applies when the shot is far from any duck** (`min_distance >= 0.3`). If the shot is close to a duck (distance < 0.3), the hotspot penalty is skipped entirely — the model is legitimately aiming at a duck, not exploiting a fixed position.

This is critical because ducks often fly through the center of the screen. Without the target-aware check, the penalty punishes the model for correctly aiming at ducks that happen to be near center, causing hits to get negative rewards and GRPO to learn to **avoid** ducks.

### How it works

The trainer tracks the last 50 shot positions. For shots that are **far from any duck** (distance ≥ 0.3) with positive reward, it calculates the **concentration** — what fraction of recent shots are within 0.10 normalized distance (~80px) of the current shot.

```
# Only if shot is NOT near a duck (distance >= 0.3):
concentration = nearby_shots / total_tracked_shots

if concentration > 0.2:
    scale = 1.0 - (concentration - 0.2) * 5.0
    reward = reward * max(scale, -1.5)
```

| Concentration | Scale | Hit (+1.0) becomes | Effect |
|---|---|---|---|
| <20% | 1.0 | +1.0 | No penalty |
| 20% | 1.0 | +1.0 | Threshold |
| 30% | 0.5 | +0.5 | Reduced |
| 40% | 0.0 | **0.0** | Hit is worthless |
| 50% | -0.5 | **-0.5** | Hit is punished |
| 60% | -1.0 | **-1.0** | Same as invalid output |
| 80%+ | -1.5 (floor) | **-1.5** | Worse than anything |

### Summary by scenario

| Shot near duck? | In hotspot? | Hotspot applies? | Result |
|---|---|---|---|
| Yes (dist < 0.3) | No | No | Full reward — legitimate aim |
| Yes (dist < 0.3) | Yes | **No** | Full reward — duck is actually there |
| No (dist ≥ 0.3) | No | No | Normal reward |
| No (dist ≥ 0.3) | Yes | **Yes** | Penalty — blind position spam |

### Configuration

```yaml
reward:
  hotspot_enabled: true  # set to false to disable entirely
```

**Parameters** (hardcoded in trainer):
- `hotspot_window`: 50 (track last N shots)
- `hotspot_radius`: 0.10 (~80px zone)
- `near_duck_threshold`: 0.3 (shots closer than this skip hotspot)
- Scale: linear from 1.0 at 20% to -1.5 floor at 50%+

## 4.5 Edge Bonus (anti-center-bias)

Computed in `src/reward.py`. **Only on hits.** Gives a bonus for hitting ducks away from the screen center, incentivizing the model to track ducks across the entire screen rather than parking at center.

```
center_distance = distance(shot_pos, (0.5, 0.5))
edge_bonus = edge_bonus_max * min(center_distance / 0.4, 1.0)
```

| Shot position | Distance from center | Edge bonus (max=0.3) | Total hit reward |
|---|---|---|---|
| (0.50, 0.50) center | 0.00 | +0.00 | +1.00 |
| (0.40, 0.40) near center | 0.14 | +0.11 | +1.11 |
| (0.30, 0.30) mid-screen | 0.28 | +0.21 | +1.21 |
| (0.15, 0.25) edge | 0.43 | +0.30 (capped) | +1.30 |

**Why this helps**: without edge bonus, center hits and edge hits are worth the same (+1.0). The model rationally prefers center because ducks pass through center more often. With edge bonus, an edge hit is worth +1.30 — GRPO sees the edge-hitting generation as better and reinforces tracking behavior over center-camping.

**Configuration:**
```yaml
reward:
  edge_bonus: 0.3  # 0 = disabled; max bonus for edge hits
```

## 4.6 Hitbox Center Bonus (precision reward)

Computed in `src/reward.py`. **Only on hits.** Rewards the model for hitting closer to the center of the duck's hitbox, encouraging precision over just "anywhere inside the hitbox."

```
duck_center = (duck_pos + hitbox_size / 2) in normalized coords
dist_to_center = distance(shot_pos, duck_center)
max_dist = half-diagonal of hitbox in normalized coords (~0.086)
hitbox_center_bonus = bonus_max * max(0, 1.0 - dist_to_center / max_dist)
```

| Shot placement | Distance to duck center | Bonus (max=0.5) | Total hit reward |
|---|---|---|---|
| Dead center of duck | 0.00 | +0.50 | +1.50 |
| Quarter from center | ~0.02 | +0.38 | +1.38 |
| Half from center | ~0.04 | +0.25 | +1.25 |
| Edge of hitbox | ~0.086 | +0.00 | +1.00 |

**Why this helps**: without this bonus, all hits get +1.0 regardless of precision. Two generations that both hit the same duck get identical rewards — GRPO sees no difference. With the center bonus, a precise hit gets +1.5 while an edge hit gets +1.0. GRPO reinforces the more precise generation.

**Configuration:**
```yaml
reward:
  hitbox_center_bonus: 0.5  # 0 = disabled; max bonus for center-of-duck hits
```

## 4.7 Generation Filtering (noise reduction)

The trainer can optionally **skip noisy generations** from the GRPO advantage computation.

### Skip invalid generations

When `skip_invalid_generations: true`, generations with unparseable output (safety refusals, text explanations, wrong format) are dropped from the GRPO batch entirely. Only generations with valid `shoot(x, y)` calls contribute to advantages.

**Why**: once the model already produces valid tool calls 95%+ of the time, the occasional refusal creates a huge reward outlier (-1.0 vs ~-0.1) that dominates the advantage computation. The gradient says "don't refuse" instead of "aim better."

### No-target filtering (automatic, upstream)

`shoot_nothing` and `shoot_dead` outcomes are **prevented at the source**: `_collect_batch` retries until a flying duck is found at observation time AND will still be flying after `latency_frames`. The simulation uses the snapshot + RNG state to forecast duck positions deterministically.

Rather than dropping bad generations after the fact, they never enter the pipeline.

### How filtering works

```
6 generations → filter → 5 remaining → compute advantages on 5

Example:
  gen 0: miss (valid)              → KEEP
  gen 1: hit  (valid)              → KEEP
  gen 2: "I'm sorry..." (invalid)  → SKIP (skip_invalid_generations)
  gen 3: miss (valid)              → KEEP
  gen 4: miss (valid)              → KEEP
  gen 5: hit  (valid)              → KEEP

  GRPO computes advantages over [gen 0, 1, 3, 4, 5] only
```

If ALL generations would be filtered, no filtering is applied (keeps the full batch to avoid empty batches).

**Configuration:**
```yaml
reward:
  skip_invalid_generations: true   # drop unparseable outputs from GRPO
```

## 5. Combined Reward

The total reward per generation is:

```
total = 1.0 * accuracy_reward + 0.3 * format_reward
```

Then the hotspot penalty may scale it (only on positive rewards):

```
if concentration > 0.2 AND total > 0:
    scale = 1.0 - (concentration - 0.2) * 5.0
    total = total * max(scale, -1.5)
```

### Example scenarios

| Scenario | Accuracy | Format | Hotspot | Total |
|----------|---------|--------|---------|-------|
| Hit duck, clean output, h=5 | +0.933 | 1.0 | none | +1.233 |
| Hit duck, clean, h=5, **30% conc.** | +0.933 | 1.0 | scale=0.5 | **+0.617** |
| Hit duck, clean, h=5, **50% conc.** | +0.933 | 1.0 | scale=-0.5 | **-0.617** |
| Hit duck, verbose output, h=5 | +0.933 | 0.5 | none | +1.083 |
| Close miss to flying duck (d=0.05), h=5, clean | -0.099 | 1.0 | — | +0.201 |
| Close miss to flying duck (d=0.05), h=20, clean | -0.199 | 1.0 | — | +0.101 |
| Close miss to dead duck (d=0.05), h=5, clean | -0.333 | 1.0 | — | -0.033 |
| Far miss (d=0.5), h=0, clean | -0.275 | 1.0 | — | +0.025 |
| Far miss (d=0.5), h=20, clean | -0.408 | 1.0 | — | -0.108 |
| Miss, no tool call | -1.0 | 0.0 | — | -1.000 |
| Miss, verbose but parseable, h=15 | -0.400 | 0.5 | — | -0.250 |
| Duck escaped during horizon | -0.700 | 1.0 | — | -0.400 |
| Double kill, h=0 | +2.500 | 1.0 | none | +2.800 |
| Double kill, h=0, **40% conc.** | +2.500 | 1.0 | scale=0.0 | **0.000** |

### Key invariants

These must always hold for the reward system to provide correct learning signal:

```
double_kill > hit > close_miss > far_miss > shoot_nothing > shoot_dead > invalid
  +2.5      +1.0    ~-0.1       ~-0.3...-0.5  -0.5           -0.7        -1.0

hit(h=30) > miss                   →  lowest hit (0.6) still better than best miss (0.0)
novel_hit > hotspot_hit(30%)       →  diverse aiming rewarded over position spam
hotspot_hit(50%) < miss            →  severe spam is worse than missing
clean > verbose                    →  format reward always higher for concise output
invalid < miss                     →  always better to produce a parseable tool call
miss_near_flying > miss_near_dead  →  proximity only to alive ducks
```

## 6. How Rewards Drive GRPO Learning

GRPO samples G generations (default 6) for each game state and computes group-normalized advantages:

```
advantages = (rewards - mean(rewards)) / std(rewards)
```

**What matters is reward VARIANCE across generations, not absolute values.**

If all 6 generations get the same reward (e.g. all miss by the same distance), `std = 0` and advantages are zero — no learning happens. The model needs different generations to produce different rewards.

### Sources of reward variance

| Source | How it creates variance |
|--------|----------------------|
| Hit vs miss | +1.0 vs -0.3 = huge signal |
| Different horizons on hits | h=3 (+0.96) vs h=20 (+0.73) = strong signal |
| Different horizons on misses | h=3 (-0.32) vs h=20 (-0.43) = useful variance |
| Proximity bonus on misses (flying only) | close miss (-0.1) vs far miss (-0.28) = moderate signal |
| Aiming at flying vs dead duck | proximity bonus (-0.1) vs no bonus (-0.3) = teaches target selection |
| Format quality | clean (1.0) vs verbose (0.5) = moderate signal |
| Dead duck penalty | miss (-0.3) vs escaped (-0.7) = useful signal for horizon tuning |
| Hotspot penalty | novel hit (+1.0) vs hotspot hit at 40% (0.0) = strong anti-exploit |

### Common failure mode

If the model collapses to one output (e.g. always `shoot(x=0.5, y=0.5, horizon=8)`), all 6 generations get identical rewards. `std = 0`, advantages = 0, no gradient. Multiple mechanisms prevent this:
- Entropy bonus/floor keeps the policy stochastic
- Hotspot penalty makes repeated-position hits negative at >40% concentration
- New match every step ensures diverse duck positions (see section 7)

## 7. Environment Diversity

The trainer forces **a new match every training step**. This means every step has fresh ducks with new random spawn positions, speeds, and directions. Without this, the same two ducks would persist for 15-60 steps while bouncing around, causing the model to see similar frames repeatedly.

Each step:
1. Advance to next match (new pair of ducks spawned off-screen)
2. Advance 8-25 random frames (ducks enter the visible area)
3. Verify at least 1 duck is flying
4. Capture frames and train

Additional diversity mechanisms:
- Ducks spawn from left, right (40% each) or top (20%) edges
- Per-duck speed with +/-20% individual variation
- Random starting round (1-5) on reset — varied speed ranges
- Mid-flight jitter: 3% chance per frame of direction nudge
- Bounce speed jitter: +/-15% on wall bounce

## 8. Curriculum Training

Training is split into two phases to help the model learn incrementally:

### Phase 1: Learn to aim (steps 0 to `curriculum_phase2_step`)

- **No horizon in tool schema** (LiquidAI): the tool call is `shoot(x, y)` only — horizon parameter is completely removed from the schema, system prompt, and few-shot example. The model focuses entirely on learning format and spatial aiming.
- **Horizon always 0**: even if the model somehow outputs a horizon value, it is ignored and set to 0.
- **Shorter completions**: `phase1_max_completion_length` (default 30 tokens) — the simpler tool call `shoot(x=0.5, y=0.3)` needs fewer tokens than `shoot(x=0.5, y=0.3, horizon=8)`.
- **System prompt**: "Predict where the duck will be after latency frames." (no mention of horizon)
- **Goal**: model learns to produce valid tool calls and aim at duck positions
- **Difficulty**: easier — shot lands after processing latency only, no extra prediction needed
- **Recommended duration**: 10,000–15,000 steps. The model needs enough steps to reach a solid hit rate (30%+) before horizon is introduced.

### Phase 2: Learn horizon optimization (steps `curriculum_phase2_step` to end)

- **Horizon added to tool schema**: the tool call becomes `shoot(x, y, horizon)`. The schema, system prompt, and few-shot are all updated to include horizon.
- **Horizon unlocked**: model's predicted horizon is used in simulation
- **Full completion length**: `max_completion_length` (default 40 tokens) — room for the longer tool call with horizon
- **System prompt**: "Predict where the duck will be after latency + horizon frames."
- **Goal**: model learns to optimize when to shoot
- **Difficulty**: harder — must predict future duck position

### Phase transition details (LiquidAI)

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| Tool schema | `shoot(x, y)` | `shoot(x, y, horizon)` |
| System prompt | "after latency frames" | "after latency + horizon frames" |
| Few-shot | `[shoot(x=0.45, y=0.32)]` | `[shoot(x=0.45, y=0.32, horizon=8)]` |
| Max tokens | 30 | 40 |
| Horizon in reward | always 0 | model's predicted value |

Note: Mistral format does not change between phases (horizon is always present in the schema). The phase-aware tool schema is LiquidAI-specific.

### Why curriculum helps

The model needs to learn three things simultaneously: tool call format, spatial aiming, and temporal prediction. Without curriculum, the model often gets stuck learning one (usually tool call format) while failing at the others, leading to collapse. Curriculum separates the easier task (aiming) from the harder one (timing).

Without Phase 1, the model tends to learn a degenerate strategy: shoot center with high horizon and hope ducks wander into the shot. With 10k+ steps of horizon-free Phase 1, the model is forced to learn actual duck tracking before horizon is introduced.

```yaml
grpo:
  curriculum_phase2_step: 15000   # 0 = disabled; recommended 10000-15000
  phase1_max_completion_length: 30
  max_completion_length: 40       # used in phase 2
```

W&B logs `train/curriculum_phase` (1 or 2) so you can see the transition.

## 9. Early Training Stabilization

LoRA weights start at zero — the first gradient updates have outsized influence on the training direction. If the first few steps happen to have ducks clustered in one area, the model develops a bias that takes hundreds of steps to unlearn. This causes different hit rate trajectories across runs with different seeds.

Three mechanisms reduce this seed sensitivity:

### 9.1 LoRA freeze

LoRA parameters are frozen for the first `lora_freeze_steps` (default 50). The model runs forward passes and collects batches but no weight updates occur. This lets the optimizer/scheduler warm up and populates the hotspot tracker before any learning signal reaches the weights.

### 9.2 Stabilization (boosted gradient accumulation)

For the first `stabilization_steps` (default 500), `gradient_accumulation_steps` is temporarily increased to `stabilization_grad_accum` (default 8, effective batch = 16). Each weight update averages over more duck positions, reducing the influence of any single state.

After `stabilization_steps`, accumulation reverts to the normal value (default 4, effective batch = 8).

### 9.3 Warmup

`warmup_ratio: 0.10` gradually ramps the learning rate from ~0 to the target over 10% of total steps. Early gradients (which are high-variance due to random game states) produce smaller weight updates.

### Timeline

```
Step 0        50          500                          15000              25000
  |-- frozen --|--- stabilization (grad_accum=8) ---|
  |------------ warmup (LR ramp) ----------|
  |------------ Phase 1: shoot(x, y) only ---------|-- Phase 2: shoot(x, y, horizon) --|
```

### W&B tracking

| Metric | Values | Description |
|--------|--------|-------------|
| `train/lora_frozen` | 0 or 1 | Whether LoRA is currently frozen |
| `train/grad_accum` | 4 or 8 | Current gradient accumulation steps |
| `train/curriculum_phase` | 1 or 2 | Curriculum phase |

```yaml
grpo:
  lora_freeze_steps: 50          # 0 = disabled
  stabilization_steps: 500       # 0 = disabled
  stabilization_grad_accum: 8    # grad_accum during stabilization

training:
  warmup_ratio: 0.10
```

## 11. Training Usage

### LFM2.5-VL-1.6B — single run

```bash
python train.py --config configs/liquidai_config.yaml --custom \
    --override training.max_steps=25000 \
    --override grpo.curriculum_phase2_step=15000
```

Phase 1 (steps 0–14999): `shoot(x, y)` only, no horizon. Phase 2 (steps 15000–24999): `shoot(x, y, horizon)`.

### LFM2-VL-3B — full training

```bash
python train.py --config configs/liquidai_3b_config.yaml --custom \
    --override training.max_steps=25000 \
    --override grpo.curriculum_phase2_step=15000 \
    --override training.warmup_ratio=0.03 \
    --override training.learning_rate=2e-5 \
    --override grpo.num_generations=6 \
    --override reward.hotspot_enabled=true \
    --override reward.edge_bonus=0.3 \
    --override reward.hitbox_center_bonus=0.5 \
    --override reward.skip_invalid_generations=true \
    --override reward.format_weight=0.0 \
    --override logging.wandb_run_name="lfm2-3b-grpo" \
    --push-to-hub --hub-model-id dmayboroda/duckhunt_liquidai_3b_grpo
```

### LFM2-VL-3B — resume from checkpoint (precision training)

Resume from a checkpoint with good hit rate (e.g. 30%) and switch to precision-focused rewards:

```bash
python train.py --config configs/liquidai_3b_config.yaml --custom \
    --override training.max_steps=25000 \
    --override grpo.curriculum_phase2_step=15000 \
    --override training.warmup_ratio=0.03 \
    --override training.learning_rate=2e-5 \
    --override grpo.num_generations=6 \
    --override reward.hotspot_enabled=true \
    --override reward.edge_bonus=0.3 \
    --override reward.hitbox_center_bonus=0.5 \
    --override reward.skip_invalid_generations=true \
    --override reward.format_weight=0.0 \
    --override logging.wandb_run_name="lfm2-3b-precision-resume" \
    --push-to-hub --hub-model-id dmayboroda/duckhunt_liquidai_3b_grpo \
    --resume outputs/lfm2_3b_duckhunt_grpo/checkpoint-XXXX
```

Replace `checkpoint-XXXX` with the checkpoint path.

### Two-phase approach (train aiming, then resume with precision)

Phase 1 — basic aiming with format reward:
```bash
python train.py --config configs/liquidai_3b_config.yaml --custom \
    --override training.max_steps=15000 \
    --override grpo.curriculum_phase2_step=99999 \
    --override reward.format_weight=0.3
```

Phase 2 — resume with precision rewards, no format reward:
```bash
python train.py --config configs/liquidai_3b_config.yaml --custom \
    --override training.max_steps=10000 \
    --override grpo.curriculum_phase2_step=0 \
    --override reward.format_weight=0.0 \
    --override reward.hitbox_center_bonus=0.5 \
    --override reward.skip_invalid_generations=true \
    --resume outputs/lfm2_3b_duckhunt_grpo/checkpoint-XXXX
```

### Multi-GPU training (2x A100)

Uses HuggingFace Accelerate with DDP. Each GPU processes its own game state independently — 2x more diverse training data per optimizer step. Gradients are averaged across GPUs automatically.

```bash
accelerate launch --config_file configs/accelerate_2gpu.yaml \
    train.py --config configs/liquidai_3b_config.yaml --custom \
    --override training.max_steps=25000 \
    --override grpo.curriculum_phase2_step=15000 \
    --override logging.wandb_run_name="lfm2-3b-2gpu" \
    --push-to-hub --hub-model-id dmayboroda/duckhunt_liquidai_3b_grpo
```

Or inline without a config file:
```bash
accelerate launch --num_processes=2 --mixed_precision=bf16 \
    train.py --config configs/liquidai_3b_config.yaml --custom \
    --override training.max_steps=25000 \
    --override grpo.curriculum_phase2_step=15000
```

**How it works:**
- Each GPU gets its own environment instance with a different random seed
- Each GPU generates G completions from a different game state
- Gradients are averaged across GPUs via DDP
- W&B logging and checkpointing only happen on rank 0
- `device_map="auto"` is automatically disabled (conflicts with DDP)

**Effective batch size with 2 GPUs:**
- 1 GPU: `batch_size × grad_accum = 1 × 8 = 8` game states per optimizer step
- 2 GPU: `batch_size × grad_accum × 2 = 1 × 8 × 2 = 16` game states per optimizer step

**Accelerate config file** (`configs/accelerate_2gpu.yaml`):
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
num_machines: 1
num_processes: 2
```

**Single GPU still works** — no code changes needed:
```bash
python train.py --config configs/liquidai_3b_config.yaml --custom ...
```

The trainer auto-detects `WORLD_SIZE > 1` and only uses Accelerate when distributed.

### CLI overrides

Any config value can be overridden with `--override section.key=value`:
```bash
--override grpo.curriculum_phase2_step=10000
--override training.learning_rate=3e-5
--override grpo.num_generations=8
--override reward.lambda_horizon=0.3
--override reward.hitbox_center_bonus=0.5
--override reward.skip_invalid_generations=true
```

## 12. Config Reference

```yaml
reward:
  hit: 1.0                  # base reward for hitting one duck
  double_kill: 2.5           # base reward for hitting both ducks
  miss: -0.3                 # base reward for missing
  shoot_nothing: -0.5        # shooting when no ducks flying
  shoot_dead_duck: -0.7      # ducks escaped during latency+horizon wait
  invalid_action: -1.0       # unparseable model output
  lambda_horizon: 0.4        # horizon penalty coefficient (on hits)
  lambda_horizon_miss: 0.2   # horizon penalty coefficient (on misses)
  proximity_bonus: 0.3       # max proximity bonus (on misses only)
  proximity_decay: 5.0       # how fast proximity bonus decays with distance
  edge_bonus: 0.3            # 0 = disabled; max bonus for hitting ducks away from center
  hitbox_center_bonus: 0.5   # 0 = disabled; max bonus for hitting center of duck
  hotspot_enabled: true      # false to disable hotspot penalty entirely
  skip_invalid_generations: true   # drop unparseable outputs from GRPO
  format_weight: 0.0         # 0 when model already learned tool calls
  format_decay_steps: 0      # 0 = no decay; N = linearly decay to 0 over N steps
  accuracy_weight: 1.0       # weight for accuracy reward in total
  max_horizon: 30            # max horizon for penalty normalization
  reward_normalization: "group"   # group (default) / moving_avg / per_component
  moving_avg_alpha: 0.01     # EMA smoothing for moving_avg mode

grpo:
  curriculum_phase2_step: 15000    # 0 = disabled; step to unlock horizon
  phase1_max_completion_length: 30 # shorter completions in phase 1
  max_completion_length: 40        # full completions in phase 2
  stabilization_steps: 500         # 0 = disabled; boosted grad_accum duration
  stabilization_grad_accum: 8      # grad_accum during stabilization
  lora_freeze_steps: 50            # 0 = disabled; freeze LoRA for N steps
  beta: 0.01                       # KL anchor (k3 estimator)
  entropy_coeff: 0.02
  entropy_floor: 0.5               # emergency brake on entropy collapse
  entropy_floor_coeff: 1.0

training:
  learning_rate: 5.0e-6            # safer than 2e-5 for LoRA on VLMs
  max_grad_norm: 0.3               # tighter than 1.0 prevents loss spikes
  warmup_ratio: 0.10               # LR warmup fraction of total optimizer steps
```

## 13. Tuning Guide

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Model never produces tool calls | `invalid_action` not negative enough, or format reward weight too low | Increase `format_weight` to 0.5 |
| Model always shoots center | Hitbox too large, or not enough game state diversity | Reduce hitbox, advance more frames between steps |
| Model uses horizon=0 always | `lambda_horizon` too high | Reduce to 0.05 |
| Model uses horizon=30 always | No penalty for dead ducks, or `lambda_horizon` too low | Check `shoot_dead_duck`, increase `lambda_horizon` and `lambda_horizon_miss` |
| All generations get same reward | Model collapsed, or proximity bonus too small | Increase `proximity_bonus`, check entropy config |
| Hit rate plateau | Proximity decay too slow (bonus too easy) | Increase `proximity_decay` to 8.0 |
| Model writes explanations | Format verbosity penalty too weak | Reduce format floor from 0.3 to 0.1 |
| Model always shoots center (with hotspot ON) | Hotspot penalty killing legitimate hits near center | Verify hotspot is target-aware (dist < 0.3 skips penalty); increase `edge_bonus` |
| Model always shoots center (with hotspot OFF) | No penalty for center-camping | Enable hotspot (`hotspot_enabled: true`) and/or increase `edge_bonus` |
| Hits get negative rewards from hotspot | Hotspot not target-aware, punishing duck-tracking | Upgrade hotspot to check `min_distance < 0.3` before penalizing (see section 4) |
| Model ignores dead ducks | Working correctly — proximity only targets flying ducks | No fix needed |
| Model shoots dead ducks | `shoot_dead_duck` penalty too weak | Increase to -0.9 |
| Same duck positions every step | Environment not advancing to new match | Verify `auto_advance_to_next_match()` is called each step |
| Hit rate stalls in phase 1 | Model learned aiming but needs horizon | Transition to phase 2 earlier (lower `curriculum_phase2_step`) |
| Hit rate drops at phase 2 start | Normal — model adjusting to harder task | Let it run 500+ steps, should recover |
| Different hit rates across runs (<1k steps) | Seed sensitivity — early duck positions bias LoRA direction | Increase `stabilization_steps` and `stabilization_grad_accum`, or increase `lora_freeze_steps` |
| Early training unstable / spiky loss | Gradients too noisy from random game states | Increase `warmup_ratio` to 0.15, increase `stabilization_grad_accum` to 16 |
| Hit rate flat at ~20% (3B model) | Hotspot penalty killing legit hits; no precision reward | Disable hotspot or use target-aware version; add `hitbox_center_bonus=0.5` |
| Model hits but doesn't improve precision | All hits get same reward regardless of accuracy | Add `hitbox_center_bonus=0.5` — center hits get more reward than edge hits |
| Safety refusals dominating GRPO signal | -1.0 invalid reward creates huge outlier advantage | Enable `skip_invalid_generations=true` |
| Model penalized for ducks escaping during latency | No longer possible | Auto-handled: `_collect_batch` retries with upstream latency validation |
| Model already knows format but format_weight > 0 | Format reward is constant for all gens, adds no GRPO signal | Set `format_weight=0.0` for precision-focused training |
| Multi-GPU: `device_map="auto"` error with DDP | device_map uses model parallelism, DDP needs full model per GPU | Auto-handled: trainer disables device_map when `WORLD_SIZE > 1` |
| Multi-GPU: `model.generate()` fails | DDP wrapper doesn't support generate | Auto-handled: trainer uses `_unwrap_model().generate()` |
| Multi-GPU: duplicate W&B logs | All ranks trying to log | Auto-handled: only rank 0 logs to W&B and saves checkpoints |
| Loss spikes / gradient blow-ups | LoRA LR too high or grad clip too loose | Lower `learning_rate` to 5e-6, set `max_grad_norm=0.3` |
| Advantages of 10⁶+ in W&B | std≈0 → division blow-up | Auto-handled: std floor 0.1 + clamp ±5 in trainer |
| Training "freezes" with zero advantages | All gens got same reward, GRPO has no signal | Auto-handled: trainer skips backward when std<1e-6 |
| Model anti-anchored from reference (KL goes negative) | Wrong KL formula | Auto-handled: replaced with Schulman k3 estimator (always ≥ 0) |
| Entropy crashes to 0 / model collapses | No entropy floor | Set `entropy_floor: 0.5`, `entropy_floor_coeff: 1.0` |
