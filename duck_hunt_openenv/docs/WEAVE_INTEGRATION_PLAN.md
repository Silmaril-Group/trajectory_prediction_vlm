# Weave Integration Plan for Duck Hunt OpenEnv Research

## Overview

[W&B Weave](https://github.com/wandb/weave) is a toolkit for developing and evaluating AI applications. It's ideal for your VLM agent research because it can:

- **Log images directly** (PIL.Image.Image supported natively)
- **Track VLM API calls** (automatic tracing for OpenAI, Anthropic, etc.)
- **Create systematic evaluations** with custom scorers
- **Version experiments** automatically
- **Compare runs** with rich UI

---

## 1. Installation & Setup

```bash
pip install weave wandb

# Login to W&B (one-time)
wandb login
```

```python
import weave
weave.init("duck-hunt-vlm-research")
```

---

## 2. Track VLM Agent Calls

### Basic Agent Tracking

```python
import weave
from PIL import Image
from anthropic import Anthropic

weave.init("duck-hunt-vlm-research")
client = Anthropic()

@weave.op
def vlm_predict_shot(
    frames: list[Image.Image],
    game_state: dict
) -> dict:
    """VLM predicts where to shoot. Automatically tracked by Weave."""

    # Convert frames to base64 for API
    frame_data = [image_to_base64(f) for f in frames]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                *[{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": d}} for d in frame_data],
                {"type": "text", "text": "Predict where to shoot. Return JSON: {x, y, horizon, confidence}"}
            ]
        }]
    )

    return parse_response(response)
```

Weave automatically captures:
- Input frames (as images in UI!)
- Game state
- VLM response
- Latency & token usage
- Cost

---

## 3. Log Observation Frames

### Direct PIL Image Logging

```python
from dataclasses import dataclass
from PIL import Image
import weave

@dataclass
class Observation:
    frames: list[Image.Image]  # Weave renders these as images!
    game_state: dict
    reward: float

@weave.op
def environment_step(action: dict) -> Observation:
    """Step environment and return observation with frames."""
    obs_dict = env.step(action)

    # Convert base64 back to PIL for logging
    frames = [base64_to_pil(f) for f in obs_dict["frames"]]

    return Observation(
        frames=frames,
        game_state={k: v for k, v in obs_dict.items() if k != "frames"},
        reward=obs_dict["reward"]
    )
```

### Resize Large Images (Recommended)

```python
def resize_frame(image: Image.Image, max_size=(256, 256)) -> Image.Image:
    img = image.copy()
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img

def postprocess_observation(obs: Observation) -> Observation:
    return Observation(
        frames=[resize_frame(f) for f in obs.frames],
        game_state=obs.game_state,
        reward=obs.reward
    )

@weave.op(postprocess_output=postprocess_observation)
def environment_step(action: dict) -> Observation:
    # ... same as above
```

---

## 4. Create Evaluation Framework

### Define Your Dataset

```python
import weave

# Create evaluation scenarios
eval_dataset = [
    {
        "scenario": "two_ducks_flying_right",
        "frames": load_frames("scenarios/two_ducks_right/"),
        "expected_region": {"x_min": 400, "x_max": 600, "y_min": 100, "y_max": 300},
    },
    {
        "scenario": "duck_at_edge",
        "frames": load_frames("scenarios/duck_edge/"),
        "expected_region": {"x_min": 0, "x_max": 100, "y_min": 200, "y_max": 400},
    },
    # ... more scenarios
]
```

### Define Scorers

```python
@weave.op
def accuracy_scorer(output: dict, expected_region: dict) -> dict:
    """Check if predicted shot is in expected region."""
    x, y = output["x"], output["y"]
    in_region = (
        expected_region["x_min"] <= x <= expected_region["x_max"] and
        expected_region["y_min"] <= y <= expected_region["y_max"]
    )
    return {"accuracy": 1.0 if in_region else 0.0}

@weave.op
def horizon_scorer(output: dict) -> dict:
    """Score horizon efficiency (lower is better)."""
    horizon = output.get("horizon", 0)
    # Normalize: 0 horizon = 1.0, 30 horizon = 0.0
    return {"horizon_efficiency": 1.0 - (horizon / 30)}

@weave.op
def confidence_scorer(output: dict) -> dict:
    """Check if confidence matches accuracy."""
    return {"confidence": output.get("confidence", "unknown")}
```

### Run Evaluation

```python
from weave import Evaluation

class VLMAgent(weave.Model):
    model_name: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0

    @weave.op
    def predict(self, frames: list, scenario: str) -> dict:
        # Your VLM prediction logic
        return vlm_predict_shot(frames)

evaluation = Evaluation(
    dataset=eval_dataset,
    scorers=[accuracy_scorer, horizon_scorer, confidence_scorer],
)

agent = VLMAgent()
results = asyncio.run(evaluation.evaluate(agent))
```

---

## 5. Track Full Episodes

### Episode Wrapper

```python
@weave.op
def run_episode(
    agent_config: dict,
    seed: int | None = None
) -> dict:
    """Run a full episode and track everything."""
    env = DuckHuntEnvironment()
    obs = env.reset()

    episode_data = {
        "config": agent_config,
        "seed": seed,
        "steps": [],
        "total_reward": 0.0,
        "ducks_hit": 0,
    }

    while not obs["done"]:
        # Get frames as PIL images for logging
        frames = [base64_to_pil(f) for f in obs["frames"]]

        # VLM prediction (automatically traced)
        action = vlm_predict_shot(frames, obs)

        # Step environment
        obs = env.step(action)

        episode_data["steps"].append({
            "action": action,
            "result": obs["last_action_result"],
            "reward": obs["reward"],
        })
        episode_data["total_reward"] += obs["reward"]

    episode_data["ducks_hit"] = obs["round_ducks_hit"]
    return episode_data
```

### Batch Experiments

```python
@weave.op
def run_experiment(
    experiment_name: str,
    agent_configs: list[dict],
    num_episodes: int = 10
) -> dict:
    """Run multiple episodes across different configurations."""
    results = {}

    for config in agent_configs:
        config_name = config.get("name", str(config))
        config_results = []

        for ep in range(num_episodes):
            result = run_episode(config, seed=ep)
            config_results.append(result)

        results[config_name] = {
            "avg_reward": sum(r["total_reward"] for r in config_results) / len(config_results),
            "avg_ducks_hit": sum(r["ducks_hit"] for r in config_results) / len(config_results),
            "episodes": config_results,
        }

    return results
```

---

## 6. Compare Horizon Strategies

### A/B Test Different Horizons

```python
horizon_configs = [
    {"name": "no_horizon", "horizon_mode": "fixed", "horizon_value": 0},
    {"name": "low_horizon", "horizon_mode": "fixed", "horizon_value": 5},
    {"name": "medium_horizon", "horizon_mode": "fixed", "horizon_value": 15},
    {"name": "high_horizon", "horizon_mode": "fixed", "horizon_value": 30},
    {"name": "adaptive_horizon", "horizon_mode": "vlm_predicted"},
]

results = run_experiment(
    experiment_name="horizon_comparison_v1",
    agent_configs=horizon_configs,
    num_episodes=50
)
```

---

## 7. Custom Metrics Dashboard

### Log Aggregate Metrics

```python
import wandb

@weave.op
def analyze_results(experiment_results: dict) -> dict:
    """Analyze and log summary metrics."""
    summary = {}

    for config_name, data in experiment_results.items():
        summary[config_name] = {
            "avg_reward": data["avg_reward"],
            "avg_ducks_hit": data["avg_ducks_hit"],
            "hit_rate": data["avg_ducks_hit"] / 10,  # max 10 ducks per episode
        }

        # Also log to W&B for charts
        wandb.log({
            f"{config_name}/avg_reward": data["avg_reward"],
            f"{config_name}/hit_rate": summary[config_name]["hit_rate"],
        })

    return summary
```

---

## 8. Project Structure with Weave

```
duck_hunt_openenv/
├── experiments/
│   ├── __init__.py
│   ├── agent.py           # VLM agent with @weave.op
│   ├── episode.py         # Episode runner with @weave.op
│   ├── evaluation.py      # Weave Evaluation setup
│   └── scorers.py         # Custom scorer functions
├── server/                # Environment server (unchanged)
├── duck_hunt_env/         # Client (unchanged)
└── notebooks/
    └── analysis.ipynb     # Query Weave data, visualize
```

---

## 9. Example: Complete Integration

```python
# experiments/run.py
import weave
import asyncio
from PIL import Image

weave.init("duck-hunt-vlm-research")

# --- Agent ---
@weave.op
def vlm_agent(frames: list[Image.Image], state: dict) -> dict:
    """Your VLM agent logic here."""
    # ... call Claude/GPT-4V ...
    return {"x": 400, "y": 250, "horizon": 5, "confidence": "high"}

# --- Episode ---
@weave.op
def run_episode() -> dict:
    from duck_hunt_env.client import DuckHuntEnv

    env = DuckHuntEnv.from_local()
    obs = env.reset()
    total_reward = 0

    while not obs.done:
        frames = [base64_to_pil(f) for f in obs.frames]
        action = vlm_agent(frames, obs.to_dict())
        obs = env.step(ShootAction(**action))
        total_reward += obs.reward

    env.close()
    return {"total_reward": total_reward, "ducks_hit": obs.round_ducks_hit}

# --- Evaluation ---
async def main():
    results = [run_episode() for _ in range(10)]
    print(f"Avg reward: {sum(r['total_reward'] for r in results) / 10}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 10. What You Get in the Weave UI

1. **Trace View**: See every VLM call with input frames rendered as images
2. **Call Hierarchy**: Parent episode → child VLM calls → nested API calls
3. **Metrics**: Automatic latency, token usage, cost tracking
4. **Evaluations**: Compare accuracy/horizon across model versions
5. **Versioning**: Every code change auto-versioned
6. **Export**: Query data via SDK for custom analysis

---

## Key Benefits for Your Research

| Feature | Benefit |
|---------|---------|
| PIL Image logging | See exactly what frames the VLM saw |
| Auto API tracing | Track Claude/GPT-4V calls without extra code |
| Evaluations | Systematic comparison of horizon strategies |
| Versioning | Reproduce any experiment |
| Cost tracking | Monitor API spend per experiment |
| Rich UI | Debug failures by viewing frames + predictions |

---

## Sources

- [W&B Weave GitHub](https://github.com/wandb/weave)
- [Weave Tracing Guide](https://docs.wandb.ai/weave/guides/tracking/tracing)
- [Weave Evaluations](https://docs.wandb.ai/weave/guides/core-types/evaluations/)
- [Weave Media Logging](https://docs.wandb.ai/weave/guides/core-types/media)
- [Weave Quickstart](https://weave-docs.wandb.ai/quickstart/)
