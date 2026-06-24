# Duck Hunt OpenEnv

A headless Duck Hunt environment for training and evaluating VLM agents.

## Project Structure

```
duck_hunt_openenv/
├── duck_hunt_env/          # Client package
│   ├── __init__.py
│   ├── models.py           # ShootAction, DuckHuntObservation
│   └── client.py           # DuckHuntEnv client wrapper
├── server/                 # Environment server
│   ├── app.py              # FastAPI application
│   ├── environment.py      # DuckHuntEnvironment class
│   ├── game_engine.py      # Duck, Match, Round classes
│   ├── renderer.py         # PIL-based headless rendering
│   ├── config.py           # All game parameters
│   ├── requirements.txt
│   └── Dockerfile
├── assets/                 # Game assets
│   ├── background.jpg
│   └── sprites.png
└── tests/                  # Test suite
    ├── test_game.py        # Unit tests
    └── test_random_agent.py # Integration tests
```

## Installation

```bash
cd duck_hunt_openenv

# Install dependencies
pip install -r server/requirements.txt

# For testing
pip install pytest
```

## Running Tests

### Unit Tests

```bash
# Run all unit tests
pytest tests/test_game.py -v

# Run specific test class
pytest tests/test_game.py::TestDuck -v
pytest tests/test_game.py::TestMatch -v
pytest tests/test_game.py::TestRound -v
pytest tests/test_game.py::TestRenderer -v
pytest tests/test_game.py::TestEnvironment -v

# Run specific test
pytest tests/test_game.py::TestDuck::test_duck_spawn -v

# Run with coverage
pytest tests/test_game.py -v --cov=server --cov-report=term-missing
```

### Integration Tests

```bash
# Run 100 episodes with random agent
python tests/test_random_agent.py

# Run custom number of episodes
python tests/test_random_agent.py --episodes 50

# Run detailed single episode (step-by-step output)
python tests/test_random_agent.py --detailed
```

### Run All Tests

```bash
# Run everything
pytest tests/ -v

# Run with short output
pytest tests/ -q
```

## Running the Server

### Local Development

```bash
cd server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build
docker build -t duck-hunt-openenv -f server/Dockerfile .

# Run
docker run -p 8000:8000 duck-hunt-openenv
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment, return initial observation |
| `/step` | POST | Execute action, return observation |

### Action Format

```json
{
  "x": 400,
  "y": 250,
  "horizon": 5,
  "confidence": "high"
}
```

### Observation Format

```json
{
  "frames": ["base64...", "base64...", "base64...", "base64..."],
  "num_frames": 4,
  "round_number": 1,
  "match_number": 1,
  "ducks_flying": 2,
  "bullets_remaining": 3,
  "match_ducks_hit": 0,
  "round_ducks_hit": 0,
  "total_misses": 0,
  "processing_latency_ms": 200,
  "last_action_result": null,
  "last_ducks_hit": 0,
  "reward": 0.0,
  "done": false
}
```

## Client Usage

```python
from duck_hunt_env.client import DuckHuntEnv
from duck_hunt_env.models import ShootAction

# Connect to server
env = DuckHuntEnv.from_local(host="localhost", port=8000)

# Reset
obs = env.reset()

# Game loop
while not obs.done:
    action = ShootAction(x=400, y=250, horizon=5)
    obs = env.step(action)
    print(f"Result: {obs.last_action_result}, Reward: {obs.reward}")

env.close()
```

## Configuration

All game parameters in `server/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SCREEN_WIDTH` | 800 | Game width |
| `SCREEN_HEIGHT` | 500 | Game height |
| `FPS` | 30 | Frames per second |
| `MATCH_DURATION_SECONDS` | 30 | Match time limit |
| `MATCHES_PER_ROUND` | 5 | Matches per round |
| `DUCKS_PER_MATCH` | 2 | Ducks per match |
| `BULLETS_PER_MATCH` | 3 | Bullets per match |
| `MAX_MISSES` | 4 | Misses for game over |
| `MAX_HORIZON` | 30 | Max horizon frames |
| `FRAME_OUTPUT_SIZE` | (512, 512) | Output image size |
| `FRAMES_PER_OBSERVATION` | 4 | Frames per observation |

## Rewards

| Event | Reward |
|-------|--------|
| Hit | +1.0 |
| Double Kill | +2.5 |
| Miss | -0.3 |
| No Target | -0.5 |
| Horizon Penalty | -0.1 × (horizon / max_horizon) |

## VLM Agent Experiments (Weave)

### Setup

```bash
pip install weave openai wandb
wandb login
export OPENAI_API_KEY="your-key"
```

### Run VLM Agent

```bash
# Single episode
python -m experiments.run --mode single --model gpt-4o

# Batch episodes
python -m experiments.run --mode batch --episodes 10 --model gpt-4o

# Evaluation
python -m experiments.run --mode evaluate --episodes 5

# With options
python -m experiments.run \
  --mode batch \
  --model gpt-4o-mini \
  --episodes 20 \
  --max-steps 100 \
  --project my-duck-hunt-research \
  --log-frames
```

### Python API

```python
import weave
from experiments.agent import DuckHuntVLMAgent
from experiments.episode import run_episode

weave.init("duck-hunt-research")

agent = DuckHuntVLMAgent(model_name="gpt-4o")
result = run_episode(agent)

print(f"Reward: {result.total_reward}")
print(f"Ducks hit: {result.ducks_hit}")
```

### OpenAI Tool Schema

The agent uses function calling with the `shoot` tool:

```json
{
  "name": "shoot",
  "parameters": {
    "x": {"type": "integer", "minimum": 0, "maximum": 800},
    "y": {"type": "integer", "minimum": 0, "maximum": 500},
    "horizon": {"type": "integer", "minimum": 0, "maximum": 30},
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
  },
  "required": ["x", "y", "horizon"]
}
```

### View Results

After running experiments, view traces and results at:
- https://wandb.ai/ (your Weave dashboard)
