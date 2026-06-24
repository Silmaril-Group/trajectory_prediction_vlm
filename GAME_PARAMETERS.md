# Game Parameters Reference

Source of truth: `duck_hunt_openenv/server/config.py` and `duck_hunt_openenv/server/game_engine.py`

| Parameter | Value | Source |
|-----------|-------|--------|
| Screen size | 800×500 px | `config.py` |
| VLM render size | 512×512 px | `config.py` |
| FPS | 30 | `config.py` |
| Match duration | 30 seconds (900 frames) | `config.py` |
| Sprite size | 81×75 px | `config.py` |
| Hitbox size | 40×36 px (~5% × 7.2% of screen) | `config.py` |
| Duck speed formula | `randint(BASE+round, BASE+VAR+round) * uniform(0.8, 1.2)` | `game_engine.py` |
| Speed base / variance | 6 / 4 → Round 1: 7–12 px/frame | `config.py` |
| Ducks per match | 2 | `config.py` |
| Bullets per match | 3 | `config.py` |
| Matches per round | 5 | `config.py` |
| Game over condition | 4 misses | `config.py` |
| Max horizon | 30 frames | `config.py` |

## Spawn System

| Parameter | Value |
|-----------|-------|
| Spawn sides | left (40%), right (40%), top (20%) |
| Spawn Y range | 15%–85% of screen height |
| Edge offset (left/right) | 0–60 px beyond edge |
| Edge offset (top) | 0–30 px beyond edge |
| Horizontal speed scale | 70%–100% of full speed (angle variation) |
| Initial dy (left/right) | uniform(-5, 5) px/frame |
| Initial dy (top) | uniform(3, 6) px/frame (always downward) |
| Initial dx (top) | speed × ±1 × uniform(0.3, 0.7) |

## Mid-flight Behavior

| Parameter | Value |
|-----------|-------|
| Jitter chance | 3% per frame |
| Jitter range | ±1 px/frame (dx and dy) |
| Bounce speed jitter | ±15% on wall collisions |
| Bounce dy range | -5 to 5 (sides), 2–5 (top), -5 to -2 (bottom) |
