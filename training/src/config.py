"""Training configuration dataclasses for Duck Hunt GRPO training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
@dataclass
class EnvironmentConfig:
    """Duck Hunt OpenEnv parameters."""

    server_host: str = "localhost"
    server_port: int = 8000

    # Screen / rendering
    screen_width: int = 800
    screen_height: int = 500
    frame_output_size: tuple[int, int] = (512, 512)

    # Observation
    frames_per_observation: int = 4
    frames_per_observation_min: int = 2
    frames_per_observation_max: int = 6
    frame_skip: int = 3  # game frames to skip between observation frames

    # Game timing
    fps: int = 30
    max_horizon: int = 30
    latency_options_ms: list[int] = field(
        default_factory=lambda: [100, 200, 300, 400, 500, 600],
    )

    # Derived
    @property
    def latency_frames_range(self) -> list[int]:
        return [int(ms / 1000 * self.fps) for ms in self.latency_options_ms]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Model loading parameters."""

    model_name: str = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = True
    device_map: str = "auto"
    quantization: str | None = None  # "4bit" or "8bit" or None


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
@dataclass
class LoRAConfig:
    """LoRA / PEFT parameters — maps to peft.LoraConfig."""

    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )
    task_type: str = "CAUSAL_LM"
    bias: str = "none"


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------
@dataclass
class GRPOConfig:
    """GRPO-specific parameters — maps to trl.GRPOConfig."""

    # Core GRPO
    num_generations: int = 4
    max_completion_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0

    # Policy optimisation
    beta: float = 0.0  # KL coefficient (0 = no ref model)
    num_iterations: int = 1
    epsilon: float = 0.2
    loss_type: str = "grpo"
    scale_rewards: str = "group"

    # Entropy bonus (prevents mode collapse)
    entropy_coeff: float = 0.03  # weight for entropy bonus in loss

    # Entropy floor (emergency brake against collapse)
    entropy_floor: float = 0.8       # 0 = disabled; if entropy < floor, extra penalty
    entropy_floor_coeff: float = 1.0  # strength of the floor penalty

    # Curriculum: two-phase training
    # Phase 1: horizon fixed to 0, shorter completions (learn x,y aiming)
    # Phase 2: horizon unlocked (learn horizon optimization)
    curriculum_phase2_step: int = 0   # 0 = disabled (no curriculum); >0 = step to unlock horizon
    phase1_max_completion_length: int = 30  # shorter completions in phase 1

    # Early training stabilization
    # Higher grad accumulation in early steps averages over more duck positions,
    # reducing seed sensitivity. Reverts to normal after stabilization_steps.
    stabilization_steps: int = 0          # 0 = disabled; number of steps with boosted accumulation
    stabilization_grad_accum: int = 8     # gradient_accumulation_steps during stabilization
    lora_freeze_steps: int = 0            # 0 = disabled; freeze LoRA for first N steps (collect experience only)

    # vLLM (disabled by default for small-scale)
    use_vllm: bool = False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """General training loop parameters — maps to TrainingArguments fields."""

    output_dir: str = "./outputs/duckhunt_grpo"
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 = use epochs
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # Precision
    bf16: bool = True
    fp16: bool = False

    # Logging
    logging_steps: int = 1
    save_steps: int = 50
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    eval_steps: int = 50

    # Misc
    seed: int = 42
    dataloader_num_workers: int = 4
    gradient_checkpointing: bool = True

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    """Reward function weights and parameters."""

    # Environment rewards (from duck_hunt_openenv config)
    hit: float = 1.0
    double_kill: float = 2.5
    miss: float = -0.1
    shoot_nothing: float = -0.5
    lambda_horizon: float = 0.4
    lambda_horizon_miss: float = 0.2
    bonus_perfect_match: float = 0.5
    bonus_perfect_round: float = 2.0

    # Invalid action (unparseable model output)
    invalid_action: float = -1.0

    # Shooting when no ducks are flying at shot time (escaped/fallen during horizon)
    shoot_dead_duck: float = -0.7

    # Distance-based reward shaping (proximity bonus on miss)
    proximity_bonus: float = 1.0  # max bonus when shot is close to duck
    proximity_decay: float = 3.0  # how fast bonus decays with distance

    # Max horizon (for penalty normalisation)
    max_horizon: int = 30

    # Edge bonus (on hits, reward aiming away from screen center)
    edge_bonus: float = 0.0  # 0 = disabled; e.g. 0.3 = up to +0.3 bonus for edge hits

    # Hitbox center bonus (on hits, reward aiming at center of duck hitbox)
    hitbox_center_bonus: float = 0.0  # 0 = disabled; e.g. 0.5 = max bonus for perfect center hit

    # Skip invalid generations in GRPO (exclude unparseable outputs from advantage computation)
    skip_invalid_generations: bool = False  # true = drop invalid gens instead of giving -1.0

    # Format reward (structured output)
    format_weight: float = 0.3
    format_decay_steps: int = 0  # 0 = no decay; steps over which format_weight decays to 0

    # Accuracy reward (did model parse / predict correctly)
    accuracy_weight: float = 1.0

    # Reward normalization mode: "group" (default), "per_component", "moving_avg"
    reward_normalization: str = "group"
    moving_avg_alpha: float = 0.01  # EMA smoothing factor for moving average baseline

    # Hotspot penalty (anti-exploit)
    hotspot_enabled: bool = True  # set to false to disable hotspot penalty entirely


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
@dataclass
class LoggingConfig:
    """Logging and experiment tracking."""

    wandb_project: str = "duckhunt-grpo"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    report_to: str = "wandb"
    log_completions: bool = True
    num_completions_to_print: int = 5


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------
@dataclass
class HubConfig:
    """Hugging Face Hub upload settings."""

    push_to_hub: bool = False
    hub_model_id: str | None = None  # e.g. "username/duckhunt-ministral-grpo"
    hub_private: bool = True


# ---------------------------------------------------------------------------
# Full config
# ---------------------------------------------------------------------------
@dataclass
class FullConfig:
    """Top-level config that combines all sub-configs."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hub: HubConfig = field(default_factory=HubConfig)

    # ------------------------------------------------------------------
    # YAML loading
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> FullConfig:
        """Load config from a YAML file."""
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        return cls._from_dict(raw)

    @classmethod
    def from_yamls(cls, *paths: str | Path) -> FullConfig:
        """Load and merge multiple YAML files (later files override earlier)."""
        merged: dict[str, Any] = {}
        for p in paths:
            with open(Path(p)) as f:
                layer = yaml.safe_load(f) or {}
            merged = _deep_merge(merged, layer)

        return cls._from_dict(merged)

    # ------------------------------------------------------------------
    # CLI override
    # ------------------------------------------------------------------
    @classmethod
    def with_cli_overrides(
        cls,
        base: FullConfig,
        overrides: dict[str, Any],
    ) -> FullConfig:
        """Apply dot-separated CLI overrides to an existing config.

        Example: ``{"training.learning_rate": 2e-5}``
        """
        raw = _config_to_dict(base)
        for key, value in overrides.items():
            parts = key.split(".")
            d = raw
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

        return cls._from_dict(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> FullConfig:
        env_raw = raw.get("environment", {})
        # tuple special-case
        if "frame_output_size" in env_raw and isinstance(
            env_raw["frame_output_size"], list
        ):
            env_raw["frame_output_size"] = tuple(env_raw["frame_output_size"])

        return cls(
            environment=EnvironmentConfig(**env_raw),
            model=ModelConfig(**raw.get("model", {})),
            lora=LoRAConfig(**raw.get("lora", {})),
            grpo=GRPOConfig(**raw.get("grpo", {})),
            training=TrainingConfig(**raw.get("training", {})),
            reward=RewardConfig(**raw.get("reward", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            hub=HubConfig(**raw.get("hub", {})),
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns new dict)."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _config_to_dict(cfg: FullConfig) -> dict[str, Any]:
    """Convert a FullConfig to a plain dict (one level of nesting)."""
    from dataclasses import asdict

    raw = asdict(cfg)
    # Convert tuples back to lists for YAML compatibility
    if "environment" in raw and "frame_output_size" in raw["environment"]:
        raw["environment"]["frame_output_size"] = list(
            raw["environment"]["frame_output_size"]
        )
    return raw
