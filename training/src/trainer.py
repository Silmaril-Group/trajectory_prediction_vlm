"""Custom GRPO training loop for Duck Hunt VLM training.

Fallback for when TRL's ``GRPOTrainer`` does not handle the
Mistral-3 multimodal + environment-reward workflow cleanly.

Implements the core GRPO algorithm:

1. For each prompt, sample ``G`` completions.
2. Score each completion with the environment reward.
3. Normalise advantages within the group.
4. Compute the clipped surrogate objective.
5. (Optionally) add KL penalty against a frozen reference model.
"""

from __future__ import annotations

import copy
import json
import logging
import random

from PIL import Image
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_scheduler

from .config import FullConfig
from .dataset import capture_snapshot, simulate_shot
from .environment import DuckHuntEnvWrapper
from .reward import compute_reward, compute_reward_detailed
from .utils import Action, build_prompt, parse_tool_call

logger = logging.getLogger(__name__)

try:
    import wandb

    _WANDB = True
except ImportError:
    _WANDB = False


# ===================================================================
#  Helpers
# ===================================================================
def _log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Token-level log-probs for *labels* given *logits*.

    ``logits``  shape: (B, T, V)
    ``labels``  shape: (B, T)
    returns     shape: (B, T)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)


# ===================================================================
#  Custom GRPO Trainer
# ===================================================================
class DuckHuntGRPOTrainer:
    """Manual GRPO loop with environment-driven reward."""

    def __init__(
        self,
        model,
        processor,
        env: DuckHuntEnvWrapper,
        config: FullConfig,
        accelerator=None,
    ) -> None:
        self.processor = processor
        self.env = env
        self.cfg = config
        self.accelerator = accelerator
        self._distributed = accelerator is not None

        grpo = config.grpo
        train = config.training

        # Reference model (frozen copy)  — skipped when beta == 0
        # Must be created BEFORE accelerator.prepare() wraps the model
        self.ref_model = None
        if grpo.beta > 0:
            self.ref_model = copy.deepcopy(model)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        # Optimiser
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=train.learning_rate,
            weight_decay=train.weight_decay,
        )

        total_steps = train.max_steps if train.max_steps > 0 else 1000
        total_optimizer_steps = total_steps // train.gradient_accumulation_steps
        warmup_optimizer_steps = int(total_optimizer_steps * train.warmup_ratio)
        scheduler = get_scheduler(
            train.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_optimizer_steps,
            num_training_steps=total_optimizer_steps,
        )
        logger.info(
            "Scheduler: %d optimizer steps, %d warmup steps (from %d loop steps, grad_accum=%d)",
            total_optimizer_steps, 
            warmup_optimizer_steps, 
            total_steps, 
            train.gradient_accumulation_steps,
        )

        # Wrap with Accelerate for distributed training
        if self._distributed:
            self.model, self.optimizer, self.scheduler = accelerator.prepare(
                model, optimizer, scheduler,
            )
            self.device = accelerator.device
            # Move ref model to correct device
            if self.ref_model is not None:
                self.ref_model = self.ref_model.to(self.device)
        else:
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.device = next(model.parameters()).device

        # Enable gradient checkpointing to save memory
        if train.gradient_checkpointing:
            unwrapped = self._unwrap_model()
            unwrapped.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Checkpoint tracking
        self.global_step = 0
        self.best_eval_hit_rate = -1.0
        self._reward_baseline = 0.0  # EMA reward baseline for moving_avg normalization
        self._saved_checkpoints: list[Path] = []
        self._last_grad_norm = 0.0
        self._total_hits = 0
        self._total_shots = 0

        # Hotspot detection — track recent shot positions to penalize repetition
        self._recent_shots: list[tuple[float, float]] = []
        self._hotspot_window: int = 50  # track last N shots
        self._hotspot_radius: float = 0.10  # normalized distance threshold (wider catch)

        # Sample outputs buffer for W&B table logging
        self._sample_outputs: list[dict] = []

        # Curriculum phase tracking
        self._curriculum_enabled = grpo.curriculum_phase2_step > 0
        self._phase2_step = grpo.curriculum_phase2_step
        self._current_phase = 1 if self._curriculum_enabled else 2
        if self._curriculum_enabled:
            logger.info(
                "Curriculum enabled: phase 1 (horizon=0) until step %d, then phase 2",
                self._phase2_step,
            )

        # Early training stabilization
        self._stabilization_steps = grpo.stabilization_steps
        self._stabilization_grad_accum = grpo.stabilization_grad_accum
        self._normal_grad_accum = train.gradient_accumulation_steps
        self._lora_freeze_steps = grpo.lora_freeze_steps
        self._lora_frozen = False

        if self._lora_freeze_steps > 0:
            self._freeze_lora()
            logger.info(
                "LoRA frozen for first %d steps (experience collection only)",
                self._lora_freeze_steps,
            )
        if self._stabilization_steps > 0:
            logger.info(
                "Stabilization: grad_accum=%d for first %d steps (then %d)",
                self._stabilization_grad_accum,
                self._stabilization_steps,
                self._normal_grad_accum,
            )

        # wandb — only on main process
        self._wandb_active = False
        if _WANDB and config.logging.report_to == "wandb" and self._is_main_process:
            wandb.init(
                project=config.logging.wandb_project,
                entity=config.logging.wandb_entity,
                name=config.logging.wandb_run_name,
                config=self._build_wandb_config(),
            )
            self._wandb_active = True

    # ------------------------------------------------------------------
    #  Distributed helpers
    # ------------------------------------------------------------------
    @property
    def _is_main_process(self) -> bool:
        if self.accelerator is not None:
            return self.accelerator.is_main_process
        return True

    def _unwrap_model(self):
        """Return the underlying model (unwrapped from DDP/Accelerate)."""
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.model)
        return self.model

    # ------------------------------------------------------------------
    #  Build full config dict for W&B (logged once)
    # ------------------------------------------------------------------
    def _build_wandb_config(self) -> dict:
        """Flatten all config sections into a single dict for W&B."""
        cfg = self.cfg
        return {
            # Model
            "model/name": cfg.model.model_name,
            "model/dtype": cfg.model.torch_dtype,
            "model/attn_implementation": cfg.model.attn_implementation,
            # LoRA
            "lora/enabled": cfg.lora.enabled,
            "lora/r": cfg.lora.r,
            "lora/alpha": cfg.lora.lora_alpha,
            "lora/dropout": cfg.lora.lora_dropout,
            "lora/target_modules": cfg.lora.target_modules,
            "lora/task_type": cfg.lora.task_type,
            # GRPO
            "grpo/num_generations": cfg.grpo.num_generations,
            "grpo/max_completion_length": cfg.grpo.max_completion_length,
            "grpo/temperature": cfg.grpo.temperature,
            "grpo/top_p": cfg.grpo.top_p,
            "grpo/top_k": cfg.grpo.top_k,
            "grpo/beta": cfg.grpo.beta,
            "grpo/epsilon": cfg.grpo.epsilon,
            "grpo/loss_type": cfg.grpo.loss_type,
            # Training
            "training/learning_rate": cfg.training.learning_rate,
            "training/max_steps": cfg.training.max_steps,
            "training/batch_size": cfg.training.per_device_train_batch_size,
            "training/gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "training/effective_batch_size": cfg.training.effective_batch_size,
            "training/warmup_ratio": cfg.training.warmup_ratio,
            "training/weight_decay": cfg.training.weight_decay,
            "training/lr_scheduler": cfg.training.lr_scheduler_type,
            "training/max_grad_norm": cfg.training.max_grad_norm,
            "training/gradient_checkpointing": cfg.training.gradient_checkpointing,
            "training/seed": cfg.training.seed,
            # Reward
            "reward/hit": cfg.reward.hit,
            "reward/double_kill": cfg.reward.double_kill,
            "reward/miss": cfg.reward.miss,
            "reward/lambda_horizon": cfg.reward.lambda_horizon,
            "reward/invalid_action": cfg.reward.invalid_action,
            # Environment
            "env/screen_width": cfg.environment.screen_width,
            "env/screen_height": cfg.environment.screen_height,
            "env/frame_output_size": cfg.environment.frame_output_size,
            "env/max_horizon": cfg.environment.max_horizon,
            "env/fps": cfg.environment.fps,
            "env/latency_options_ms": cfg.environment.latency_options_ms,
            "env/frames_per_observation": cfg.environment.frames_per_observation,
        }

    # ------------------------------------------------------------------
    #  LoRA freeze / unfreeze
    # ------------------------------------------------------------------
    def _freeze_lora(self) -> None:
        """Freeze all LoRA parameters (base model already frozen by PEFT)."""
        for name, param in self._unwrap_model().named_parameters():
            if "lora" in name.lower():
                param.requires_grad_(False)
        self._lora_frozen = True

    def _unfreeze_lora(self) -> None:
        """Unfreeze LoRA parameters."""
        for name, param in self._unwrap_model().named_parameters():
            if "lora" in name.lower():
                param.requires_grad_(True)
        self._lora_frozen = False

    def _get_format_weight(self, step: int) -> float:
        """Return format_weight with optional decay."""
        base = self.cfg.reward.format_weight
        decay_steps = self.cfg.reward.format_decay_steps
        if decay_steps <= 0 or base <= 0:
            return base
        return base * max(0.0, 1.0 - step / decay_steps)

    def _get_grad_accum(self, step: int) -> int:
        """Return gradient accumulation steps for the current step."""
        if self._stabilization_steps > 0 and step < self._stabilization_steps:
            return self._stabilization_grad_accum
        return self._normal_grad_accum

    # ------------------------------------------------------------------
    #  Main training loop
    # ------------------------------------------------------------------
    def train(self, resume_from: str | None = None) -> None:
        cfg = self.cfg
        train = cfg.training
        grpo = cfg.grpo

        total_steps = train.max_steps if train.max_steps > 0 else 1000
        start_step = 0

        # Resume from checkpoint if provided
        if resume_from is not None:
            start_step = self._load_checkpoint(resume_from)
            logger.info("Resumed from step %d", start_step)

        self.model.train()
        self.env.reset()

        for step in range(start_step, total_steps):
            # LoRA unfreeze transition
            if self._lora_frozen and step >= self._lora_freeze_steps:
                self._unfreeze_lora()
                logger.info(
                    "=== LoRA unfrozen at step %d (training begins) ===", step,
                )

            # Stabilization → normal grad accum transition
            if (
                self._stabilization_steps > 0
                and step == self._stabilization_steps
            ):
                logger.info(
                    "=== Stabilization complete at step %d: grad_accum %d → %d ===",
                    step, self._stabilization_grad_accum, self._normal_grad_accum,
                )

            # Curriculum phase transition
            if self._curriculum_enabled:
                if step >= self._phase2_step and self._current_phase == 1:
                    # Save Phase 1 checkpoint before transitioning
                    if self._is_main_process:
                        logger.info("Saving Phase 1 checkpoint at step %d ...", step)
                        self._save_phase_checkpoint(phase=1, step=step)
                    self._current_phase = 2
                    logger.info(
                        "=== CURRICULUM: switching to phase 2 (horizon unlocked) at step %d ===",
                        step,
                    )

            # 1. Collect a batch from the environment
            batch = self._collect_batch()

            # 2. Compute GRPO loss
            loss, metrics = self._compute_grpo_loss(batch)

            # 3. Backprop (with grad accumulation)
            grad_accum = self._get_grad_accum(step)

            if loss is None:
                # No signal in this group (all rewards equal). Skip update.
                if self._is_main_process and step % 50 == 0:
                    logger.info("Step %d: skipped (no signal, std=0)", step)
            else:
                scaled_loss = loss / grad_accum
                if self._distributed:
                    self.accelerator.backward(scaled_loss)
                else:
                    scaled_loss.backward()

            if (step + 1) % grad_accum == 0:
                if self._lora_frozen:
                    # LoRA frozen — discard gradients, no weight update
                    self.optimizer.zero_grad()
                else:
                    # Compute gradient norm before clipping
                    if self._distributed:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), train.max_grad_norm
                        )
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), train.max_grad_norm
                        )
                    self._last_grad_norm = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

            # Track hit rate (actual hits, not positive rewards)
            rewards = batch["rewards"]
            actual_hits = batch["hit_flags"]
            self._total_shots += len(rewards)
            self._total_hits += sum(1 for h in actual_hits if h)
            metrics["gradient_norm"] = self._last_grad_norm
            metrics["hit_rate"] = self._total_hits / max(self._total_shots, 1)
            metrics["total_hits"] = self._total_hits
            metrics["total_shots"] = self._total_shots

            # 4. Logging (main process only in distributed)
            if (step + 1) % train.logging_steps == 0 and self._is_main_process:
                self._log_metrics(step, metrics)

            # 5. Save checkpoint (main process only)
            if (step + 1) % train.save_steps == 0 and self._is_main_process:
                self._save_checkpoint(step, metrics)

            # 6. Evaluation
            if (
                train.eval_strategy == "steps"
                and train.eval_steps > 0
                and (step + 1) % train.eval_steps == 0
            ):
                eval_metrics = self._run_eval()
                if eval_metrics is not None:
                    self._log_eval_to_wandb(eval_metrics, step)
                    # Track best checkpoint by hit_rate
                    hit_rate = eval_metrics.get("core", {}).get("hit_rate", 0.0)
                    if hit_rate > self.best_eval_hit_rate:
                        self.best_eval_hit_rate = hit_rate
                        self._save_best_checkpoint(step, eval_metrics)

        # Final save — Phase 2 (or Phase 1 if curriculum disabled)
        if self._is_main_process:
            self._save_checkpoint(total_steps - 1, {})
            self._save_phase_checkpoint(phase=self._current_phase, step=total_steps - 1)
        logger.info("Training complete after %d steps.", total_steps)

    # ------------------------------------------------------------------
    #  Collect batch
    # ------------------------------------------------------------------
    def _collect_batch(self) -> dict:
        """Generate a batch of (prompt, completions, rewards, snapshots)."""
        grpo = self.cfg.grpo
        G = grpo.num_generations

        # Retry until we find a valid state:
        # 1. Duck flying at observation time
        # 2. Duck still flying AFTER latency (doesn't escape during processing)
        for _attempt in range(50):
            self.env.auto_advance_to_next_match()

            # Randomize latency every attempt
            self.env._env.latency_ms = random.choice(self.env.config.latency_options_ms)
            self.env._env.latency_frames = int(
                self.env._env.latency_ms / 1000 * self.env.config.fps
            )

            self.env.advance_frames(random.randint(8, 25))

            frames = self.env.get_frames()
            state = self.env.get_state()
            flying = state.get("ducks_flying", 0)

            if flying < 1 or len(frames) == 0:
                if self.env.is_done():
                    self.env.reset()
                continue

            # Verify ducks will STILL be flying after latency (no shoot_dead outcomes)
            from .dataset import _snapshot_duck, _restore_duck
            inner = self.env._env
            match = inner.round.current_match
            latency_frames = inner.latency_frames
            round_number = inner.round_number

            # Snapshot current ducks
            duck_a_snap = _snapshot_duck(match.duck_a)
            duck_b_snap = _snapshot_duck(match.duck_b) if match.duck_b is not None else None

            # Save RNG, simulate forward, check if ducks still flying
            rng_state = random.getstate()
            from game_engine import DuckState
            duck_a_sim = _restore_duck(duck_a_snap, round_number)
            duck_b_sim = _restore_duck(duck_b_snap, round_number) if duck_b_snap else None
            for _ in range(latency_frames):
                duck_a_sim.update(round_number)
                if duck_b_sim is not None:
                    duck_b_sim.update(round_number)
            random.setstate(rng_state)  # restore RNG so training isn't affected

            still_flying_after_latency = (
                duck_a_sim.state == DuckState.FLYING
                or (duck_b_sim is not None and duck_b_sim.state == DuckState.FLYING)
            )

            if still_flying_after_latency:
                break  # valid state found
            # Otherwise retry with new match
        else:
            logger.warning("Could not find valid state after 50 attempts, using current state")

        # Build prompt (phase 1 = no horizon in tool schema for LiquidAI)
        messages, tools = build_prompt(frames, state, phase=self._current_phase)

        # Snapshot for deterministic reward
        snap = capture_snapshot(self.env)

        # Tokenise prompt
        inputs = self.processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        # Choose completion length based on curriculum phase
        if self._current_phase == 1:
            max_tokens = grpo.phase1_max_completion_length
        else:
            max_tokens = grpo.max_completion_length

        # Generate G completions (use unwrapped model — DDP wrapper breaks generate)
        gen_model = self._unwrap_model()
        gen_model.eval()
        with torch.no_grad():
            output_ids = gen_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=grpo.temperature,
                top_p=grpo.top_p,
                top_k=grpo.top_k if grpo.top_k > 0 else None,
                num_return_sequences=G,
            )
        self.model.train()

        # Decode and compute rewards
        completion_ids_list = []
        rewards = []
        completions_text = []
        parsed_actions = []
        hit_flags = []  # True if hit at least one duck
        reward_breakdowns = []  # detailed reward components for W&B

        for i in range(G):
            comp_ids = output_ids[i, prompt_len:]
            completion_ids_list.append(comp_ids)

            decoded = self.processor.decode(comp_ids, skip_special_tokens=False)
            completions_text.append(decoded)
            action = parse_tool_call(
                decoded,
                max_horizon=self.cfg.environment.max_horizon,
                phase=self._current_phase,
            )

            parsed_actions.append(action)

            if action is not None:
                result = simulate_shot(snap, action)
            else:
                result = {"hit_a": False, "hit_b": False, "had_target": True}

            is_hit = result.get("hit_a", False) or result.get("hit_b", False)
            hit_flags.append(is_hit)

            bd = compute_reward_detailed(result, action, self.cfg.reward)
            reward = bd.total
            reward_breakdowns.append(bd)

            # Hotspot penalty — only penalize repetitive coordinates when NOT aiming at a duck.
            # If the shot is close to a duck (min_distance < 0.3), it's legitimate aiming.
            # If the shot is far from any duck AND in a hotspot, it's exploitation.
            if self.cfg.reward.hotspot_enabled and action is not None and reward > 0 and len(self._recent_shots) >= 10:
                # Only apply hotspot if shot is NOT near a duck.
                # Hits are always near a duck — skip hotspot for hits.
                is_near_duck = is_hit or (bd.min_distance >= 0 and bd.min_distance < 0.3)
                if not is_near_duck:
                    shot_pos = (action.x, action.y)
                    nearby = sum(
                        1 for sx, sy in self._recent_shots
                        if abs(sx - shot_pos[0]) < self._hotspot_radius
                        and abs(sy - shot_pos[1]) < self._hotspot_radius
                    )
                    concentration = nearby / len(self._recent_shots)
                    if concentration > 0.2:
                        scale = 1.0 - (concentration - 0.2) * 5.0
                        reward = reward * max(scale, -1.5)
                        logger.info(
                            "  [gen %d] hotspot: %d/%d (%.0f%%) near (%.2f, %.2f), scale=%.2f, reward=%.2f",
                            i, nearby, len(self._recent_shots), concentration * 100,
                            shot_pos[0], shot_pos[1], scale, reward,
                        )

            rewards.append(reward)

            # Track shot positions for hotspot detection
            if action is not None:
                self._recent_shots.append((action.x, action.y))
                if len(self._recent_shots) > self._hotspot_window:
                    self._recent_shots.pop(0)

            # Log full completion with reward breakdown to console
            action_str = f"x={action.x:.3f}, y={action.y:.3f}, h={action.horizon}" if action else "INVALID"
            hit_str = ""
            if action:
                hit_str = f" hit_a={result['hit_a']} hit_b={result['hit_b']}"
            dist_str = f" dist={bd.min_distance:.3f}" if bd.min_distance >= 0 else ""
            prox_str = f" prox={bd.proximity_bonus:.3f}" if bd.proximity_bonus > 0 else ""
            hpen_str = f" hpen={bd.horizon_penalty:.3f}" if bd.horizon_penalty > 0 else ""
            logger.info(
                "  [gen %d] reward=%.2f (base=%.2f%s%s%s) %s action=%s%s\n    output: %s",
                i, reward, bd.base, hpen_str, prox_str, dist_str,
                bd.outcome, action_str, hit_str, decoded,
            )

        # Render a "shot-time" frame showing where ducks are when shot lands
        # Uses the first valid action's latency+horizon to advance the snapshot
        shot_frame = self._render_shot_frame(snap, parsed_actions)

        # Log frames and completions to W&B
        self._log_batch_to_wandb(
            frames, completions_text, rewards, parsed_actions, hit_flags,
            shot_frame=shot_frame, reward_breakdowns=reward_breakdowns,
        )

        # Advance the real environment (larger range for diverse states)
        self.env.advance_frames(random.randint(15, 60))

        # Optionally filter out invalid (unparseable) generations from GRPO
        skip_mask = [False] * G
        if self.cfg.reward.skip_invalid_generations:
            for idx in range(G):
                if parsed_actions[idx] is None:
                    skip_mask[idx] = True

        if any(skip_mask) and not all(skip_mask):
            filtered_ids = []
            filtered_rewards = []
            filtered_full = []
            filtered_hits = []
            for idx in range(G):
                if not skip_mask[idx]:
                    filtered_ids.append(completion_ids_list[idx])
                    filtered_rewards.append(rewards[idx])
                    filtered_full.append(output_ids[idx])
                    filtered_hits.append(hit_flags[idx])
            skipped = sum(skip_mask)
            completion_ids_list = filtered_ids
            rewards = filtered_rewards
            hit_flags = filtered_hits
            output_ids = torch.stack(filtered_full)
            logger.info(
                "  Skipped %d generation(s), %d remaining",
                skipped, len(rewards),
            )

        return {
            "prompt_ids": inputs["input_ids"][0],  # (T_prompt,)
            "completion_ids": completion_ids_list,  # list of G tensors
            "rewards": rewards,  # list of G floats
            "hit_flags": hit_flags,  # list of G bools (actual hits)
            "full_ids": output_ids,  # (G, T_prompt + T_comp)
            "prompt_len": prompt_len,
            "attention_mask": inputs.get("attention_mask"),
        }

    # ------------------------------------------------------------------
    #  GRPO loss computation
    # ------------------------------------------------------------------
    def _compute_grpo_loss(self, batch: dict) -> tuple[torch.Tensor | None, dict]:
        grpo = self.cfg.grpo
        G = len(batch["rewards"])

        rewards_t = torch.tensor(batch["rewards"], dtype=torch.float32)
        prompt_len = batch["prompt_len"]
        norm_mode = self.cfg.reward.reward_normalization

        # --- 1. Advantages ---
        mean_r = rewards_t.mean()
        std_r = rewards_t.std()

        # Bounded normalization with std floor and clamp to prevent blow-up
        STD_FLOOR = 0.3
        ADV_CLAMP = 3.0

        # No reward signal in this group: zero advantages so the policy loss
        # contributes nothing, but DO NOT early-return. We still run the forward
        # pass so the entropy bonus (+ KL anchor) provides exploration pressure.
        zero_signal = bool(std_r < 1e-6)
        if zero_signal:
            advantages = torch.zeros(G, dtype=torch.float32)

        elif norm_mode == "moving_avg":
            alpha = self.cfg.reward.moving_avg_alpha
            self._reward_baseline = (1 - alpha) * self._reward_baseline + alpha * mean_r.item()
            centered = rewards_t - self._reward_baseline
            # Use the larger of group std and centered std, with a floor
            std_used = max(centered.std().item(), std_r.item(), STD_FLOOR)
            advantages = (centered / std_used).clamp(-ADV_CLAMP, ADV_CLAMP)

        elif norm_mode == "per_component":
            accuracy_rewards = torch.tensor(
                [bd.total for bd in batch.get("reward_breakdowns", [])],
                dtype=torch.float32,
            ) if "reward_breakdowns" in batch else rewards_t

            format_rewards = torch.tensor(
                batch.get("format_rewards", [0.0] * G),
                dtype=torch.float32,
            )

            def _norm(t: torch.Tensor) -> torch.Tensor:
                s = max(t.std().item(), STD_FLOOR)
                return ((t - t.mean()) / s).clamp(-ADV_CLAMP, ADV_CLAMP)

            acc_weight = self.cfg.reward.accuracy_weight
            fmt_weight = self._get_format_weight(batch.get("step", 0))
            total_weight = acc_weight + fmt_weight + 1e-8

            advantages = (
                acc_weight * _norm(accuracy_rewards)
                + fmt_weight * _norm(format_rewards)
            ) / total_weight

        else:
            # Default: group normalization with std floor + clamp
            std_used = max(std_r.item(), STD_FLOOR)
            advantages = ((rewards_t - mean_r) / std_used).clamp(-ADV_CLAMP, ADV_CLAMP)

        advantages = advantages.to(self.device)

        # Token-level accumulators (sum over ALL tokens in the group, then
        # divide by the total token count). This removes the length bias of
        # per-sequence .mean(), which over-weighted shorter completions and
        # pushed the policy toward shorter, more deterministic outputs.
        policy_num = torch.tensor(0.0, device=self.device)
        kl_num = torch.tensor(0.0, device=self.device)
        entropy_num = torch.tensor(0.0, device=self.device)
        total_tokens = 0

        for i in range(G):
            full_ids = batch["full_ids"][i].unsqueeze(0)  # (1, T)
            comp_ids = batch["completion_ids"][i]
            comp_len = comp_ids.shape[0]

            if comp_len == 0:
                continue

            # Labels: -100 for prompt tokens, real ids for completion
            labels = full_ids.clone()
            labels[0, :prompt_len] = -100

            # --- 2. Policy log-probs ---
            outputs = self.model(input_ids=full_ids, labels=labels)
            logits = outputs.logits[:, prompt_len - 1 : -1, :]  # align with comp
            comp_labels = full_ids[:, prompt_len:]

            log_probs = _log_probs_from_logits(logits, comp_labels)  # (1, comp_len)

            # --- 2b. Token-level entropy (prevents mode collapse) ---
            probs = F.softmax(logits, dim=-1)
            log_probs_full = F.log_softmax(logits, dim=-1)
            entropy_tok = -(probs * log_probs_full).sum(dim=-1)  # (1, comp_len)

            # --- 3. Reference log-probs ---
            if self.ref_model is not None:
                with torch.no_grad():
                    ref_out = self.ref_model(input_ids=full_ids, labels=labels)
                    ref_logits = ref_out.logits[:, prompt_len - 1 : -1, :]
                    ref_log_probs = _log_probs_from_logits(ref_logits, comp_labels)
            else:
                ref_log_probs = log_probs.detach()

            # --- 4. Accumulate token-level terms (no per-sequence mean) ---
            adv = advantages[i]
            policy_num = policy_num + -(log_probs * adv).sum()

            # KL (Schulman k3, per-token; always ≥ 0)
            diff = ref_log_probs - log_probs
            kl_tok = (diff.exp() - 1 - diff)
            kl_num = kl_num + kl_tok.sum()

            entropy_num = entropy_num + entropy_tok.sum()
            total_tokens += comp_len

        denom = max(total_tokens, 1)
        policy_loss = policy_num / denom
        mean_kl_t = kl_num / denom
        mean_entropy_t = entropy_num / denom

        total_loss = policy_loss + grpo.beta * mean_kl_t - grpo.entropy_coeff * mean_entropy_t

        # --- 7. Entropy floor penalty (emergency brake) ---
        mean_entropy = mean_entropy_t.item()
        entropy_floor_penalty = 0.0
        if grpo.entropy_floor > 0 and mean_entropy < grpo.entropy_floor:
            entropy_floor_penalty = grpo.entropy_floor_coeff * (grpo.entropy_floor - mean_entropy)
            total_loss = total_loss + entropy_floor_penalty
            logger.warning(
                "Entropy below floor: %.2f < %.2f, adding penalty %.4f",
                mean_entropy, grpo.entropy_floor, entropy_floor_penalty,
            )

        metrics = {
            "loss": total_loss.item(),
            "mean_reward": mean_r.item(),
            "std_reward": std_r.item(),
            "mean_kl": mean_kl_t.item(),
            "mean_entropy": mean_entropy,
            "entropy_floor_penalty": entropy_floor_penalty,
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item() if G > 1 else 0.0,
            "advantages_max": advantages.max().item(),
            "advantages_min": advantages.min().item(),
            "zero_signal": zero_signal,
        }

        return total_loss, metrics

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------
    def _run_eval(self) -> dict | None:
        """Run evaluation and return metrics dict."""
        try:
            # Lazy import to avoid circular dependency
            import importlib
            import sys as _sys

            # evaluate.py is at training/evaluate.py — parent of src/
            eval_dir = str(Path(__file__).resolve().parent.parent)
            if eval_dir not in _sys.path:
                _sys.path.insert(0, eval_dir)

            from evaluate import evaluate as eval_fn

            logger.info("Running evaluation at step %d …", self.global_step)
            metrics = eval_fn(
                model=self.model,
                processor=self.processor,
                env=self.env,
                config=self.cfg,
                num_episodes=2,  # keep eval fast during training
                max_steps_per_episode=50,
            )
            logger.info(
                "Eval hit_rate=%.1f%%  avg_reward=%.3f",
                metrics.get("core", {}).get("hit_rate", 0) * 100,
                metrics.get("core", {}).get("average_reward", 0),
            )
            return metrics
        except Exception as e:
            logger.warning("Evaluation failed: %s", e)
            return None

    # ------------------------------------------------------------------
    #  Shot-time frame rendering
    # ------------------------------------------------------------------
    def _render_shot_frame(
        self, snap: dict, actions: list,
    ) -> Image.Image | None:
        """Render the game frame at shot time (after latency + horizon).

        This shows where the ducks ACTUALLY ARE when the shot lands,
        not where they were when the model saw the input frames.
        """
        # Find the first valid action to determine advancement
        action = next((a for a in actions if a is not None), None)
        if action is None:
            return None

        try:
            from .dataset import _restore_duck, _rng_restore
            from config import SCREEN_WIDTH, SCREEN_HEIGHT
            import json

            round_number = snap["round_number"]
            latency_frames = snap["latency_frames"]

            # Restore RNG and ducks from snapshot
            _rng_restore(snap["rng_state"])
            duck_a = _restore_duck(snap["duck_a"], round_number)
            has_duck_b = "duck_b" in snap
            duck_b = _restore_duck(snap["duck_b"], round_number) if has_duck_b else None

            # Advance by latency + horizon (same as simulate_shot)
            total_advance = latency_frames + action.horizon
            for _ in range(total_advance):
                duck_a.update(round_number)
                if duck_b is not None:
                    duck_b.update(round_number)

            # Build a game_state dict for the renderer
            game_state = {
                "duck_a": {
                    "x": duck_a.x, "y": duck_a.y,
                    "state": duck_a.state.value,
                    "sprite_dir": duck_a.sprite_dir,
                },
            }
            if duck_b is not None:
                game_state["duck_b"] = {
                    "x": duck_b.x, "y": duck_b.y,
                    "state": duck_b.state.value,
                    "sprite_dir": duck_b.sprite_dir,
                }

            # Render using the environment's renderer
            renderer = self.env._env.renderer
            frame = renderer.render_and_resize(game_state, total_advance)

            # Draw hitbox rectangles on top of ducks
            frame = self._draw_hitbox(
                frame, duck_a.x, duck_a.y, duck_a.state.value,
                color=(255, 255, 0), label="A",
            )
            if duck_b is not None:
                frame = self._draw_hitbox(
                    frame, duck_b.x, duck_b.y, duck_b.state.value,
                    color=(0, 255, 255), label="B",
                )
            return frame

        except Exception as e:
            logger.warning("Failed to render shot frame: %s", e)
            return None

    # ------------------------------------------------------------------
    #  Logging
    # ------------------------------------------------------------------
    # Crosshair sprite (loaded once, shared across instances)
    _crosshair_sprite: Image.Image | None = None

    @classmethod
    def _get_crosshair_sprite(cls) -> Image.Image:
        """Load the crosshair sprite from assets (cached)."""
        if cls._crosshair_sprite is None:
            from pathlib import Path
            assets = Path(__file__).resolve().parent.parent.parent / "duck_hunt_openenv" / "assets"
            sprite_path = assets / "crosshairs.png"
            cls._crosshair_sprite = Image.open(sprite_path).convert("RGBA")
        return cls._crosshair_sprite

    @staticmethod
    def _draw_hitbox(
        frame: Image.Image,
        duck_x: float,
        duck_y: float,
        duck_state: str,
        color: tuple = (255, 255, 0),
        label: str | None = None,
    ) -> Image.Image:
        """Draw a hitbox rectangle on a copy of the frame for a duck."""
        from PIL import ImageDraw
        try:
            from config import HITBOX_WIDTH, HITBOX_HEIGHT, SPRITE_WIDTH, SPRITE_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT
        except ImportError:
            return frame

        if duck_state != "flying":
            return frame

        img = frame.copy().convert("RGBA")
        w, h = img.size

        # Duck position is top-left of sprite in game coords
        # Hitbox is centered on sprite
        hx = duck_x + (SPRITE_WIDTH - HITBOX_WIDTH) / 2
        hy = duck_y + (SPRITE_HEIGHT - HITBOX_HEIGHT) / 2

        # Convert to frame pixel coords (frame might be 512x512, game is 800x500)
        scale_x = w / SCREEN_WIDTH
        scale_y = h / SCREEN_HEIGHT
        x1 = int(hx * scale_x)
        y1 = int(hy * scale_y)
        x2 = int((hx + HITBOX_WIDTH) * scale_x)
        y2 = int((hy + HITBOX_HEIGHT) * scale_y)

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([x1, y1, x2, y2], outline=color + (200,), width=2)
        if label:
            draw.text((x1, y1 - 12), label, fill=color + (255,))

        img = Image.alpha_composite(img, overlay)
        return img.convert("RGB")

    @staticmethod
    def _draw_crosshair(
        frame: Image.Image,
        x_norm: float,
        y_norm: float,
        color: tuple = (255, 0, 0),
        label: str | None = None,
    ) -> Image.Image:
        """Draw a crosshair sprite on a copy of the frame at normalised (x, y)."""
        from PIL import ImageDraw, ImageChops

        img = frame.copy().convert("RGBA")
        w, h = img.size
        cx = int(x_norm * w)
        cy = int(y_norm * h)

        # Get crosshair sprite and tint it
        raw = DuckHuntGRPOTrainer._get_crosshair_sprite()
        # Scale crosshair to ~8% of frame width
        ch_size = max(24, w // 12)
        sprite = raw.resize((ch_size, ch_size), Image.LANCZOS)

        # Tint the sprite: replace non-transparent pixels with the color
        r_ch, g_ch, b_ch, a_ch = sprite.split()
        tinted = Image.merge("RGBA", (
            r_ch.point(lambda _: color[0]),
            g_ch.point(lambda _: color[1]),
            b_ch.point(lambda _: color[2]),
            a_ch,
        ))

        # Paste centered on (cx, cy)
        paste_x = cx - ch_size // 2
        paste_y = cy - ch_size // 2
        img.paste(tinted, (paste_x, paste_y), tinted)

        # Label text
        if label:
            draw = ImageDraw.Draw(img)
            draw.text(
                (cx + ch_size // 2 + 4, cy - 8),
                label, fill=color + (255,),
            )

        return img.convert("RGB")

    def _log_batch_to_wandb(
        self,
        frames: list,
        completions: list[str],
        rewards: list[float],
        actions: list | None = None,
        hit_flags: list[bool] | None = None,
        shot_frame: Image.Image | None = None,
        reward_breakdowns: list | None = None,
    ) -> None:
        """Log observation frames and shot-time frames with crosshairs to W&B.

        ``frames`` are what the model SAW (input).
        ``shot_frame`` is where the ducks ARE when the shot lands (after latency+horizon).
        Crosshairs are drawn on the shot_frame so they align with duck positions.
        """
        if not self._wandb_active or wandb.run is None:
            return

        try:
            self._log_batch_to_wandb_inner(
                frames, completions, rewards, actions, hit_flags,
                shot_frame, reward_breakdowns,
            )
        except OSError as e:
            logger.warning("W&B batch logging failed (I/O error), skipping: %s", e)

    def _log_batch_to_wandb_inner(
        self,
        frames: list,
        completions: list[str],
        rewards: list[float],
        actions: list | None = None,
        hit_flags: list[bool] | None = None,
        shot_frame: Image.Image | None = None,
        reward_breakdowns: list | None = None,
    ) -> None:
        log_dict: dict = {}

        # Log raw observation frames (what the model saw)
        wandb_images = []
        for i, frame in enumerate(frames):
            wandb_images.append(wandb.Image(frame, caption=f"input_frame_{i}"))
        if wandb_images:
            log_dict["train/observation_frames"] = wandb_images

        if actions and hit_flags:
            # Use shot_frame (duck positions at shot time) for crosshairs.
            # Fall back to last input frame if shot_frame rendering failed.
            base_frame = shot_frame if shot_frame is not None else (frames[-1] if frames else None)
            if base_frame is None:
                # No frame to draw on
                pass
            else:
                hit_color = (0, 255, 0)    # green for hits
                miss_color = (255, 0, 0)   # red for misses

                hit_images = []
                miss_images = []

                for i, (action, is_hit) in enumerate(zip(actions, hit_flags)):
                    if action is None:
                        continue

                    color = hit_color if is_hit else miss_color
                    tag = "HIT" if is_hit else "MISS"
                    label = f"g{i} h={action.horizon} {tag} r={rewards[i]:.2f}"
                    annotated = self._draw_crosshair(
                        base_frame, action.x, action.y,
                        color=color, label=label,
                    )
                    caption = (
                        f"gen{i}: ({action.x:.2f},{action.y:.2f}) "
                        f"h={action.horizon} {tag} r={rewards[i]:.2f}"
                    )

                    if is_hit:
                        hit_images.append(wandb.Image(annotated, caption=caption))
                    else:
                        miss_images.append(wandb.Image(annotated, caption=caption))

                if hit_images:
                    log_dict["train/hits"] = hit_images
                if miss_images:
                    log_dict["train/misses"] = miss_images

                # Combined overlay with all shots on shot-time frame
                if any(a is not None for a in actions):
                    combined = base_frame.copy().convert("RGB")
                    for i, (action, is_hit) in enumerate(zip(actions, hit_flags)):
                        if action is not None:
                            color = hit_color if is_hit else miss_color
                            label = f"g{i} h={action.horizon}"
                            combined = self._draw_crosshair(
                                combined, action.x, action.y,
                                color=color, label=label,
                            )
                    frame_type = "shot-time" if shot_frame is not None else "input"
                    log_dict["train/all_shots_overlay"] = wandb.Image(
                        combined,
                        caption=f"All shots on {frame_type} frame: green=hit, red=miss",
                    )

        # Log completions table with reward breakdown
        columns = [
            "gen_idx", "completion", "reward", "hit", "outcome",
            "base", "horizon_penalty", "proximity_bonus", "min_distance",
        ]
        comp_table = wandb.Table(columns=columns)
        for i, (text, reward) in enumerate(zip(completions, rewards)):
            is_hit = hit_flags[i] if hit_flags else False
            if reward_breakdowns and i < len(reward_breakdowns):
                bd = reward_breakdowns[i]
                comp_table.add_data(
                    i, text, reward, is_hit, bd.outcome,
                    bd.base, bd.horizon_penalty, bd.proximity_bonus, bd.min_distance,
                )
            else:
                comp_table.add_data(i, text, reward, is_hit, "", 0, 0, 0, -1)
        log_dict["train/completions"] = comp_table

        # Log aggregate reward component metrics
        if reward_breakdowns:
            bds = reward_breakdowns
            valid_bds = [b for b in bds if b.outcome != "invalid"]
            miss_bds = [b for b in bds if b.outcome == "miss"]

            log_dict["reward/mean_base"] = sum(b.base for b in bds) / max(len(bds), 1)
            log_dict["reward/mean_proximity"] = (
                sum(b.proximity_bonus for b in miss_bds) / max(len(miss_bds), 1)
                if miss_bds else 0.0
            )
            log_dict["reward/mean_distance"] = (
                sum(b.min_distance for b in miss_bds if b.min_distance >= 0)
                / max(sum(1 for b in miss_bds if b.min_distance >= 0), 1)
                if miss_bds else 0.0
            )
            log_dict["reward/mean_horizon_penalty"] = (
                sum(b.horizon_penalty for b in bds) / max(len(bds), 1)
            )

            # Outcome counts
            outcomes = [b.outcome for b in bds]
            log_dict["reward/n_hits"] = outcomes.count("hit") + outcomes.count("double_kill")
            log_dict["reward/n_misses"] = outcomes.count("miss")
            log_dict["reward/n_invalid"] = outcomes.count("invalid")
            log_dict["reward/n_shoot_dead"] = outcomes.count("shoot_dead")

        try:
            wandb.log(log_dict)
        except OSError as e:
            logger.warning("W&B logging failed (I/O error), skipping: %s", e)

    def _log_metrics(self, step: int, metrics: dict) -> None:
        lr = self.scheduler.get_last_lr()[0]
        grad_norm = metrics.get("gradient_norm", 0.0)
        hit_rate = metrics.get("hit_rate", 0.0)
        entropy = metrics.get("mean_entropy", 0.0)
        msg = (
            f"step={step + 1}  "
            f"loss={metrics.get('loss', 0):.4f}  "
            f"reward={metrics.get('mean_reward', 0):.3f} +/- {metrics.get('std_reward', 0):.3f}  "
            f"hits={metrics.get('total_hits', 0)}/{metrics.get('total_shots', 0)} ({hit_rate:.1%})  "
            f"entropy={entropy:.2f}  "
            f"grad={grad_norm:.4f}  "
            f"lr={lr:.2e}"
        )
        logger.info(msg)

        if self._wandb_active and wandb.run is not None:
            log_dict = {
                "train/loss": metrics.get("loss", 0),
                "train/mean_reward": metrics.get("mean_reward", 0),
                "train/std_reward": metrics.get("std_reward", 0),
                "train/learning_rate": lr,
                "train/advantages_mean": metrics.get("advantages_mean", 0),
                "train/advantages_std": metrics.get("advantages_std", 0),
                "train/advantages_max": metrics.get("advantages_max", 0),
                "train/advantages_min": metrics.get("advantages_min", 0),
                "train/gradient_norm": grad_norm,
                "train/hit_rate": hit_rate,
                "train/mean_entropy": entropy,
                "train/total_hits": metrics.get("total_hits", 0),
                "train/step": step + 1,
                "train/curriculum_phase": self._current_phase,
                "train/lora_frozen": int(self._lora_frozen),
                "train/grad_accum": self._get_grad_accum(step),
            }
            try:
                wandb.log(log_dict)
            except OSError as e:
                logger.warning("W&B metrics log failed, skipping: %s", e)

    def _log_eval_to_wandb(self, eval_metrics: dict, step: int) -> None:
        """Log rich evaluation data to W&B: metrics, histogram, table, chart."""
        if not self._wandb_active or wandb.run is None:
            return

        log_dict: dict = {}

        # ---- All evaluation metrics ----
        core = eval_metrics.get("core", {})
        for k, v in core.items():
            if isinstance(v, (int, float)):
                log_dict[f"eval/{k}"] = v

        horizon = eval_metrics.get("horizon", {})
        for k, v in horizon.items():
            if isinstance(v, (int, float)):
                log_dict[f"eval/horizon_{k}"] = v

        hw = eval_metrics.get("hardware_aware", {})
        for k, v in hw.items():
            if isinstance(v, (int, float)):
                log_dict[f"eval/{k}"] = v

        # ---- Horizon distribution (histogram) ----
        by_latency = eval_metrics.get("by_latency", {})
        all_horizons = []
        for lat_data in by_latency.values():
            avg_h = lat_data.get("average_horizon", 0)
            if avg_h > 0:
                all_horizons.append(avg_h)
        if all_horizons:
            log_dict["eval/horizon_distribution"] = wandb.Histogram(all_horizons)

        # ---- Hit rate by latency (line chart data) ----
        if by_latency:
            lat_table = wandb.Table(
                columns=["latency_ms", "hit_rate", "avg_horizon", "avg_reward"],
            )
            for lat_ms in sorted(by_latency.keys()):
                m = by_latency[lat_ms]
                lat_table.add_data(
                    lat_ms,
                    m.get("hit_rate", 0),
                    m.get("average_horizon", 0),
                    m.get("average_reward", 0),
                )
            log_dict["eval/by_latency"] = lat_table

        # ---- Sample outputs (table) ----
        if self._sample_outputs:
            recent = self._sample_outputs[-self.cfg.logging.num_completions_to_print:]
            sample_table = wandb.Table(
                columns=["step", "gen_idx", "output", "action", "reward", "hit"],
            )
            for s in recent:
                sample_table.add_data(
                    s["step"], s["generation_idx"],
                    s["output"][:200], s["action"],
                    s["reward"], s["hit"],
                )
            log_dict["eval/sample_outputs"] = sample_table

        log_dict["eval/best_hit_rate"] = self.best_eval_hit_rate

        try:
            wandb.log(log_dict)
        except OSError as e:
            logger.warning("W&B eval log failed, skipping: %s", e)

    # ------------------------------------------------------------------
    #  Checkpointing (9.1)
    # ------------------------------------------------------------------
    def _save_checkpoint(self, step: int, metrics: dict) -> None:
        """Save a full checkpoint: model, optimizer, scheduler, trainer state."""
        out_dir = Path(self.cfg.training.output_dir) / f"checkpoint-{step + 1}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. LoRA weights (adapter_model.safetensors + adapter_config.json)
        self._unwrap_model().save_pretrained(str(out_dir))
        self.processor.save_pretrained(str(out_dir))

        # 2. Optimizer state
        torch.save(self.optimizer.state_dict(), str(out_dir / "optimizer.pt"))

        # 3. Scheduler state
        torch.save(self.scheduler.state_dict(), str(out_dir / "scheduler.pt"))

        # 4. Trainer state (step, config, metrics)
        trainer_state = {
            "global_step": self.global_step,
            "step": step + 1,
            "best_eval_hit_rate": self.best_eval_hit_rate,
            "last_metrics": {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float))
            },
            "config": {
                "model_name": self.cfg.model.model_name,
                "learning_rate": self.cfg.training.learning_rate,
                "lora_r": self.cfg.lora.r,
                "grpo_beta": self.cfg.grpo.beta,
                "num_generations": self.cfg.grpo.num_generations,
                "max_steps": self.cfg.training.max_steps,
                "output_dir": self.cfg.training.output_dir,
            },
        }
        with open(out_dir / "trainer_state.json", "w") as f:
            json.dump(trainer_state, f, indent=2)

        logger.info("Saved checkpoint to %s", out_dir)

        # 5. Manage checkpoint rotation (save_total_limit)
        self._saved_checkpoints.append(out_dir)
        limit = self.cfg.training.save_total_limit
        if limit > 0 and len(self._saved_checkpoints) > limit:
            oldest = self._saved_checkpoints.pop(0)
            # Don't delete best_checkpoint
            if oldest.name != "best_checkpoint" and oldest.exists():
                shutil.rmtree(oldest)
                logger.info("Removed old checkpoint %s (save_total_limit=%d)", oldest, limit)

    def _save_best_checkpoint(self, step: int, eval_metrics: dict) -> None:
        """Save the best checkpoint (by eval hit_rate)."""
        best_dir = Path(self.cfg.training.output_dir) / "best_checkpoint"
        best_dir.mkdir(parents=True, exist_ok=True)

        # Model + processor
        self._unwrap_model().save_pretrained(str(best_dir))
        self.processor.save_pretrained(str(best_dir))

        # Optimizer + scheduler
        torch.save(self.optimizer.state_dict(), str(best_dir / "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), str(best_dir / "scheduler.pt"))

        # Trainer state with eval metrics
        trainer_state = {
            "global_step": self.global_step,
            "step": step + 1,
            "best_eval_hit_rate": self.best_eval_hit_rate,
            "eval_metrics": eval_metrics,
            "config": {
                "model_name": self.cfg.model.model_name,
                "learning_rate": self.cfg.training.learning_rate,
                "lora_r": self.cfg.lora.r,
                "grpo_beta": self.cfg.grpo.beta,
                "num_generations": self.cfg.grpo.num_generations,
            },
        }
        with open(best_dir / "trainer_state.json", "w") as f:
            json.dump(trainer_state, f, indent=2, default=str)

        logger.info(
            "New best checkpoint saved (hit_rate=%.1f%%) at step %d",
            self.best_eval_hit_rate * 100, step + 1,
        )

    def _save_phase_checkpoint(self, phase: int, step: int) -> None:
        """Save a phase checkpoint and push to Hub if configured.

        Phase checkpoints are NOT subject to checkpoint rotation (save_total_limit)
        so they persist for the entire run.
        """
        phase_dir = Path(self.cfg.training.output_dir) / f"phase{phase}_checkpoint"
        phase_dir.mkdir(parents=True, exist_ok=True)

        self._unwrap_model().save_pretrained(str(phase_dir))
        self.processor.save_pretrained(str(phase_dir))

        trainer_state = {
            "global_step": self.global_step,
            "step": step + 1,
            "phase": phase,
            "hit_rate": self._total_hits / max(self._total_shots, 1),
            "total_hits": self._total_hits,
            "total_shots": self._total_shots,
        }
        with open(phase_dir / "trainer_state.json", "w") as f:
            json.dump(trainer_state, f, indent=2)

        logger.info("Phase %d checkpoint saved to %s", phase, phase_dir)

        # Push to Hub (same repo, same branch — latest phase overwrites)
        if self.cfg.hub.push_to_hub and self.cfg.hub.hub_model_id:
            logger.info("Pushing phase %d checkpoint to Hub: %s", phase, self.cfg.hub.hub_model_id)
            self.push_to_hub(checkpoint_dir=str(phase_dir))

    def _load_checkpoint(self, checkpoint_dir: str) -> int:
        """Resume training from a saved checkpoint.

        Returns the step number to resume from.
        """
        ckpt = Path(checkpoint_dir)

        # 1. Load model weights (LoRA adapter)
        from peft import PeftModel

        unwrapped = self._unwrap_model()
        if hasattr(unwrapped, "load_adapter"):
            unwrapped.load_adapter(str(ckpt), adapter_name="default")
        else:
            logger.warning("Model does not support load_adapter; skipping weight load.")

        # 2. Optimizer state
        opt_path = ckpt / "optimizer.pt"
        if opt_path.exists():
            self.optimizer.load_state_dict(torch.load(str(opt_path), map_location=self.device))
            logger.info("Loaded optimizer state from %s", opt_path)

        # 3. Scheduler state
        sched_path = ckpt / "scheduler.pt"
        if sched_path.exists():
            self.scheduler.load_state_dict(torch.load(str(sched_path), map_location=self.device))
            logger.info("Loaded scheduler state from %s", sched_path)

        # 4. Trainer state
        state_path = ckpt / "trainer_state.json"
        resume_step = 0
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            resume_step = state.get("step", 0)
            self.global_step = state.get("global_step", 0)
            self.best_eval_hit_rate = state.get("best_eval_hit_rate", -1.0)
            logger.info(
                "Loaded trainer state: step=%d, global_step=%d, best_hit_rate=%.1f%%",
                resume_step, self.global_step, self.best_eval_hit_rate * 100,
            )

        return resume_step

    # ------------------------------------------------------------------
    #  Hugging Face Hub upload
    # ------------------------------------------------------------------
    def push_to_hub(self, checkpoint_dir: str | None = None, repo_id_override: str | None = None) -> None:
        """Push model checkpoint to Hugging Face Hub.

        Parameters
        ----------
        checkpoint_dir : str, optional
            Path to the checkpoint to upload.  Defaults to
            ``best_checkpoint`` if it exists, otherwise the latest
            checkpoint in the output directory.
        repo_id_override : str, optional
            Override the Hub repo ID (e.g. for phase-specific uploads).
        """
        from huggingface_hub import HfApi, ModelCard, ModelCardData

        hub = self.cfg.hub
        repo_id = repo_id_override or hub.hub_model_id
        if not repo_id:
            logger.warning("hub.hub_model_id is not set — skipping push_to_hub.")
            return

        # Resolve which checkpoint to upload
        if checkpoint_dir is not None:
            upload_dir = Path(checkpoint_dir)
        else:
            best = Path(self.cfg.training.output_dir) / "best_checkpoint"
            if best.exists():
                upload_dir = best
            else:
                # Fall back to latest checkpoint-N/
                out = Path(self.cfg.training.output_dir)
                ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
                if not ckpts:
                    logger.warning("No checkpoints found to push.")
                    return
                upload_dir = ckpts[-1]

        logger.info("Pushing %s to HF Hub: %s …", upload_dir, repo_id)

        api = HfApi()

        # Create repo if it doesn't exist
        api.create_repo(repo_id, private=hub.hub_private, exist_ok=True)

        # Upload adapter files
        api.upload_folder(
            folder_path=str(upload_dir),
            repo_id=repo_id,
            ignore_patterns=["optimizer.pt", "scheduler.pt"],
        )

        # Create a model card
        trainer_state_path = upload_dir / "trainer_state.json"
        eval_info = ""
        if trainer_state_path.exists():
            with open(trainer_state_path) as f:
                state = json.load(f)
            hit_rate = state.get("best_eval_hit_rate", state.get("eval_metrics", {}).get("core", {}).get("hit_rate"))
            if hit_rate is not None:
                eval_info = f"- **Best eval hit rate**: {hit_rate:.1%}\n"

        card_content = f"""\
---
library_name: peft
base_model: {self.cfg.model.model_name}
tags:
  - grpo
  - reinforcement-learning
  - duck-hunt
  - vision-language-model
  - tool-calling
---

# {repo_id.split('/')[-1]}

LoRA adapter for [{self.cfg.model.model_name}](https://huggingface.co/{self.cfg.model.model_name}),
fine-tuned with GRPO to play Duck Hunt.

## Training

- **Method**: Group Relative Policy Optimization (GRPO)
- **LoRA rank**: {self.cfg.lora.r}, alpha: {self.cfg.lora.lora_alpha}
- **Target modules**: {', '.join(self.cfg.lora.target_modules)}
- **Learning rate**: {self.cfg.training.learning_rate}
- **Generations per prompt**: {self.cfg.grpo.num_generations}
{eval_info}
## Usage

The adapter produces Mistral native tool calls (`[TOOL_CALLS]`), compatible with
OpenAI SDK via vLLM or TGI.

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoProcessor

model = AutoPeftModelForCausalLM.from_pretrained("{repo_id}")
processor = AutoProcessor.from_pretrained("{repo_id}")
```
"""
        card = ModelCard(card_content)
        card.push_to_hub(repo_id)

        logger.info("Pushed to https://huggingface.co/%s", repo_id)
