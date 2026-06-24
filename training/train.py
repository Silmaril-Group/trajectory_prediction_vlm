"""Main training script for Duck Hunt GRPO.

Supports both Mistral and LiquidAI model families — auto-detected from config.

Usage::

    # Mistral (TRL GRPOTrainer)
    python train.py --config configs/ministral_config.yaml

    # LiquidAI with LoRA
    python train.py --config configs/liquidai_config.yaml

    # LiquidAI without LoRA
    python train.py --config configs/liquidai_nolora_config.yaml

    # Custom GRPO loop (any model)
    python train.py --config configs/liquidai_config.yaml --custom

    # CLI overrides
    python train.py --config configs/liquidai_config.yaml \\
        --override training.learning_rate=2e-5 \\
        --override grpo.num_generations=8

    # Layer configs (base + model-specific)
    python train.py \\
        --config configs/base_config.yaml \\
        --config configs/liquidai_config.yaml

    # Resume from checkpoint (custom mode)
    python train.py --config configs/liquidai_config.yaml --custom \\
        --resume outputs/lfm25_duckhunt_grpo/checkpoint-500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make training/src importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import FullConfig
from src.dataset import (
    DuckHuntPromptGenerator,
    make_format_reward_function,
    make_reward_function,
)
from src.model import load_model_and_processor, apply_lora
from src.utils import set_model_format

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===================================================================
#  Config helpers
# ===================================================================
def _parse_override(s: str) -> tuple[str, str | float | int | bool]:
    """Parse ``key=value`` into (key, typed_value)."""
    key, _, raw_val = s.partition("=")
    for converter in (int, float):
        try:
            return key, converter(raw_val)
        except ValueError:
            continue
    if raw_val.lower() in ("true", "false"):
        return key, raw_val.lower() == "true"
    return key, raw_val


def load_config(args: argparse.Namespace) -> FullConfig:
    """Build a ``FullConfig`` from CLI args (yamls + overrides)."""
    if len(args.config) == 1:
        cfg = FullConfig.from_yaml(args.config[0])
    else:
        cfg = FullConfig.from_yamls(*args.config)

    if args.override:
        overrides = dict(_parse_override(o) for o in args.override)
        cfg = FullConfig.with_cli_overrides(cfg, overrides)

    return cfg


# ===================================================================
#  TRL GRPOTrainer path
# ===================================================================
def train_trl(cfg: FullConfig, num_samples: int) -> None:
    """Train using TRL's ``GRPOTrainer``."""
    from peft import LoraConfig as PeftLoraConfig
    from trl import GRPOConfig as TRLGRPOConfig
    from trl import GRPOTrainer

    # 1. Generate dataset from environment
    logger.info("Generating %d training samples from environment …", num_samples)
    generator = DuckHuntPromptGenerator(cfg.environment)
    dataset = generator.generate(num_samples)
    logger.info("Dataset ready: %d rows", len(dataset))

    # 2. Load model + processor
    model, processor = load_model_and_processor(cfg.model)

    # 3. LoRA config (let GRPOTrainer apply it via peft_config)
    peft_config = None
    if cfg.lora.enabled:
        peft_config = PeftLoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=cfg.lora.target_modules,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
        )

    # 4. Build TRL GRPOConfig
    grpo = cfg.grpo
    train = cfg.training
    log_cfg = cfg.logging

    trl_config = TRLGRPOConfig(
        output_dir=train.output_dir,
        num_train_epochs=train.num_train_epochs,
        max_steps=train.max_steps,
        per_device_train_batch_size=train.per_device_train_batch_size,
        gradient_accumulation_steps=train.gradient_accumulation_steps,
        learning_rate=train.learning_rate,
        warmup_ratio=train.warmup_ratio,
        weight_decay=train.weight_decay,
        lr_scheduler_type=train.lr_scheduler_type,
        max_grad_norm=train.max_grad_norm,
        bf16=train.bf16,
        fp16=train.fp16,
        logging_steps=train.logging_steps,
        save_steps=train.save_steps,
        save_total_limit=train.save_total_limit,
        eval_strategy=train.eval_strategy,
        eval_steps=train.eval_steps,
        seed=train.seed,
        dataloader_num_workers=train.dataloader_num_workers,
        gradient_checkpointing=train.gradient_checkpointing,
        # GRPO args
        num_generations=grpo.num_generations,
        max_completion_length=grpo.max_completion_length,
        temperature=grpo.temperature,
        top_p=grpo.top_p,
        top_k=grpo.top_k,
        beta=grpo.beta,
        num_iterations=grpo.num_iterations,
        epsilon=grpo.epsilon,
        loss_type=grpo.loss_type,
        scale_rewards=grpo.scale_rewards,
        use_vllm=grpo.use_vllm,
        # Logging
        report_to=log_cfg.report_to,
        run_name=log_cfg.wandb_run_name,
        log_completions=log_cfg.log_completions,
        # Reward weights: [accuracy, format]
        reward_weights=[cfg.reward.accuracy_weight, cfg.reward.format_weight],
    )

    # 5. Create reward functions (use active format's parser)
    accuracy_fn = make_reward_function(cfg.reward, cfg.environment.max_horizon)
    format_fn = make_format_reward_function(cfg.environment.max_horizon)

    # 6. Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=trl_config,
        train_dataset=dataset,
        processing_class=processor,
        reward_funcs=[accuracy_fn, format_fn],
        peft_config=peft_config,
    )

    # 7. Train
    logger.info("Starting TRL GRPOTrainer …")
    trainer.train()

    # 8. Save final model
    final_dir = Path(train.output_dir) / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    logger.info("Training complete. Model saved to %s", final_dir)

    # 9. Push to HF Hub
    if cfg.hub.push_to_hub and cfg.hub.hub_model_id:
        _push_to_hub(cfg, str(final_dir))


# ===================================================================
#  HF Hub upload (shared by both paths)
# ===================================================================
def _push_to_hub(cfg: FullConfig, checkpoint_dir: str) -> None:
    """Push a checkpoint to Hugging Face Hub."""
    from huggingface_hub import HfApi, ModelCard

    repo_id = cfg.hub.hub_model_id
    logger.info("Pushing %s to HF Hub: %s …", checkpoint_dir, repo_id)

    api = HfApi()
    api.create_repo(repo_id, private=cfg.hub.hub_private, exist_ok=True)

    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        ignore_patterns=["optimizer.pt", "scheduler.pt"],
    )

    # Determine model-specific tags and tool format description
    model_lower = cfg.model.model_name.lower()
    if "liquidai" in model_lower or "lfm" in model_lower:
        extra_tags = "  - liquidai\n  - lfm2.5"
        tool_format = "LiquidAI Pythonic (`<|tool_call_start|>[shoot(...)]<|tool_call_end|>`)"
    else:
        extra_tags = "  - mistral"
        tool_format = "Mistral native (`[TOOL_CALLS] [{...}]`)"

    lora_info = ""
    if cfg.lora.enabled:
        lora_info = (
            f"\n- **LoRA rank**: {cfg.lora.r}, alpha: {cfg.lora.lora_alpha}"
            f"\n- **Target modules**: {', '.join(cfg.lora.target_modules)}"
        )
    else:
        lora_info = "\n- **Mode**: Full fine-tune (no LoRA)"

    card_content = f"""\
---
library_name: {"peft" if cfg.lora.enabled else "transformers"}
base_model: {cfg.model.model_name}
tags:
  - grpo
  - reinforcement-learning
  - duck-hunt
  - vision-language-model
  - tool-calling
{extra_tags}
---

# {repo_id.split('/')[-1]}

{"LoRA adapter" if cfg.lora.enabled else "Fine-tuned model"} for \
[{cfg.model.model_name}](https://huggingface.co/{cfg.model.model_name}),
trained with GRPO to play Duck Hunt.

## Training

- **Method**: Group Relative Policy Optimization (GRPO)
- **Tool format**: {tool_format}
{lora_info}
- **Learning rate**: {cfg.training.learning_rate}

## Usage

```python
{"from peft import AutoPeftModelForCausalLM" if cfg.lora.enabled else "from transformers import AutoModelForImageTextToText"}
from transformers import AutoProcessor

{"model = AutoPeftModelForCausalLM" if cfg.lora.enabled else "model = AutoModelForImageTextToText"}.from_pretrained("{repo_id}")
processor = AutoProcessor.from_pretrained("{repo_id}")
```
"""
    card = ModelCard(card_content)
    card.push_to_hub(repo_id)

    logger.info("Pushed to https://huggingface.co/%s", repo_id)


# ===================================================================
#  Custom GRPO loop path
# ===================================================================
def train_custom(cfg: FullConfig, resume_from: str | None = None) -> None:
    """Train using the custom ``DuckHuntGRPOTrainer``."""
    import os
    from src.environment import DuckHuntEnvWrapper
    from src.trainer import DuckHuntGRPOTrainer

    # Detect distributed mode
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    accelerator = None
    if distributed:
        from accelerate import Accelerator
        from accelerate import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            mixed_precision="bf16" if cfg.training.bf16 else ("fp16" if cfg.training.fp16 else "no"),
            kwargs_handlers=[ddp_kwargs],
        )
        logger.info(
            "Distributed training: %d GPUs, rank %d, device %s",
            accelerator.num_processes, accelerator.process_index, accelerator.device,
        )
        # Different random seed per rank for environment diversity
        import random
        rank_seed = cfg.training.seed + accelerator.process_index
        random.seed(rank_seed)

    # 1. Environment (each rank gets its own instance)
    env = DuckHuntEnvWrapper(cfg.environment)

    # 2. Model + LoRA
    model, processor = load_model_and_processor(cfg.model, distributed=distributed)
    if cfg.lora.enabled:
        model = apply_lora(model, cfg.lora)

    # 3. Trainer
    trainer = DuckHuntGRPOTrainer(
        model=model,
        processor=processor,
        env=env,
        config=cfg,
        accelerator=accelerator,
    )

    # 4. Train (with optional resume)
    logger.info("Starting custom GRPO training loop …")
    trainer.train(resume_from=resume_from)


# ===================================================================
#  CLI
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Duck Hunt GRPO Training")
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        required=True,
        help="Path to YAML config file (can be repeated to layer configs).",
    )
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        help="Dot-separated override, e.g. training.learning_rate=2e-5",
    )
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Use custom GRPO training loop instead of TRL GRPOTrainer.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of prompt samples to generate (TRL mode only).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint dir to resume training from (custom mode only).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push final model to Hugging Face Hub after training.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="HF Hub repo id, e.g. 'username/duckhunt-ministral-grpo'.",
    )
    parser.add_argument(
        "--latency-ms",
        type=int,
        default=None,
        help="Force a single fixed processing-time latency (ms). "
             "Overrides environment.latency_options_ms to [value].",
    )
    args = parser.parse_args()

    cfg = load_config(args)

    # Force a single latency value if requested
    if args.latency_ms is not None:
        cfg.environment.latency_options_ms = [args.latency_ms]
        logger.info("Latency pinned to single value: %d ms", args.latency_ms)

    # Apply HF Hub CLI overrides
    if args.push_to_hub:
        cfg.hub.push_to_hub = True
    if args.hub_model_id:
        cfg.hub.hub_model_id = args.hub_model_id

    # Activate the right tool-call format based on model name
    set_model_format(cfg.model.model_name)

    logger.info(
        "Config loaded: model=%s, lora=%s",
        cfg.model.model_name, cfg.lora.enabled,
    )

    if args.custom:
        train_custom(cfg, resume_from=args.resume)
    else:
        train_trl(cfg, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
