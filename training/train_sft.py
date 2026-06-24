"""SFT training for Duck Hunt VLM — teaches duck detection via hitbox prediction.

Trains the model to output locate(x1, y1, x2, y2) — the duck's hitbox coordinates
after latency frames. Uses standard cross-entropy loss on completion tokens only.

Supports single GPU and multi-GPU (via Accelerate).

Usage:
    # Single GPU
    python train_sft.py --dataset sft_dataset --model LiquidAI/LFM2-VL-3B

    # 2x GPU
    accelerate launch --num_processes=2 --mixed_precision=bf16 \
        train_sft.py --dataset sft_dataset --model LiquidAI/LFM2-VL-3B
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import get_scheduler
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

import re


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def _parse_locate_call(text: str) -> tuple[float, float, float, float] | None:
    """Parse locate(x1=..., y1=..., x2=..., y2=...) from model output."""
    match = re.search(
        r"locate\s*\("
        r".*?x1\s*=\s*([0-9.]+).*?y1\s*=\s*([0-9.]+)"
        r".*?x2\s*=\s*([0-9.]+).*?y2\s*=\s*([0-9.]+)",
        text, re.DOTALL,
    )
    if match:
        try:
            return (
                float(match.group(1)), float(match.group(2)),
                float(match.group(3)), float(match.group(4)),
            )
        except ValueError:
            pass
    return None


def _draw_bbox(
    frame: Image.Image,
    x1: float, y1: float, x2: float, y2: float,
    color: tuple = (0, 255, 0),
    label: str = "",
    width: int = 3,
) -> Image.Image:
    """Draw a bounding box on a copy of the frame. Coords are normalized 0-1."""
    from PIL import ImageDraw, ImageFont

    img = frame.copy().convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    px1 = int(x1 * w)
    py1 = int(y1 * h)
    px2 = int(x2 * w)
    py2 = int(y2 * h)

    draw.rectangle([px1, py1, px2, py2], outline=color, width=width)

    if label:
        draw.text((px1 + 2, py1 - 14), label, fill=color)

    return img


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_sft_dataset(dataset_dir: str) -> list[dict]:
    """Load the SFT dataset from disk."""
    ds_path = Path(dataset_dir)
    meta_path = ds_path / "dataset.json"

    with open(meta_path) as f:
        records = json.load(f)

    logger.info("Loaded %d records from %s", len(records), meta_path)
    return records


def _build_token_weights(
    full_ids: torch.Tensor,
    prompt_len: int,
    processor,
    value_weight: float = 5.0,
) -> torch.Tensor:
    """Build per-token loss weights.

    - Prompt tokens: weight 0 (effectively masked via labels=-100)
    - Structural completion tokens (tool_call_start, [locate(, x1=, ...): weight 1.0
    - Value tokens (digits): weight `value_weight` (amplified)

    This keeps the model producing structural tokens correctly while
    focusing learning capacity on the coordinate values.
    """
    weights = torch.zeros(full_ids.shape, dtype=torch.float32)
    # All completion tokens get weight 1.0 by default
    weights[prompt_len:] = 1.0

    # Value tokens get higher weight
    completion_ids = full_ids[prompt_len:]
    for i, tok_id in enumerate(completion_ids.tolist()):
        tok_text = processor.tokenizer.decode([tok_id], skip_special_tokens=False)
        if any(c.isdigit() for c in tok_text) and not any(c.isalpha() for c in tok_text):
            weights[prompt_len + i] = value_weight

    return weights


def build_training_sample(
    record: dict,
    processor,
    device: torch.device,
    value_weight: float = 5.0,
) -> dict | None:
    """Build tokenized input/labels for one SFT sample.

    Returns dict with 'input_ids', 'attention_mask', 'labels', 'token_weights'.
    Prompt tokens are masked (-100). All completion tokens contribute to loss,
    but value tokens are weighted higher (default 5x).
    """
    # Load images
    images = []
    for img_path in record["image_paths"]:
        images.append(Image.open(img_path).convert("RGB"))

    # Build user content with images (multimodal format)
    user_content = []
    for img in images:
        user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": record["user_text"]})

    # All content must use the list-of-dicts format for multimodal models
    full_messages = [
        {"role": "system", "content": [{"type": "text", "text": record["system_prompt"]}]},
        {"role": "user", "content": [{"type": "text", "text": record["user_fewshot"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": record["assistant_fewshot"]}]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": record["completion"]}]},
    ]

    # Prompt only (without the final assistant message)
    prompt_messages = [
        {"role": "system", "content": [{"type": "text", "text": record["system_prompt"]}]},
        {"role": "user", "content": [{"type": "text", "text": record["user_fewshot"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": record["assistant_fewshot"]}]},
        {"role": "user", "content": user_content},
    ]

    try:
        # Tokenize full conversation
        full_inputs = processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Tokenize prompt only (to find where completion starts)
        prompt_inputs = processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        full_ids = full_inputs["input_ids"][0]
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Standard completion-only labels: mask prompt, all completion tokens active
        labels = full_ids.clone()
        labels[:prompt_len] = -100

        # Per-token weights: structural tokens = 1.0, value tokens = value_weight
        token_weights = _build_token_weights(
            full_ids, prompt_len, processor, value_weight=value_weight,
        )

        n_value_tokens = int((token_weights == value_weight).sum().item())
        n_struct_tokens = int((token_weights == 1.0).sum().item())

        return {
            "input_ids": full_ids.unsqueeze(0).to(device),
            "attention_mask": full_inputs["attention_mask"].to(device),
            "labels": labels.unsqueeze(0).to(device),
            "token_weights": token_weights.unsqueeze(0).to(device),
            "prompt_len": prompt_len,
            "comp_len": len(full_ids) - prompt_len,
            "n_value_tokens": n_value_tokens,
            "n_struct_tokens": n_struct_tokens,
        }
    except Exception as e:
        logger.warning("Failed to build sample %s: %s", record.get("id", "?"), e)
        return None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_sft(
    model_name: str = "LiquidAI/LFM2.5-VL-1.6B",
    dataset_dir: str = "sft_dataset",
    output_dir: str = "outputs/sft",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.05,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_grad_norm: float = 1.0,
    lora_r: int = 16,
    lora_alpha: int = 32,
    save_steps: int = 200,
    wandb_project: str = "duckhunt-sft",
    wandb_run_name: str | None = None,
    value_weight: float = 5.0,
    seed: int = 42,
) -> None:
    """Run SFT training with optional distributed support."""
    random.seed(seed)
    torch.manual_seed(seed)

    # Detect distributed mode
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    accelerator = None
    if distributed:
        from accelerate import Accelerator, DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            mixed_precision="bf16",
            kwargs_handlers=[ddp_kwargs],
        )
        logger.info(
            "Distributed SFT: %d GPUs, rank %d, device %s",
            accelerator.num_processes, accelerator.process_index, accelerator.device,
        )
        # Different seed per rank for data shuffling diversity
        random.seed(seed + accelerator.process_index)

    is_main = accelerator.is_main_process if accelerator else True

    # Load dataset and split into train/val
    all_records = load_sft_dataset(dataset_dir)
    random.shuffle(all_records)

    val_size = max(1, int(len(all_records) * 0.1))  # 10% validation
    val_records = all_records[:val_size]
    train_records = all_records[val_size:]

    if is_main:
        logger.info("Train/val split: %d train, %d val", len(train_records), len(val_records))

    # In distributed mode, split TRAIN records across ranks (val stays full on each rank)
    if distributed:
        rank = accelerator.process_index
        num_ranks = accelerator.num_processes
        train_records = [r for i, r in enumerate(train_records) if i % num_ranks == rank]
        logger.info("Rank %d: %d train records", rank, len(train_records))

    records = train_records
    random.shuffle(records)

    # Load model + processor
    from src.model import load_model_and_processor, apply_lora
    from src.config import ModelConfig, LoRAConfig

    model_config = ModelConfig(
        model_name=model_name,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map="auto" if not distributed else None,
    )

    # Detect LoRA targets based on model
    # SFT needs vision encoder LoRA too — the LM alone can't learn to map
    # vision features to precise coordinates without tuning the vision side.
    if "lfm" in model_name.lower() or "liquid" in model_name.lower():
        # LM decoder + vision encoder + vision-to-LM projector
        target_modules = [
            # LM decoder
            "q_proj", "k_proj", "v_proj", "out_proj", "in_proj",
            "w1", "w2", "w3",
            # Vision encoder (SigLIP2) — attn already covered by q/k/v/out_proj
            # Vision encoder — FFN
            "fc1", "fc2",
            # Vision-to-LM projector (multi_modal_projector.linear_1/linear_2)
            "linear_1", "linear_2",
        ]
    elif "qwen" in model_name.lower():
        # LM decoder layers + vision encoder attention + merger projection
        target_modules = [
            # LM decoder
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            # Vision encoder — self-attention
            "qkv",
            "attn.proj",
            # Vision encoder — FFN
            "linear_fc1", "linear_fc2",
            # This also catches merger.linear_fc1/fc2 and deepstack_merger_list.N.linear_fc1/fc2
            # which are the critical vision→LM projection layers
        ]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoRAConfig(
        enabled=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
    )

    model, processor = load_model_and_processor(model_config, distributed=distributed)
    model = apply_lora(model, lora_config)

    # SFT uses right-padding (unlike GRPO which uses left)
    processor.tokenizer.padding_side = "right"

    # Gradient checkpointing
    model.gradient_checkpointing_enable()

    # Optimizer + scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    # Scheduler total steps = optimizer steps (not raw loop iterations)
    # In distributed mode, Accelerate wraps the scheduler and may step it
    # per-process, so use the per-rank optimizer step count.
    samples_per_rank = len(records)
    optimizer_steps_per_epoch = samples_per_rank // grad_accum
    total_steps = optimizer_steps_per_epoch * num_epochs
    # Multiply by world_size — Accelerate scales scheduler stepping in multi-GPU
    if distributed and world_size > 1:
        total_steps = total_steps * world_size
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Wrap with Accelerate
    if distributed:
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        device = accelerator.device
    else:
        device = next(model.parameters()).device

    if is_main:
        logger.info("SFT Training:")
        logger.info("  Model: %s", model_name)
        logger.info("  Dataset: %d records (per rank)", len(records))
        logger.info("  Epochs: %d", num_epochs)
        logger.info("  Total optimizer steps: %d (warmup: %d)", total_steps, warmup_steps)
        logger.info("  LR: %s, LoRA r=%d alpha=%d", learning_rate, lora_r, lora_alpha)
        logger.info("  Grad accum: %d", grad_accum)
        logger.info("  Distributed: %s (%d GPUs)", distributed, world_size)
        logger.info("  Value weight: %.1f (structural=1.0)", value_weight)

        # Diagnostic: check token distribution in a sample
        if records:
            sample = build_training_sample(records[0], processor, device, value_weight=value_weight)
            if sample:
                logger.info(
                    "  Sample diagnostic: %d value tokens + %d structural tokens = %d total completion tokens",
                    sample["n_value_tokens"], sample["n_struct_tokens"], sample["comp_len"],
                )

    # W&B — main process only
    wandb_active = False
    if _WANDB and wandb_project and is_main:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f"sft-{model_name.split('/')[-1]}",
            config={
                "model": model_name,
                "dataset_size": len(records) * world_size,
                "epochs": num_epochs,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "grad_accum": grad_accum,
                "distributed": distributed,
                "num_gpus": world_size,
            },
        )
        wandb_active = True

    # Training
    model.train()
    global_step = 0
    running_loss = 0.0
    samples_processed = 0

    out_path = Path(output_dir)
    if is_main:
        out_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        random.shuffle(records)
        epoch_loss = 0.0
        epoch_samples = 0

        for i, record in enumerate(records):
            sample = build_training_sample(record, processor, device, value_weight=value_weight)
            if sample is None:
                continue

            # Forward pass (don't pass labels — we compute loss manually with weights)
            outputs = model(
                input_ids=sample["input_ids"],
                attention_mask=sample["attention_mask"],
            )
            logits = outputs.logits  # (1, T, V)

            # Shift for causal LM: predict token t+1 from logits[t]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = sample["labels"][:, 1:].contiguous()
            shift_weights = sample["token_weights"][:, 1:].contiguous()

            # Per-token cross entropy
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )  # (B*T,)
            per_token_loss = per_token_loss.view(shift_labels.shape)  # (B, T)

            # Apply weights (weight is 0 for prompt, 1 for structural, value_weight for values)
            weighted_loss = per_token_loss * shift_weights
            # Normalize by sum of weights (so loss scale is consistent)
            total_weight = shift_weights.sum().clamp(min=1.0)
            loss_raw = weighted_loss.sum() / total_weight
            loss = loss_raw / grad_accum

            if distributed:
                accelerator.backward(loss)
            else:
                loss.backward()

            running_loss += loss_raw.item()
            epoch_loss += loss_raw.item()
            epoch_samples += 1
            samples_processed += 1

            # Optimizer step
            if (i + 1) % grad_accum == 0:
                if distributed:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                grad_norm_val = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log — main process only
                avg_loss = running_loss / grad_accum
                lr = scheduler.get_last_lr()[0]

                if global_step % 10 == 0 and is_main:
                    logger.info(
                        "epoch=%d  step=%d  loss=%.4f  lr=%.2e  grad=%.4f",
                        epoch + 1, global_step, avg_loss, lr, grad_norm_val,
                    )

                if wandb_active:
                    wandb.log({
                        "sft/loss": avg_loss,
                        "sft/learning_rate": lr,
                        "sft/gradient_norm": grad_norm_val,
                        "sft/epoch": epoch + 1,
                        "sft/step": global_step,
                        "sft/samples_processed": samples_processed,
                    })

                running_loss = 0.0

                # Save checkpoint — main process only
                if global_step % save_steps == 0 and is_main:
                    ckpt_dir = out_path / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(exist_ok=True)
                    unwrapped = accelerator.unwrap_model(model) if distributed else model
                    unwrapped.save_pretrained(str(ckpt_dir))
                    processor.save_pretrained(str(ckpt_dir))
                    logger.info("Saved checkpoint to %s", ckpt_dir)

        if is_main:
            avg_epoch_loss = epoch_loss / max(epoch_samples, 1)
            logger.info(
                "Epoch %d complete: train_loss=%.4f, samples=%d",
                epoch + 1, avg_epoch_loss, epoch_samples,
            )

        # Validation — run on main process only
        if is_main:
            gen_model = accelerator.unwrap_model(model) if distributed else model
            gen_model.eval()
            val_loss = 0.0
            val_samples = 0
            input_images = []        # separate W&B panel: input frames
            future_images = []       # separate W&B panel: future frames with GT + pred boxes
            num_visual_samples = 8

            with torch.no_grad():
                for vi, val_record in enumerate(val_records):
                    val_sample = build_training_sample(val_record, processor, device, value_weight=value_weight)
                    if val_sample is None:
                        continue

                    val_out = gen_model(
                        input_ids=val_sample["input_ids"],
                        attention_mask=val_sample["attention_mask"],
                    )
                    vlogits = val_out.logits[:, :-1, :].contiguous()
                    vlabels = val_sample["labels"][:, 1:].contiguous()
                    vweights = val_sample["token_weights"][:, 1:].contiguous()
                    vfct = torch.nn.CrossEntropyLoss(reduction="none")
                    vptl = vfct(
                        vlogits.view(-1, vlogits.size(-1)),
                        vlabels.view(-1),
                    ).view(vlabels.shape)
                    vl = (vptl * vweights).sum() / vweights.sum().clamp(min=1.0)
                    val_loss += vl.item()
                    val_samples += 1

                    # Visual logging for a few samples
                    if vi < num_visual_samples and wandb_active:
                        # Run inference to get predicted hitbox
                        prompt_len = val_sample["prompt_len"]
                        gen_ids = gen_model.generate(
                            input_ids=val_sample["input_ids"][:, :prompt_len],
                            attention_mask=val_sample["attention_mask"][:, :prompt_len],
                            max_new_tokens=60,
                            do_sample=False,
                        )
                        pred_tokens = gen_ids[0, prompt_len:]
                        pred_text = processor.decode(pred_tokens, skip_special_tokens=False)
                        pred_bbox = _parse_locate_call(pred_text)

                        gt_bbox = (val_record["x1"], val_record["y1"],
                                   val_record["x2"], val_record["y2"])

                        if pred_bbox:
                            pred_str = f"({pred_bbox[0]:.2f},{pred_bbox[1]:.2f})→({pred_bbox[2]:.2f},{pred_bbox[3]:.2f})"
                        else:
                            pred_str = f"PARSE_FAIL: {pred_text[:60]}"
                        gt_str = f"({gt_bbox[0]:.2f},{gt_bbox[1]:.2f})→({gt_bbox[2]:.2f},{gt_bbox[3]:.2f})"

                        # --- INPUT FRAMES panel ---
                        img_paths = val_record["image_paths"]
                        for fi, ip in enumerate(img_paths):
                            input_images.append(wandb.Image(
                                Image.open(ip).convert("RGB"),
                                caption=f"val_{vi} frame_{fi} lat={val_record['latency_frames']}f",
                            ))

                        # --- FUTURE FRAME panel: GT box (green) + pred box (red) ---
                        future_path = val_record.get("future_frame_path")
                        if future_path and Path(future_path).exists():
                            future_frame = Image.open(future_path).convert("RGB")

                            # Green = ground truth (should align with duck in this frame)
                            future_annotated = _draw_bbox(
                                future_frame,
                                gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3],
                                color=(0, 255, 0), label="GT", width=3,
                            )

                            # Red = model prediction
                            if pred_bbox:
                                future_annotated = _draw_bbox(
                                    future_annotated,
                                    pred_bbox[0], pred_bbox[1],
                                    pred_bbox[2], pred_bbox[3],
                                    color=(255, 0, 0), label="PRED", width=3,
                                )

                            future_images.append(wandb.Image(
                                future_annotated,
                                caption=f"val_{vi} lat={val_record['latency_frames']}f GT={gt_str} PRED={pred_str}",
                            ))

                        logger.info(
                            "  val[%d] lat=%df GT=%s PRED=%s",
                            vi, val_record["latency_frames"], gt_str, pred_str,
                        )

            avg_val_loss = val_loss / max(val_samples, 1)
            logger.info(
                "Epoch %d validation: val_loss=%.4f, val_samples=%d",
                epoch + 1, avg_val_loss, val_samples,
            )

            if wandb_active:
                log_dict = {
                    "sft/val_loss": avg_val_loss,
                    "sft/train_loss_epoch": avg_epoch_loss,
                    "sft/epoch": epoch + 1,
                }
                if input_images:
                    log_dict["sft/val_input_frames"] = input_images
                if future_images:
                    log_dict["sft/val_future_frames"] = future_images
                wandb.log(log_dict)

            model.train()

    # Final save — main process only
    if is_main:
        final_dir = out_path / "final"
        final_dir.mkdir(exist_ok=True)
        unwrapped = accelerator.unwrap_model(model) if distributed else model
        unwrapped.save_pretrained(str(final_dir))
        processor.save_pretrained(str(final_dir))
        logger.info("Training complete. Final model saved to %s", final_dir)

    if wandb_active:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SFT training for Duck Hunt VLM")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to SFT dataset directory")
    parser.add_argument("--model", type=str, default="LiquidAI/LFM2.5-VL-1.6B",
                        help="Base model name")
    parser.add_argument("--output", type=str, default="outputs/sft",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--wandb-project", type=str, default="duckhunt-sft")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--value-weight", type=float, default=5.0,
                        help="Loss weight for value (digit) tokens vs structural tokens (=1.0)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_sft(
        model_name=args.model,
        dataset_dir=args.dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        grad_accum=args.grad_accum,
        save_steps=args.save_steps,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_name,
        value_weight=args.value_weight,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
