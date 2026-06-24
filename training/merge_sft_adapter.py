"""Merge SFT LoRA adapter into base model weights.

After SFT, merge the adapter into the base model so GRPO can apply
a fresh LoRA on top for refinement.

Usage:
    python merge_sft_adapter.py \
        --base LiquidAI/LFM2.5-VL-1.6B \
        --adapter outputs/sft/final \
        --output outputs/sft_merged
"""

import argparse
import logging
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge SFT LoRA into base model")
    parser.add_argument("--base", type=str, required=True, help="Base model name/path")
    parser.add_argument("--adapter", type=str, required=True, help="SFT adapter path")
    parser.add_argument("--output", type=str, required=True, help="Output path for merged model")
    args = parser.parse_args()

    logger.info("Loading base model: %s", args.base)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )

    logger.info("Loading adapter: %s", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)

    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    logger.info("Saving merged model to: %s", args.output)
    model.save_pretrained(args.output)

    # Also save processor
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    processor.save_pretrained(args.output)

    logger.info("Done. Use this as model_name in GRPO config for refinement.")


if __name__ == "__main__":
    main()
