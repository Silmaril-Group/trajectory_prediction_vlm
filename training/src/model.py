"""Model loading, LoRA setup, and inference verification for Ministral-3-8B."""

from __future__ import annotations

import logging

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from .config import FullConfig, LoRAConfig, ModelConfig
from .utils import TOOLS, parse_tool_call

logger = logging.getLogger(__name__)

# Dtype string → torch dtype
_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# 2.1  Load model + processor
# ---------------------------------------------------------------------------
def load_model_and_processor(
    config: ModelConfig,
    distributed: bool = False,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load the model and its processor.

    Uses ``AutoModelForImageTextToText`` which resolves to the
    correct architecture based on the model config.

    Parameters
    ----------
    distributed : bool
        If True, skip device_map (Accelerate handles placement for DDP).
    """
    model_name = config.model_name
    dtype = _DTYPE_MAP.get(config.torch_dtype, torch.bfloat16)

    logger.info("Loading processor from %s …", model_name)
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=config.trust_remote_code,
    )

    # GRPO / batch generation requires left-padding
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Quantization config (QLoRA)
    quantization_config = None
    if config.quantization in ("4bit", "4"):
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Loading model in 4-bit (QLoRA mode)")
    elif config.quantization in ("8bit", "8"):
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Loading model in 8-bit")

    # In distributed mode, skip device_map — Accelerate handles GPU placement.
    # device_map="auto" uses model parallelism which conflicts with DDP.
    device_map = None if distributed else config.device_map

    logger.info("Loading model from %s (dtype=%s, distributed=%s) …", model_name, config.torch_dtype, distributed)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=config.attn_implementation,
        trust_remote_code=config.trust_remote_code,
        quantization_config=quantization_config,
    )

    # ---- Diagnostics ----
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model architecture: %s", model.__class__.__name__)
    logger.info("Total parameters:   %.2f B", total_params / 1e9)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info("VRAM allocated:     %.2f GB", allocated)

    return model, processor


# ---------------------------------------------------------------------------
# 2.2  Apply LoRA
# ---------------------------------------------------------------------------
def _find_target_modules(model: torch.nn.Module) -> list[str]:
    """Auto-detect linear projection layers suitable for LoRA."""
    targets: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Keep the last part of the qualified name (e.g. "q_proj")
            short = name.rsplit(".", 1)[-1]
            if short.endswith("_proj"):
                targets.add(short)
    found = sorted(targets)
    logger.info("Auto-detected LoRA target modules: %s", found)
    return found


def apply_lora(
    model: AutoModelForImageTextToText,
    lora_cfg: LoRAConfig,
) -> AutoModelForImageTextToText:
    """Wrap *model* with LoRA adapters.

    Returns the PeftModel (same reference, but with LoRA layers injected).
    """
    if not lora_cfg.enabled:
        logger.info("LoRA disabled — returning base model unchanged.")
        return model

    target_modules = lora_cfg.target_modules
    if not target_modules or target_modules == ["auto"]:
        target_modules = _find_target_modules(model)

    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=target_modules,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )

    # Required for gradient checkpointing with LoRA
    model.enable_input_require_grads()

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


# ---------------------------------------------------------------------------
# 2.3  Verify VLM inference  (uses native Mistral tool-calling tokens)
# ---------------------------------------------------------------------------
def test_inference(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
) -> bool:
    """Run a quick forward pass with dummy images to verify the pipeline.

    Tools are passed via ``processor.apply_chat_template(tools=…)`` so the
    tokenizer inserts ``[AVAILABLE_TOOLS]…[/AVAILABLE_TOOLS]`` and the model
    is expected to reply with ``[TOOL_CALLS] [{…}]``.
    """
    logger.info("Running inference smoke-test …")

    # 1. Dummy images (4 × 256×256 RGB)
    dummy_images = [
        Image.new("RGB", (256, 256), color=(r * 60, 120, 80))
        for r in range(4)
    ]

    # 2. Build chat messages with images + tool definitions
    user_content: list[dict] = [
        {"type": "image", "image": img} for img in dummy_images
    ]
    user_content.append(
        {
            "type": "text",
            "text": (
                "Above are 4 game frames showing duck movement. "
                "Call the shoot tool with your best prediction."
            ),
        }
    )

    messages = [{"role": "user", "content": user_content}]

    # 3. Process through the processor with native tool tokens
    inputs = processor.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 4. Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    # Decode new tokens (keep special tokens so we can see [TOOL_CALLS])
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    decoded_raw = processor.decode(new_tokens, skip_special_tokens=False)
    decoded_clean = processor.decode(new_tokens, skip_special_tokens=True)

    logger.info("Smoke-test raw output:\n%s", decoded_raw)

    # 5. Try parsing as a tool call
    action = parse_tool_call(decoded_raw)
    if action is not None:
        logger.info(
            "Parsed tool call: x=%.3f, y=%.3f, horizon=%d",
            action.x, action.y, action.horizon,
        )

    success = len(decoded_clean.strip()) > 0
    if success:
        logger.info("Smoke-test PASSED.")
    else:
        logger.warning("Smoke-test FAILED — empty output.")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info("VRAM after inference: %.2f GB", allocated)

    return success


# ---------------------------------------------------------------------------
# Convenience: full setup pipeline
# ---------------------------------------------------------------------------
def setup_model(config: FullConfig) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load model → apply forward_mods → apply LoRA → return (model, processor).

    Forward mods MUST be applied before LoRA.  In transformers 5.x,
    ``get_peft_model()`` overrides ``PreTrainedModel.base_model`` (which uses
    ``base_model_prefix="model"`` to traverse the nested
    ``Mistral3ForConditionalGeneration.model.{vision_tower,language_model}``
    structure).  After LoRA wraps the model, that property resolves to
    ``LoraModel`` instead of ``Mistral3Model``, shifting every navigation path
    by one level and causing all forward-mod hooks to silently miss their
    targets (vision tower, decoder layers, etc.).  Applying forward mods first
    keeps them anchored to the raw model graph.
    """
    model, processor = load_model_and_processor(config.model)

    # Apply forward modifications BEFORE LoRA (raw model navigation required)
    if hasattr(config, "forward_mod") and config.forward_mod.enabled:
        from .forward_mods import apply_forward_mods

        model, applied_mods = apply_forward_mods(model, config.forward_mod)
        extra_params: list[torch.nn.Parameter] = []
        for mod in applied_mods:
            extra_params.extend(mod.extra_parameters())
        if extra_params:
            logger.info(
                "Forward mods added %d extra trainable parameters",
                sum(p.numel() for p in extra_params),
            )
            model._forward_mod_params = extra_params  # kept alive by the model

    model = apply_lora(model, config.lora)
    return model, processor
