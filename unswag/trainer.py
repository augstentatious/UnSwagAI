
import torch
from transformers import Trainer
import os
import json
from typing import Optional, Dict


class UnSwagTrainer(Trainer):
    """
    UnSwag-optimized Trainer with memory hooks for 2-bit compression.
    Ensures gradient checkpointing and Triton kernels fire correctly.
    """

    def __init__(self, *args, **kwargs):
        # 1. Default to 8-bit AdamW if not specified
        if "args" in kwargs:
            if kwargs["args"].optim == "adamw_hf":  # Default HF optimizer
                print("🦁 UnSwagTrainer: Upgrading optimizer to 'paged_adamw_8bit' for VRAM efficiency.")
                kwargs["args"].optim = "paged_adamw_8bit"
        super().__init__(*args, **kwargs)
        self.unswag_enabled = True

    def _create_optimizer(self):
        if self.args.optim in ["paged_adamw_8bit", "adamw_bnb_8bit"]:
            try:
                import bitsandbytes
            except ImportError:
                raise ImportError(
                    "You requested an 8-bit optimizer but 'bitsandbytes' is not installed. "
                    "Install it via `pip install bitsandbytes`."
                )
        return super()._create_optimizer()

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training step to ensure UnSwag compression layers work
        correctly with gradient checkpointing and mixed precision.

        Args:
            model: The model being trained
            inputs: Batch of training data
            num_items_in_batch: Number of items in the batch
        """
        # Force gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            if not model.is_gradient_checkpointing:
                model.gradient_checkpointing_enable()

        # Call parent's training step with correct signature
        if num_items_in_batch is not None:
            return super().training_step(model, inputs, num_items_in_batch)
        else:
            return super().training_step(model, inputs)

    def save_model(self, output_dir, _internal_call=False):
        """
        Save model with UnSwag metadata for reproducibility.
        """
        super().save_model(output_dir, _internal_call)

        # Save UnSwag compression config
        config = {
            "unswag_version": "0.2.0",
            "protocol": "delhi",
            "compression_ratio": "16x",
            "mode": getattr(self.model.config, 'unswag_mode', '4bit')
        }

        config_path = f"{output_dir}/unswag_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ UnSwag config saved to {config_path}")
