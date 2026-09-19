"""
LoRA (Low-Rank Adaptation) adapter for parameter-efficient fine-tuning of iris feature models.

Enables adapting foundation models (like DINOv2) using less than 1% of the original
parameters while preventing catastrophic forgetting of base representations.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class LoRALinear(nn.Module):
        """Low-Rank Adaptation wrapper around an nn.Linear layer."""

        def __init__(
            self,
            base_layer: nn.Linear,
            rank: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.base_layer = base_layer
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank

            in_features = base_layer.in_features
            out_features = base_layer.out_features

            # Freeze base weights
            self.base_layer.weight.requires_grad = False
            if self.base_layer.bias is not None:
                self.base_layer.bias.requires_grad = False

            # LoRA low-rank matrices: B @ A
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

            # Initialize: Gaussian for A, zero for B so delta is initially zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = self.base_layer(x)
            lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return base_out + lora_out


    def apply_lora_to_model(
        model: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: Optional[List[str]] = None,
    ) -> int:
        """Replace target linear layers in model with LoRALinear layers.

        Returns:
            Number of adapted layers.
        """
        if target_modules is None:
            target_modules = ["qkv", "proj", "linear", "fc"]

        adapted_count = 0
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                if any(t in name.lower() for t in target_modules):
                    lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                    setattr(model, name, lora_layer)
                    adapted_count += 1
            else:
                adapted_count += apply_lora_to_model(module, rank, alpha, target_modules)

        return adapted_count

else:
    class LoRALinear:
        pass

    def apply_lora_to_model(*args, **kwargs):
        return 0
