"""Flash-MoE runtime integration for CROSSFIRE-X.

Wraps anemll-flash-llama.cpp -- the llama.cpp fork that adds slot-bank
expert streaming for Mixture-of-Experts models. Supports both GPU-bank
(all experts resident) and slot-bank (hot experts cached, cold streamed
from NVMe) modes.

Reference: https://github.com/Anemll/anemll-flash-llama.cpp
"""

from __future__ import annotations

from crossfire.flashmoe.config import (
    FlashMoEBuildConfig,
    FlashMoEMode,
    SidecarConfig,
    SlotBankConfig,
)
from crossfire.flashmoe.runtime import FlashMoERuntime

__all__ = [
    "FlashMoEBuildConfig",
    "FlashMoEMode",
    "FlashMoERuntime",
    "SidecarConfig",
    "SlotBankConfig",
]
