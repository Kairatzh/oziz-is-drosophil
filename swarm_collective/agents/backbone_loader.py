"""
backbone_loader.py — Loads the pretrained GlobalMapCNN + LocalCNN from the
Multi-Agent-Systems-Project checkpoint and exposes them as frozen feature
extractors for the Swarm Collective system.

The original CnnQNetwork checkpoint contains:
  global_cnn.*   – GlobalMapCNN (output: 1024-dim)
  local_cnn.*    – LocalCNN (output: 384-dim)
  fc1, fc2, fc3  – Dueling head MLP layers (NOT loaded)
  value_head, advantage_head – Q-head (NOT loaded)

We load only the CNN encoders and freeze them (or partially fine-tune).
"""

import sys
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# ── Path setup: allow importing from the Multi-Agent repo ────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]  # oziz/
_MAS_PATH  = _REPO_ROOT / "Multi-Agent-Systems-Project-main"

if str(_MAS_PATH) not in sys.path:
    sys.path.insert(0, str(_MAS_PATH))

# Now we can import the original network classes unchanged
from agents.networks import GlobalMapCNN, LocalCNN  # noqa: E402

# ── Constants (mirrors Multi-Agent/agents/networks.py) ───────────────────────
GLOBAL_CHANNELS  = 6
LOCAL_CHANNELS   = 5
CTX_DIM          = 5
FEAT_GLOBAL_DIM  = 1024
FEAT_LOCAL_DIM   = 384
FEAT_COMBINED    = FEAT_GLOBAL_DIM + FEAT_LOCAL_DIM + CTX_DIM  # 1413

# Which parameter name prefixes to freeze (backbone encoder layers)
FROZEN_PREFIXES = [
    "global_cnn.stem",
    "global_cnn.stage1",
    "global_cnn.stage2",
    "global_cnn.stage3",
    "global_cnn.transformer",
    "local_cnn.stem",
    "local_cnn.blocks",
    "local_cnn.linear",
]

# Which prefixes to fine-tune with a reduced LR (top projection layers)
FINETUNE_PREFIXES = [
    "global_cnn.head",
    "local_cnn.film",
]


class BackboneExtractor(nn.Module):
    """
    Wraps GlobalMapCNN + LocalCNN, optionally freezes encoder weights.

    Forward:
        obs_global : (B, 6, 32, 32)  or flat (B, 6144) — reshaped internally
        obs_local  : (B, 5, 32, 32)  or flat (B, 5120)
        ctx        : (B, 5)
    Returns:
        feat_global : (B, 1024)
        feat_local  : (B,  384)
        combined    : (B, 1413)  = cat[feat_global, feat_local, ctx]
    """

    def __init__(self, grid_size: int = 32):
        super().__init__()
        self.grid_size = grid_size
        self.global_cnn = GlobalMapCNN()
        self.local_cnn  = LocalCNN()

    # ------------------------------------------------------------------
    def load_pretrained(self, ckpt_path: str, strict: bool = False) -> None:
        """
        Load encoder weights from a CnnQNetwork checkpoint.
        We only load keys that start with 'global_cnn.' or 'local_cnn.'.
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Download it via:\n"
                "  from huggingface_hub import hf_hub_download\n"
                "  hf_hub_download('simoswish/MAS_MultiDroneExploration', "
                "'mas_50k_dr_faults_ep50000.pt', local_dir='pretrained/')"
            )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Strip 'network.' prefix if saved from DQNAgent wrapper
        if all(k.startswith("network.") for k in list(state.keys())[:5]):
            state = {k[len("network."):]: v for k, v in state.items()}

        encoder_state = {
            k: v for k, v in state.items()
            if k.startswith("global_cnn.") or k.startswith("local_cnn.")
        }
        missing, unexpected = self.load_state_dict(encoder_state, strict=strict)
        loaded = len(encoder_state)
        print(f"[BackboneExtractor] Loaded {loaded} tensors from {ckpt_path.name}")
        if missing:
            print(f"  Missing  ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"  Unexpected ({len(unexpected)}): {unexpected[:5]}")

    # ------------------------------------------------------------------
    def set_frozen(self, freeze_encoder: bool = True,
                   finetune_lr_mult: float = 0.1) -> None:
        """
        Freeze encoder layers; return param groups for the optimizer.
        Call this AFTER load_pretrained().
        """
        for name, param in self.named_parameters():
            if any(name.startswith(p) for p in FROZEN_PREFIXES):
                param.requires_grad_(not freeze_encoder)
            elif any(name.startswith(p) for p in FINETUNE_PREFIXES):
                param.requires_grad_(True)
                # Store multiplier as a tag for the optimizer builder
                param._lr_mult = finetune_lr_mult
            else:
                param.requires_grad_(True)

        n_frozen = sum(
            1 for n, p in self.named_parameters() if not p.requires_grad
        )
        n_total  = sum(1 for _ in self.parameters())
        print(f"[BackboneExtractor] Frozen {n_frozen}/{n_total} parameter tensors")

    # ------------------------------------------------------------------
    def forward(
        self,
        obs_global: torch.Tensor,   # (B, 6*32*32) or (B, 6, 32, 32)
        obs_local:  torch.Tensor,   # (B, 5*32*32) or (B, 5, 32, 32)
        ctx:        torch.Tensor,   # (B, 5)
    ):
        G = self.grid_size

        # Reshape if flat
        if obs_global.dim() == 2:
            obs_global = obs_global.view(-1, GLOBAL_CHANNELS, G, G)
        if obs_local.dim() == 2:
            obs_local = obs_local.view(-1, LOCAL_CHANNELS, G, G)

        feat_global = self.global_cnn(obs_global, ctx)          # (B, 1024)
        feat_local  = self.local_cnn(obs_local, feat_global)    # (B,  384)
        combined    = torch.cat([feat_global, feat_local, ctx], dim=-1)  # (B, 1413)
        return feat_global, feat_local, combined


# ── Convenience factory ───────────────────────────────────────────────────────

def build_backbone(
    ckpt_path: Optional[str] = None,
    freeze: bool = True,
    finetune_lr_mult: float = 0.1,
    grid_size: int = 32,
) -> BackboneExtractor:
    """
    Build and optionally load + freeze the backbone.
    If ckpt_path is None, returns a randomly-initialized backbone (for debugging).
    """
    backbone = BackboneExtractor(grid_size=grid_size)
    if ckpt_path is not None:
        backbone.load_pretrained(ckpt_path)
    backbone.set_frozen(freeze_encoder=freeze, finetune_lr_mult=finetune_lr_mult)
    return backbone


def get_optimizer_param_groups(
    model: nn.Module,
    base_lr: float = 3e-4,
) -> list:
    """
    Build parameter groups respecting the _lr_mult tag set by set_frozen().
    Returns a list suitable for torch.optim.Adam(..., params=groups).
    """
    frozen, finetune, trainable = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = getattr(param, "_lr_mult", 1.0)
        if mult < 1.0:
            finetune.append(param)
        else:
            trainable.append(param)

    groups = [{"params": trainable, "lr": base_lr}]
    if finetune:
        groups.append({"params": finetune, "lr": base_lr * 0.1})
    return groups
