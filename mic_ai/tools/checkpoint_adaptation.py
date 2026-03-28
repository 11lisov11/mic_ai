from __future__ import annotations

from typing import Dict, List

import torch


_ACTION_HEAD_KEYS = {
    "actor_mu.weight",
    "actor_mu.bias",
    "actor_head.weight",
    "actor_head.bias",
    "log_std",
}

_ID_LIKE_SECOND_SLOT_MODES = {"ai_current", "ai_speed", "foc_assist"}


def _should_remap_single_action_to_second_slot(
    *,
    key: str,
    value: torch.Tensor,
    target: torch.Tensor,
    target_control_mode: str | None,
) -> bool:
    mode = str(target_control_mode or "").strip().lower()
    if mode not in _ID_LIKE_SECOND_SLOT_MODES:
        return False
    if key not in _ACTION_HEAD_KEYS:
        return False
    if value.ndim != target.ndim:
        return False
    if value.ndim in (1, 2):
        return int(value.shape[0]) == 1 and int(target.shape[0]) == 2
    return False


def adapt_checkpoint_state_dict_for_model(
    state_dict: Dict[str, torch.Tensor],
    model_state_dict: Dict[str, torch.Tensor],
    *,
    target_control_mode: str | None = None,
) -> tuple[Dict[str, torch.Tensor], List[str]]:
    adapted: Dict[str, torch.Tensor] = {}
    adjusted: List[str] = []
    for key, value in state_dict.items():
        target = model_state_dict.get(key)
        if target is None or not isinstance(value, torch.Tensor):
            adapted[key] = value
            continue
        if tuple(value.shape) == tuple(target.shape):
            adapted[key] = value
            continue
        if _should_remap_single_action_to_second_slot(
            key=key,
            value=value,
            target=target,
            target_control_mode=target_control_mode,
        ):
            padded = target.detach().clone()
            padded.zero_()
            if value.ndim == 2:
                cols = min(int(value.shape[1]), int(target.shape[1]))
                padded[1, :cols] = value[0, :cols].to(dtype=target.dtype)
            else:
                padded[1] = value[0].to(dtype=target.dtype)
            adapted[key] = padded
            adjusted.append(key)
            continue
        if value.ndim == target.ndim and value.ndim in (1, 2):
            padded = target.detach().clone()
            padded.zero_()
            slices = tuple(slice(0, min(int(vdim), int(tdim))) for vdim, tdim in zip(value.shape, target.shape))
            padded[slices] = value[slices].to(dtype=target.dtype)
            adapted[key] = padded
            adjusted.append(key)
            continue
        adapted[key] = value
    return adapted, adjusted
