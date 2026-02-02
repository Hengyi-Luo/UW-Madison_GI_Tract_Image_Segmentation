from __future__ import annotations

import os
from dataclasses import fields
from typing import Any, Dict, Optional, Tuple

import torch


def checkpoint_paths(out_path: str) -> Tuple[str, str]:
    out_dir = os.path.dirname(out_path)
    return out_path, os.path.join(out_dir, "last.pt")


def load_resume_checkpoint(resume_from: str) -> Optional[dict]:
    if not resume_from:
        return None
    if not os.path.exists(resume_from):
        raise FileNotFoundError(f"resume_from not found: {resume_from}")
    return torch.load(resume_from, map_location="cpu")


def apply_cfg_from_checkpoint(cfg: Any, ckpt: dict) -> None:
    """Strict resume: load previous cfg from checkpoint, keep a small override set."""
    ckpt_cfg = ckpt.get("cfg")
    if not isinstance(ckpt_cfg, dict):
        return

    keep = {
        "resume_from": getattr(cfg, "resume_from", ""),
        "epochs": getattr(cfg, "epochs", None),
        "save_last_every": getattr(cfg, "save_last_every", 1),
        "debug": getattr(cfg, "debug", False),
    }
    if keep["epochs"] is None:
        keep.pop("epochs")

    allowed = {f.name for f in fields(type(cfg))}
    for k, v in ckpt_cfg.items():
        if k in allowed:
            setattr(cfg, k, v)
    for k, v in keep.items():
        setattr(cfg, k, v)


def _inner_optimizer(opt):
    return getattr(opt, "optimizer", opt)


def move_optimizer_state_to_device(opt, device: torch.device) -> None:
    opt = _inner_optimizer(opt)
    for state in opt.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "net", "network"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return {k: v for k, v in obj.items() if torch.is_tensor(v)}
    raise TypeError(f"Unsupported checkpoint type: {type(obj)}")


def _strip_known_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = k
        for p in ("module.", "model.", "net."):
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def load_init_weights(model: torch.nn.Module, init_from: str) -> int:
    """Load pretrained weights by shape match. Returns number of loaded tensors."""
    if not init_from:
        return 0
    if not os.path.exists(init_from):
        raise FileNotFoundError(f"init_from not found: {init_from}")

    raw = torch.load(init_from, map_location="cpu")
    sd = _strip_known_prefixes(_extract_state_dict(raw))
    msd = model.state_dict()

    loadable = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
    msd.update(loadable)
    model.load_state_dict(msd)
    return len(loadable)

