from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime


def cfg_hash(cfg) -> str:
    raw = json.dumps(asdict(cfg), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def default_run_name(cfg) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    model = getattr(cfg, "model", "unet")
    return f"{ts}-{model}-pd{cfg.patch_d}-ph{cfg.patch_h}-pw{cfg.patch_w}-lr{cfg.lr}-seed{cfg.seed}"


def _fmt_template(s: str, ctx: dict) -> str:
    try:
        return s.format(**ctx)
    except KeyError as e:
        missing = e.args[0]
        allowed = ", ".join(sorted(ctx.keys()))
        raise ValueError(f"Unknown template key '{missing}' in '{s}'. Allowed keys: {allowed}")


def resolve_run_naming(cfg) -> None:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_ctx = dict(asdict(cfg))
    base_ctx.update({"ts": ts, "experiment": cfg.mlflow_experiment or "Default"})

    run_name = _fmt_template(cfg.mlflow_run_name, base_ctx) if cfg.mlflow_run_name else default_run_name(cfg)
    if cfg.debug and not run_name.endswith("-debug"):
        run_name = f"{run_name}-debug"

    cfg.mlflow_run_name = run_name
    if cfg.run_dir:
        cfg.run_dir = _fmt_template(cfg.run_dir, {**base_ctx, "run_name": run_name})


def git_info(repo_dir: str) -> dict:
    info = {}
    try:
        sha = subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        info["sha"] = sha.decode("utf-8", "replace").strip()
    except Exception:
        return info

    try:
        branch = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        )
        info["branch"] = branch.decode("utf-8", "replace").strip()
    except Exception:
        pass

    try:
        dirty = (
            subprocess.check_output(["git", "-C", repo_dir, "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode("utf-8", "replace")
            .strip()
            != ""
        )
        info["dirty"] = dirty
    except Exception:
        pass

    return info

