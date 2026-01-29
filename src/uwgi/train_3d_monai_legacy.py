# Legacy training script (kept for reference).
import os
import re
import time
import json
import random
import hashlib
import subprocess
from datetime import datetime
from dataclasses import dataclass, fields
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    ScaleIntensityRange,
    RandFlip,
    RandRotate90,
    EnsureType,
)
from monai.utils import set_determinism


# -------------------------
# RLE decode (2D)
# -------------------------
def rle_decode(rle: str, height: int, width: int) -> np.ndarray:
    """
    Kaggle RLE decode for 2D mask.
    rle: "start length start length ..."
    Returns (H, W) uint8 mask {0,1}

    Note: Kaggle RLE for this comp is in column-major order.
    The reshape+transpose below matches common Kaggle implementations.
    """
    if rle is None or rle == "" or (isinstance(rle, float) and np.isnan(rle)):
        return np.zeros((height, width), dtype=np.uint8)

    s = rle.strip().split()
    starts = np.asarray(s[0::2], dtype=int) - 1  # to 0-index
    lengths = np.asarray(s[1::2], dtype=int)
    ends = starts + lengths

    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # column-major to (H,W)
    return img.reshape((width, height)).T


# -------------------------
# Parse scan filename
# -------------------------
_scan_re = re.compile(r"slice_(\d+)_([0-9]+)_([0-9]+)_([0-9.]+)_([0-9.]+)\.png")


def parse_scan_filename(fname: str) -> Tuple[int, int, int]:
    """
    From slice_XXXX_H_W_psx_psy.png get slice index and nominal H,W in filename.
    Returns (slice_idx, H, W)
    """
    m = _scan_re.search(os.path.basename(fname))
    if not m:
        raise ValueError(f"Unexpected scan filename: {fname}")
    slice_idx = int(m.group(1))
    h = int(m.group(2))
    w = int(m.group(3))
    return slice_idx, h, w


# -------------------------
# Build case_day -> sorted scan file paths
# -------------------------
def build_case_day_slices(train_dir: str) -> Dict[str, List[str]]:
    """
    Returns dict: key = "case123_day0", value = sorted list of scan file paths
    """
    out: Dict[str, List[str]] = {}
    for case in sorted(os.listdir(train_dir)):
        case_path = os.path.join(train_dir, case)
        if not os.path.isdir(case_path):
            continue

        for day in sorted(os.listdir(case_path)):
            day_path = os.path.join(case_path, day)
            scans_dir = os.path.join(day_path, "scans")
            if not os.path.isdir(scans_dir):
                continue

            files = [os.path.join(scans_dir, f) for f in os.listdir(scans_dir) if f.endswith(".png")]
            files.sort(key=lambda p: parse_scan_filename(p)[0])  # sort by slice index
            key = day  # e.g. "case123_day0"
            out[key] = files

    return out


# -------------------------
# Read train.csv and map RLE by (case_day, slice_idx)
# -------------------------
CLASSES = ["large_bowel", "small_bowel", "stomach"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}


def build_rle_index(train_csv_path: str) -> Dict[Tuple[str, int], Dict[str, str]]:
    """
    train.csv columns: id, class, segmentation
    id example: "case123_day0_slice_0001"
    Returns:
      idx[(case_day, slice_idx)] = {class_name: rle_string}
    """
    df = pd.read_csv(train_csv_path)
    idx: Dict[Tuple[str, int], Dict[str, str]] = {}

    for _, row in df.iterrows():
        _id = row["id"]
        cls = row["class"]
        seg = row["segmentation"]

        parts = _id.split("_")
        case_day = "_".join(parts[0:2])      # case123_day0
        slice_idx = int(parts[-1])           # 0001 -> 1 (int)
        key = (case_day, slice_idx)

        if key not in idx:
            idx[key] = {}
        idx[key][cls] = seg

    return idx


# -------------------------
# Dataset: sample random 3D patches
# -------------------------
class UWGI3DPatchDataset(Dataset):
    """
    Loads a case_day volume (D,H,W) + masks (3,D,H,W), then samples random patches.
    Fixes inconsistent PNG shapes by transposing or resizing to match (H,W) of first slice.
    """

    def __init__(
        self,
        case_days: List[str],
        case_day_slices: Dict[str, List[str]],
        rle_index: Dict[Tuple[str, int], Dict[str, str]],
        patch_size: Tuple[int, int, int],   # (D,H,W)
        samples_per_volume: int,
        cache_in_ram: bool = False,
        seed: int = 42,
        is_train: bool = True,
        verbose_shape_fix: bool = False,
    ):
        self.case_days = case_days
        self.case_day_slices = case_day_slices
        self.rle_index = rle_index

        self.patch_d, self.patch_h, self.patch_w = patch_size
        self.samples_per_volume = samples_per_volume
        self.cache_in_ram = cache_in_ram
        self.is_train = is_train
        self.verbose_shape_fix = verbose_shape_fix

        self.rng = random.Random(seed)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Basic intensity normalization: uint8 [0,255] -> float [0,1]
        self.xform = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),  # (D,H,W) -> (1,D,H,W)
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            EnsureType(data_type="tensor"),
        ])

        # Augmentations (spatial_axis indices are over spatial dims (D,H,W): 0/1/2)
        # Our tensor shape is (C, D, H, W)
        self.aug = Compose([
            RandFlip(prob=0.5, spatial_axis=2),                 # flip W
            RandFlip(prob=0.5, spatial_axis=1),                 # flip H
            RandRotate90(prob=0.3, max_k=3, spatial_axes=(1, 2)) # rotate in H-W plane
        ]) if is_train else None

        # Expand into an index list: each case_day repeated samples_per_volume times
        self._index = []
        for cd in self.case_days:
            for _ in range(self.samples_per_volume):
                self._index.append(cd)

    def __len__(self):
        return len(self._index)

    @staticmethod
    def _fix_to_hw(img2d: np.ndarray, H: int, W: int, mode: str, info: str = "") -> np.ndarray:
        """
        mode: "image" or "mask"
        For swapped shapes (W,H), transpose.
        Otherwise resize to (H,W).
        """
        if img2d.shape == (H, W):
            return img2d

        if img2d.shape == (W, H):
            return img2d.T

        # last resort: resize
        interp = cv2.INTER_AREA if mode == "image" else cv2.INTER_NEAREST
        fixed = cv2.resize(img2d, (W, H), interpolation=interp)
        return fixed

    def _load_volume(self, case_day: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          img_vol: (D,H,W) uint8
          mask_vol: (3,D,H,W) uint8 {0,1}
        """
        if self.cache_in_ram and case_day in self._cache:
            return self._cache[case_day]

        slice_files = self.case_day_slices[case_day]
        if len(slice_files) == 0:
            raise RuntimeError(f"No slice files for {case_day}")

        # IMPORTANT: determine H,W from the actual first image, not from filename
        img0 = cv2.imread(slice_files[0], cv2.IMREAD_GRAYSCALE)
        if img0 is None:
            raise RuntimeError(f"Failed to read image: {slice_files[0]}")
        H, W = img0.shape
        D = len(slice_files)

        img_vol = np.zeros((D, H, W), dtype=np.uint8)
        mask_vol = np.zeros((len(CLASSES), D, H, W), dtype=np.uint8)

        for z, f in enumerate(slice_files):
            slice_idx, _, _ = parse_scan_filename(f)

            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image: {f}")

            if img.shape != (H, W) and self.verbose_shape_fix:
                print(f"[WARN] shape mismatch {case_day} {os.path.basename(f)} "
                      f"img.shape={img.shape} expected={(H, W)}")

            img = self._fix_to_hw(img, H, W, mode="image")
            img_vol[z] = img

            # masks from RLE (decode to (H,W) to match our chosen reference)
            rle_map = self.rle_index.get((case_day, slice_idx), {})
            for cls_name, cls_i in CLASS2IDX.items():
                rle = rle_map.get(cls_name, "")
                m2d = rle_decode(rle, H, W)

                # Just in case (should already be (H,W))
                if m2d.shape != (H, W):
                    m2d = self._fix_to_hw(m2d, H, W, mode="mask")

                mask_vol[cls_i, z] = m2d

        if self.cache_in_ram:
            self._cache[case_day] = (img_vol, mask_vol)

        return img_vol, mask_vol

    def _random_crop_coords(self, D: int, H: int, W: int) -> Tuple[int, int, int]:
        """
        Random crop start (z0,y0,x0). If volume smaller than patch, start at 0.
        """
        z0 = 0 if D <= self.patch_d else self.rng.randint(0, D - self.patch_d)
        y0 = 0 if H <= self.patch_h else self.rng.randint(0, H - self.patch_h)
        x0 = 0 if W <= self.patch_w else self.rng.randint(0, W - self.patch_w)
        return z0, y0, x0

    def __getitem__(self, idx: int):
        case_day = self._index[idx]
        img_vol, mask_vol = self._load_volume(case_day)  # (D,H,W), (3,D,H,W)
        D, H, W = img_vol.shape

        # Ensure patch <= vol size in each dim; if not, take as much as possible
        pd = min(self.patch_d, D)
        ph = min(self.patch_h, H)
        pw = min(self.patch_w, W)

        z0 = 0 if D <= pd else self.rng.randint(0, D - pd)
        y0 = 0 if H <= ph else self.rng.randint(0, H - ph)
        x0 = 0 if W <= pw else self.rng.randint(0, W - pw)

        img_patch = img_vol[z0:z0+pd, y0:y0+ph, x0:x0+pw]              # (D,H,W)
        mask_patch = mask_vol[:, z0:z0+pd, y0:y0+ph, x0:x0+pw]          # (3,D,H,W)

        x = self.xform(img_patch)                                      # (1,D,H,W) float
        y = torch.from_numpy(mask_patch.astype(np.float32))            # (3,D,H,W)

        if self.aug is not None:
            # apply same augmentation to image+mask
            xy = torch.cat([x, y], dim=0)                              # (4,D,H,W)
            xy = self.aug(xy)
            x = xy[0:1]
            y = xy[1:]

        return x, y, case_day


# -------------------------
# Split by case to avoid leakage
# -------------------------
def split_case_days(case_day_keys: List[str], val_ratio: float = 0.2, seed: int = 42):
    rng = random.Random(seed)

    case_to_days: Dict[str, List[str]] = {}
    for cd in case_day_keys:
        case = cd.split("_")[0]  # "case123"
        case_to_days.setdefault(case, []).append(cd)

    cases = sorted(case_to_days.keys())
    rng.shuffle(cases)

    n_val = max(1, int(len(cases) * val_ratio))
    val_cases = set(cases[:n_val])

    train_days, val_days = [], []
    for c, days in case_to_days.items():
        if c in val_cases:
            val_days.extend(days)
        else:
            train_days.extend(days)

    return train_days, val_days


# -------------------------
# Train config
# -------------------------
@dataclass
class TrainCfg:
    data_root: str = "./input/uw-madison-gi-tract-image-segmentation"
    out: str = "./checkpoints/best.pt"
    run_dir: str = ""
    resume_from: str = ""  # path to a checkpoint like .../last.pt
    save_last_every: int = 1  # epochs; save .../last.pt every N epochs
    seed: int = 42

    patch_d: int = 80
    patch_h: int = 224
    patch_w: int = 224

    samples_per_volume: int = 12
    cache_in_ram: bool = False

    epochs: int = 30
    batch_size: int = 1
    lr: float = 1e-4
    num_workers: int = 4
    val_ratio: float = 0.2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose_shape_fix: bool = False
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    use_tensorboard: bool = False
    tb_dir: str = "./outputs/train_run/tb"
    log_steps: int = 50
    use_mlflow: bool = True
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "uwgi"
    mlflow_run_name: str = ""
    # If provided (or loaded from resume checkpoint), resume logging into the same MLflow run.
    mlflow_run_id: str = ""
    debug: bool = False


def _cfg_hash(cfg: TrainCfg) -> str:
    raw = json.dumps(cfg.__dict__, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _default_run_name(cfg: TrainCfg) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    return (
        f"{ts}-unet-none-"
        f"pd{cfg.patch_d}-ph{cfg.patch_h}-pw{cfg.patch_w}-"
        f"lr{cfg.lr}-seed{cfg.seed}"
    )


def _fmt_template(s: str, ctx: dict) -> str:
    try:
        return s.format(**ctx)
    except KeyError as e:
        missing = e.args[0]
        allowed = ", ".join(sorted(ctx.keys()))
        raise ValueError(f"Unknown template key '{missing}' in '{s}'. Allowed keys: {allowed}")


def _resolve_run_naming(cfg: TrainCfg) -> None:
    """Resolve optional `{...}` templates in mlflow_run_name/run_dir.

    Supported keys:
      - ts: timestamp like 20260127-163012
      - experiment: cfg.mlflow_experiment (or "Default")
      - run_name: the resolved MLflow run name
      - plus any TrainCfg fields (seed, lr, patch_d, ...)
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_ctx = dict(cfg.__dict__)
    base_ctx.update({"ts": ts, "experiment": cfg.mlflow_experiment or "Default"})

    if cfg.mlflow_run_name:
        run_name = _fmt_template(cfg.mlflow_run_name, base_ctx)
    else:
        run_name = _default_run_name(cfg)

    if cfg.debug and not run_name.endswith("-debug"):
        run_name = f"{run_name}-debug"

    cfg.mlflow_run_name = run_name

    if cfg.run_dir:
        cfg.run_dir = _fmt_template(cfg.run_dir, {**base_ctx, "run_name": run_name})


def _inner_optimizer(opt):
    # accelerate wraps optimizers; keep this resilient.
    return getattr(opt, "optimizer", opt)


def _move_optimizer_state_to_device(opt, device: torch.device) -> None:
    opt = _inner_optimizer(opt)
    for state in opt.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _checkpoint_paths(cfg: TrainCfg) -> tuple[str, str]:
    # Keep everything colocated with cfg.out for backward compatibility.
    out_dir = os.path.dirname(cfg.out)
    return cfg.out, os.path.join(out_dir, "last.pt")


def _git_info(repo_dir: str) -> dict:
    """Best-effort git metadata for experiment tracking.

    Returns a dict with optional keys: sha, branch, dirty.
    Never raises (training should not fail if git is unavailable).
    """
    info = {}
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8", "replace").strip()
        info["sha"] = sha
    except Exception:
        return info

    try:
        branch = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8", "replace").strip()
        info["branch"] = branch
    except Exception:
        pass

    try:
        # "dirty" if there are uncommitted changes (including untracked).
        dirty = subprocess.check_output(
            ["git", "-C", repo_dir, "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8", "replace").strip() != ""
        info["dirty"] = dirty
    except Exception:
        pass

    return info


def _load_resume_checkpoint(cfg: TrainCfg):
    if not cfg.resume_from:
        return None
    if not os.path.exists(cfg.resume_from):
        raise FileNotFoundError(f"resume_from not found: {cfg.resume_from}")
    return torch.load(cfg.resume_from, map_location="cpu")


def _apply_cfg_from_checkpoint(cfg: TrainCfg, ckpt: dict) -> None:
    """For strict resume, load the previous TrainCfg from checkpoint.

    We intentionally only preserve a small set of "override" fields from the
    current cfg (e.g., total epochs) to avoid accidental drift.
    """
    ckpt_cfg = ckpt.get("cfg")
    if not isinstance(ckpt_cfg, dict):
        return

    # Preserve only a few fields that are safe/expected to change on resume.
    keep = {
        "resume_from": cfg.resume_from,
        # If epochs is not provided for resume, use the checkpoint cfg's total epochs.
        "epochs": cfg.epochs,
        "save_last_every": cfg.save_last_every,
        "debug": cfg.debug,
    }
    if keep["epochs"] is None:
        keep.pop("epochs")

    allowed = {f.name for f in fields(TrainCfg)}
    for k, v in ckpt_cfg.items():
        if k in allowed:
            setattr(cfg, k, v)

    for k, v in keep.items():
        setattr(cfg, k, v)


def train(cfg: TrainCfg):
    set_determinism(seed=cfg.seed)
    ckpt = _load_resume_checkpoint(cfg)

    # If resuming and run_dir still uses templates, keep writing into the existing run folder.
    if ckpt is not None and cfg.resume_from:
        _apply_cfg_from_checkpoint(cfg, ckpt)
        resume_dir = os.path.dirname(cfg.resume_from)
        if (not cfg.run_dir) or ("{" in cfg.run_dir and "}" in cfg.run_dir):
            cfg.run_dir = resume_dir

        # Resume MLflow run_id if present in checkpoint (type-3 resume into the same run).
        if not cfg.mlflow_run_id:
            cfg.mlflow_run_id = ckpt.get("mlflow_run_id", "") or ckpt.get("mlflow", {}).get("run_id", "")

    _resolve_run_naming(cfg)
    if cfg.run_dir:
        cfg.out = os.path.join(cfg.run_dir, "best.pt")
        if cfg.use_tensorboard:
            cfg.tb_dir = os.path.join(cfg.run_dir, "tb")
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)

    train_dir = os.path.join(cfg.data_root, "train")
    train_csv = os.path.join(cfg.data_root, "train.csv")

    print("Building slice index...")
    case_day_slices = build_case_day_slices(train_dir)
    all_case_days = sorted(case_day_slices.keys())
    print(f"Found case_day volumes: {len(all_case_days)}")

    print("Building RLE index...")
    rle_index = build_rle_index(train_csv)

    train_days, val_days = split_case_days(all_case_days, val_ratio=cfg.val_ratio, seed=cfg.seed)
    print(f"Train volumes: {len(train_days)} | Val volumes: {len(val_days)}")

    if cfg.samples_per_volume > 512:
        print(f"[WARN] samples_per_volume={cfg.samples_per_volume} is huge. "
              f"This will make each epoch extremely long. Consider 8~32 for a start.")

    train_ds = UWGI3DPatchDataset(
        case_days=train_days,
        case_day_slices=case_day_slices,
        rle_index=rle_index,
        patch_size=(cfg.patch_d, cfg.patch_h, cfg.patch_w),
        samples_per_volume=cfg.samples_per_volume,
        cache_in_ram=cfg.cache_in_ram,
        seed=cfg.seed,
        is_train=True,
        verbose_shape_fix=cfg.verbose_shape_fix,
    )

    val_ds = UWGI3DPatchDataset(
        case_days=val_days,
        case_day_slices=case_day_slices,
        rle_index=rle_index,
        patch_size=(cfg.patch_d, cfg.patch_h, cfg.patch_w),
        samples_per_volume=max(1, cfg.samples_per_volume // 2),
        cache_in_ram=cfg.cache_in_ram,
        seed=cfg.seed + 999,
        is_train=False,
        verbose_shape_fix=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    accelerator = Accelerator(
        mixed_precision=None if cfg.mixed_precision == "no" else cfg.mixed_precision
    )
    device = accelerator.device

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2,
    ).to(device)

    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    start_epoch = 1
    best_dice = -1.0
    if ckpt is not None:
        try:
            model.load_state_dict(ckpt["model"])
        except Exception as e:
            raise RuntimeError(f"Failed to load model state from resume checkpoint: {cfg.resume_from}") from e
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_dice = float(ckpt.get("best_dice", -1.0))

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    if ckpt is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            _move_optimizer_state_to_device(optimizer, accelerator.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load optimizer state from resume checkpoint: {cfg.resume_from}") from e

    mlflow = None
    mlflow_active = False
    mlflow_run_id = ""
    if cfg.use_mlflow and accelerator.is_main_process:
        try:
            import mlflow as _mlflow
        except Exception:
            raise RuntimeError("MLflow not installed. Please: pip install mlflow")
        mlflow = _mlflow
        if cfg.mlflow_tracking_uri:
            mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        if cfg.mlflow_experiment:
            mlflow.set_experiment(cfg.mlflow_experiment)
        run_name = cfg.mlflow_run_name or _default_run_name(cfg)
        resume_same_run = bool(cfg.mlflow_run_id)
        if resume_same_run:
            mlflow.start_run(run_id=cfg.mlflow_run_id)
        else:
            mlflow.start_run(run_name=run_name)
        mlflow_run_id = mlflow.active_run().info.run_id
        # Avoid overwriting tags on an existing run_id; only add resume-related tags below.
        if not resume_same_run:
            mlflow.set_tag("model_name", "unet")
            mlflow.set_tag("backbone", "none")
            mlflow.set_tag("cfg_hash", _cfg_hash(cfg))
            mlflow.set_tag("debug", str(cfg.debug))
            git = _git_info(os.getcwd())
            if git.get("sha"):
                mlflow.set_tag("git_sha", git["sha"])
            if git.get("branch"):
                mlflow.set_tag("git_branch", git["branch"])
            if "dirty" in git:
                mlflow.set_tag("git_dirty", str(bool(git["dirty"])))
        # Params are immutable in MLflow. When resuming into the *same* run_id we must
        # not re-log params (or change values), otherwise the tracking store errors.
        if not resume_same_run:
            for k, v in cfg.__dict__.items():
                if k in {"resume_from", "mlflow_run_id"}:
                    continue
                mlflow.log_param(k, v)
        if ckpt is not None:
            mlflow.set_tag("resume", "true")
            mlflow.set_tag("resume_from", str(cfg.resume_from))
            mlflow.set_tag("resumed_at", datetime.now().strftime("%Y%m%d-%H%M%S"))
        mlflow_active = True

    writer = None
    if cfg.use_tensorboard and accelerator.is_main_process:
        os.makedirs(cfg.tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=cfg.tb_dir)

    if start_epoch > cfg.epochs:
        if accelerator.is_main_process:
            print(f"[WARN] resume epoch ({start_epoch}) > cfg.epochs ({cfg.epochs}). Nothing to do.")
        if mlflow_active:
            mlflow.end_run()
        return

    best_path, last_path = _checkpoint_paths(cfg)

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0

        for step, (x, y, _) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)  # (B,1,D,H,W)
            y = y.to(device, non_blocking=True)  # (B,3,D,H,W)

            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            accelerator.backward(loss)
            optimizer.step()

            run_loss += float(loss.item())
            if writer and (step % cfg.log_steps == 0):
                global_step = (epoch - 1) * max(1, len(train_loader)) + step
                writer.add_scalar("train/loss", float(loss.item()), global_step)

        avg_loss = run_loss / max(1, len(train_loader))

        # validation
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with accelerator.autocast():
                    logits = model(x)
                    prob = torch.sigmoid(logits)
                    pred = (prob > 0.5).float()
                dice_metric(y_pred=pred, y=y)

            dice = float(dice_metric.aggregate().item())

        dt = time.time() - t0
        if writer:
            writer.add_scalar("train/epoch_loss", float(avg_loss), epoch)
            writer.add_scalar("val/dice", float(dice), epoch)
            writer.add_scalar("train/epoch_time_sec", float(dt), epoch)
        if mlflow_active:
            mlflow.log_metric("train/epoch_loss", float(avg_loss), step=epoch)
            mlflow.log_metric("val/dice", float(dice), step=epoch)
            mlflow.log_metric("train/epoch_time_sec", float(dt), step=epoch)

        print(f"Epoch {epoch:03d}/{cfg.epochs} | loss={avg_loss:.4f} | val_dice={dice:.4f} | {dt:.1f}s")

        if dice > best_dice and accelerator.is_main_process:
            best_dice = dice
            unwrapped = accelerator.unwrap_model(model)
            torch.save(
                {
                    "model": unwrapped.state_dict(),
                    "optimizer": _inner_optimizer(optimizer).state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "cfg": cfg.__dict__,
                    "mlflow_run_id": mlflow_run_id,
                },
                best_path,
            )
            print(f"  Saved best: {best_path} (dice={best_dice:.4f})")
            if mlflow_active:
                mlflow.log_metric("best/val_dice", float(best_dice), step=epoch)
                mlflow.log_artifact(best_path, artifact_path="checkpoints")

        # Save "last" checkpoint for true resume.
        if accelerator.is_main_process and cfg.save_last_every > 0 and (epoch % cfg.save_last_every == 0):
            unwrapped = accelerator.unwrap_model(model)
            torch.save(
                {
                    "model": unwrapped.state_dict(),
                    "optimizer": _inner_optimizer(optimizer).state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "cfg": cfg.__dict__,
                    "mlflow_run_id": mlflow_run_id,
                },
                last_path,
            )

    if writer:
        writer.close()

    if accelerator.is_main_process:
        print(f"Done. Best val dice: {best_dice:.4f}")
    if mlflow_active:
        mlflow.end_run()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./input/uw-madison-gi-tract-image-segmentation")
    p.add_argument("--out", type=str, default="./checkpoints/best.pt")
    p.add_argument("--run_dir", type=str, default="")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--patch_d", type=int, default=80)
    p.add_argument("--patch_h", type=int, default=224)
    p.add_argument("--patch_w", type=int, default=224)

    p.add_argument("--samples_per_volume", type=int, default=12)
    p.add_argument("--cache_in_ram", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--verbose_shape_fix", action="store_true")
    p.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    p.add_argument("--use_tensorboard", action="store_true")
    p.add_argument("--tb_dir", type=str, default="./outputs/train_run/tb")
    p.add_argument("--log_steps", type=int, default=50)
    args = p.parse_args()

    cfg = TrainCfg(
        data_root=args.data_root,
        out=args.out,
        run_dir=args.run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patch_d=args.patch_d,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        samples_per_volume=args.samples_per_volume,
        cache_in_ram=args.cache_in_ram,
        num_workers=args.num_workers,
        seed=args.seed,
        val_ratio=args.val_ratio,
        verbose_shape_fix=args.verbose_shape_fix,
        mixed_precision=args.mixed_precision,
        use_tensorboard=args.use_tensorboard,
        tb_dir=args.tb_dir,
        log_steps=args.log_steps,
    )
    train(cfg)
