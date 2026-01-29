from __future__ import annotations

from typing import Optional, Tuple

import torch


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    train_loader,
    accelerator,
    device: torch.device,
    epoch: int,
    log_steps: int = 50,
    writer=None,
    global_step_base: int = 0,
) -> Tuple[float, int]:
    model.train()
    run_loss = 0.0

    for step, (x, y, _) in enumerate(train_loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            logits = model(x)
            loss = loss_fn(logits, y)
        accelerator.backward(loss)
        optimizer.step()

        run_loss += float(loss.item())
        if writer and log_steps > 0 and (step % log_steps == 0):
            writer.add_scalar("train/loss", float(loss.item()), global_step_base + step)

    avg_loss = run_loss / max(1, len(train_loader))
    return avg_loss, global_step_base + max(1, len(train_loader))


@torch.no_grad()
def validate_dice(
    model: torch.nn.Module,
    val_loader,
    dice_metric,
    accelerator,
    device: torch.device,
    threshold: float = 0.5,
) -> float:
    model.eval()
    dice_metric.reset()

    for x, y, _ in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with accelerator.autocast():
            logits = model(x)
            prob = torch.sigmoid(logits)
            pred = (prob > threshold).float()
        dice_metric(y_pred=pred, y=y)

    return float(dice_metric.aggregate().item())

