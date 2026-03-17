from __future__ import annotations

import torch.nn as nn
from monai.losses import DiceLoss
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss


class DiceBceMultilabelLoss(_Loss):
    def __init__(
        self,
        w_dice: float = 0.5,
        w_bce: float = 0.5,
        reduction: str | LossReduction = LossReduction.MEAN,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        self.w_dice = float(w_dice)
        self.w_bce = float(w_bce)
        self.dice_loss = DiceLoss(
            sigmoid=True,
            smooth_nr=0.01,
            smooth_dr=0.01,
            include_background=True,
            batch=True,
            squared_pred=True,
        )
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, label):
        return self.dice_loss(pred, label) * self.w_dice + self.bce_loss(pred, label) * self.w_bce
