import os
from dataclasses import fields
from typing import Any, Dict, Optional, Tuple

import torch


def load_resume_checkpoint(resume_from: str) -> Optional[dict]:
    if not resume_from:
        return None
    if not os.path.exists(resume_from):
        raise FileNotFoundError(f"resume_from not found: {resume_from}")
    return torch.load(resume_from, map_location="cpu")