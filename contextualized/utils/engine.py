# contextualized/utils/engine.py
import os, torch
from typing import Tuple, Union

def _under_torchrun() -> bool:
    e = os.environ
    return any(k in e for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))

def _visible_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0

def pick_engine(
    accelerator: str | None = None,
    devices: Union[int, str, list[int]] | None = None,
    strategy: str | None = None,
    prefer_spawn: bool = True,
) -> Tuple[str, Union[int, str, list[int]], Union[str, object]]:
    """
    CPU / 1-GPU / multi-GPU auto-selection WITHOUT requiring torchrun.
    - If user passes any of (accelerator/devices/strategy), we respect them.
    - Else:
        GPUs == 0 => cpu, devices='auto'
        GPUs == 1 => gpu, devices=1
        GPUs  > 1 =>
           - if launched with torchrun => gpu, devices=1, strategy='ddp'
           - else                       => gpu, devices=<ngpu>, strategy='ddp_spawn'
    """
    if accelerator is not None or devices is not None or strategy is not None:
        return accelerator or "auto", devices or "auto", strategy or "auto"

    ngpu = _visible_gpus()
    if ngpu == 0:
        return "cpu", "auto", "auto"

    if ngpu == 1:
        return "gpu", 1, "auto"

    if _under_torchrun():
        return "gpu", 1, "ddp"         # one proc per GPU (torchrun sets ranks)
    return "gpu", ngpu, ("ddp_spawn" if prefer_spawn else "auto")
