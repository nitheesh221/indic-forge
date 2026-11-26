"""Start multi-GPU inference before torch is imported.

``CUDA_VISIBLE_DEVICES`` only takes effect if it is set before torch loads.
This module stays free of torch imports so the translate command can relaunch
first.
"""

from __future__ import annotations

import os
import sys


def parse_gpu_ids(gpu_ids: str) -> list[str]:
    ids = [item.strip() for item in gpu_ids.split(",") if item.strip()]
    if not ids or any(not item.isdigit() for item in ids):
        raise SystemExit("--gpu-ids must be a comma-separated list of GPU indices, for example 0,1.")
    return ids


def strip_gpu_id_args(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip = False
    for item in argv:
        if skip:
            skip = False
            continue
        if item in {"--gpu-ids", "--gpu_ids"}:
            skip = True
            continue
        if item.startswith("--gpu-ids=") or item.startswith("--gpu_ids="):
            continue
        cleaned.append(item)
    return cleaned


def translate_launch_command(argv: list[str], nproc: int) -> list[str]:
    """torchrun command that re-enters this CLI once per GPU."""
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc),
        "--module",
        "indictrans2_finetune.cli",
        *strip_gpu_id_args(argv),
    ]


def relaunch_translate_if_needed(gpu_ids: str | None) -> None:
    """Pin one GPU, or replace this process with one worker per GPU.

    A single id stays in this process. Two or more ids relaunch under torchrun.
    The child processes do not receive ``--gpu-ids``, because the visible
    devices were already selected here.
    """
    distributed = os.environ.get("RANK") is not None or os.environ.get("LOCAL_RANK") is not None
    if distributed and gpu_ids:
        raise SystemExit("Do not pass --gpu-ids when launching with torchrun.")
    if distributed or not gpu_ids:
        return
    ids = parse_gpu_ids(gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    if len(ids) == 1:
        return
    argv = strip_gpu_id_args(sys.argv[1:])
    cmd = translate_launch_command(argv, len(ids))
    os.execvpe(cmd[0], cmd, os.environ)
