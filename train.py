#!/usr/bin/env python3
"""IndicTrans2 fine-tuning.

One GPU:
    python train.py --gpu_ids 0 --model_hf_id ... --train_mode lora ...

Several GPUs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --model_hf_id ...
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from indictrans2_finetune.config import pin_visible_gpus

pin_visible_gpus()

from indictrans2_finetune.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
