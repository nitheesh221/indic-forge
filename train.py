#!/usr/bin/env python3
"""IndicTrans2 fine-tuning.

One GPU:
    python train.py --gpu_ids 0 --model_hf_id ... --train_mode lora ...

Several GPUs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --model_hf_id ...
"""

from indictrans2_finetune.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
