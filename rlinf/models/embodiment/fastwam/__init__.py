# Copyright 2026 The RLinf Authors.

from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from rlinf.models.embodiment.fastwam.fastwam_policy import FastWAMPolicy
from rlinf.scheduler import Worker


def get_model(cfg: DictConfig, torch_dtype=None):
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.runtime import create_fastwam

    if torch_dtype is None:
        torch_dtype = torch.float32

    model_path = Path(str(cfg.get("model_path"))).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"FastWAM model_path does not exist: {model_path}")

    dataset_stats_path = Path(str(cfg.get("dataset_stats_path"))).expanduser()
    if not dataset_stats_path.exists():
        raise FileNotFoundError(
            f"FastWAM dataset_stats_path does not exist: {dataset_stats_path}"
        )

    device = Worker.torch_device_type if Worker.torch_platform.is_available() else "cpu"

    processor = instantiate(cfg.processor).eval()
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor.set_normalizer_from_stats(dataset_stats)

    model = create_fastwam(
        model_id=cfg.model_id,
        tokenizer_model_id=cfg.tokenizer_model_id,
        video_dit_config=cfg.video_dit_config,
        tokenizer_max_len=int(cfg.get("tokenizer_max_len", 128)),
        load_text_encoder=bool(cfg.get("load_text_encoder", True)),
        proprio_dim=int(cfg.get("proprio_dim", 8)),
        action_dit_config=cfg.action_dit_config,
        action_dit_pretrained_path=cfg.action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(cfg.get("skip_dit_load_from_pretrain", False)),
        video_scheduler=cfg.video_scheduler,
        action_scheduler=cfg.action_scheduler,
        loss=cfg.loss,
        mot_checkpoint_mixed_attn=bool(cfg.get("mot_checkpoint_mixed_attn", True)),
        redirect_common_files=bool(cfg.get("redirect_common_files", True)),
        model_dtype=torch_dtype,
        device=device,
    )
    model.load_checkpoint(str(model_path))

    return FastWAMPolicy(
        cfg=cfg,
        model=model,
        processor=processor,
    )


__all__ = ["get_model", "FastWAMPolicy"]
