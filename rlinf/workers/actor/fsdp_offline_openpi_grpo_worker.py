import json
import logging
import os
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler

from rlinf.data.datasets.openpi_grpo import (
    OpenPIGRPODataset,
    openpi_grpo_collate_fn,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


logger = logging.getLogger(__name__)


def repeat_obs_for_grpo(
    obs: dict[str, Any],
    group_size: int,
) -> dict[str, Any]:
    repeated_obs: dict[str, Any] = {}
    for key, value in obs.items():
        if value is None:
            repeated_obs[key] = None
            continue
        if key == "task_descriptions":
            repeated_obs[key] = [text for text in value for _ in range(group_size)]
            continue
        repeated_obs[key] = value.repeat_interleave(group_size, dim=0)
    return repeated_obs


def discretize_gripper_actions(gripper_actions: torch.Tensor) -> torch.Tensor:
    return torch.where(
        gripper_actions >= 0,
        torch.ones_like(gripper_actions),
        -torch.ones_like(gripper_actions),
    )


def compute_mse_rewards(
    sampled_actions: torch.Tensor,
    gt_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert sampled_actions.shape == gt_actions.shape, (
        f"sampled_actions shape {sampled_actions.shape} does not match "
        f"gt_actions shape {gt_actions.shape}"
    )
    mse = torch.mean((sampled_actions - gt_actions) ** 2, dim=-1)
    rewards = -mse
    return rewards, mse


def compute_l1_rewards(
    sampled_actions: torch.Tensor,
    gt_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert sampled_actions.shape == gt_actions.shape, (
        f"sampled_actions shape {sampled_actions.shape} does not match "
        f"gt_actions shape {gt_actions.shape}"
    )
    assert sampled_actions.shape[-1] >= 2, (
        "compute_l1_rewards expects action_dim >= 2 so the last dimension "
        "can be reserved for gripper reward."
    )

    action_l1 = torch.mean(
        torch.abs(sampled_actions[..., :-1] - gt_actions[..., :-1]),
        dim=-1,
    )
    sampled_gripper = discretize_gripper_actions(sampled_actions[..., -1])
    gt_gripper = discretize_gripper_actions(gt_actions[..., -1])
    gripper_match = (sampled_gripper == gt_gripper).to(sampled_actions.dtype)
    rewards = 0.8 * torch.exp(-5.0 * action_l1) + 0.2 * gripper_match
    return rewards, action_l1, gripper_match


def build_offline_rollout_batch(
    *,
    prev_logprobs: torch.Tensor,
    forward_inputs: dict[str, torch.Tensor],
    rewards: torch.Tensor,
    action_chunk: int,
) -> dict[str, Any]:
    batch_size = rewards.shape[0]
    loss_mask = torch.ones((1, batch_size, 1), dtype=torch.bool)
    loss_mask_sum = torch.full((1, batch_size, 1), action_chunk, dtype=torch.long)
    dones = torch.ones((2, batch_size, action_chunk), dtype=torch.bool)
    dones[0] = False

    return {
        "prev_logprobs": prev_logprobs.detach().cpu().unsqueeze(0),
        "forward_inputs": {
            key: value.detach().cpu().unsqueeze(0)
            for key, value in forward_inputs.items()
        },
        "rewards": rewards.detach().cpu().unsqueeze(0),
        "dones": dones,
        "loss_mask": loss_mask,
        "loss_mask_sum": loss_mask_sum,
    }


class OfflineOpenPIGRPOActor(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.prompt_batch_size = int(cfg.data.prompt_batch_size_per_gpu)
        self._data_epoch = 0
        self._data_iter_offset = 0

    def init_worker(self) -> None:
        self.setup_model_and_optimizer()
        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._build_offline_grpo_dataloader()

    def _build_offline_grpo_dataloader(self) -> None:
        dataset = OpenPIGRPODataset(self.cfg.data.train_data_paths, self.cfg)

        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=self.cfg.data.shuffle,
                seed=self.cfg.data.seed,
                drop_last=True,
            )

        self.data_loader = DataLoader(
            dataset,
            batch_size=self.prompt_batch_size,
            sampler=sampler,
            shuffle=sampler is None and self.cfg.data.shuffle,
            num_workers=self.cfg.data.num_workers,
            drop_last=True,
            collate_fn=openpi_grpo_collate_fn,
            pin_memory=True,
        )
        self.data_iter = iter(self.data_loader)
        self.num_data_batches = len(self.data_loader)
        assert self.num_data_batches > 0, "Offline GRPO dataloader is empty."

    def _next_batch(self) -> dict[str, Any]:
        if self._data_iter_offset >= self.num_data_batches:
            self._data_epoch += 1
            if isinstance(self.data_loader.sampler, DistributedSampler):
                self.data_loader.sampler.set_epoch(self._data_epoch)
            self.data_iter = iter(self.data_loader)
            self._data_iter_offset = 0

        batch = next(self.data_iter)
        self._data_iter_offset += 1
        return batch

    def _save_data_state(self, save_path: str) -> None:
        state = {
            "data_epoch": self._data_epoch,
            "data_iter_offset": self._data_iter_offset,
        }
        with open(os.path.join(save_path, "data_state.json"), "w") as f:
            json.dump(state, f)

    def _load_data_state(self, load_path: str) -> None:
        data_state_path = os.path.join(load_path, "data_state.json")
        if not os.path.exists(data_state_path):
            return

        with open(data_state_path, "r") as f:
            state = json.load(f)

        self._data_epoch = int(state["data_epoch"])
        self._data_iter_offset = int(state["data_iter_offset"])

        if isinstance(self.data_loader.sampler, DistributedSampler):
            self.data_loader.sampler.set_epoch(self._data_epoch)

        self.data_iter = iter(self.data_loader)
        for _ in range(self._data_iter_offset):
            next(self.data_iter)

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)
        if self._rank == 0:
            self._save_data_state(save_path)

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)
        self._load_data_state(load_path)

    @Worker.timer("prepare_rollout_batch")
    def prepare_rollout_batch(self) -> dict[str, float]:
        batch = self._next_batch()
        obs = batch["obs"]
        gt_actions = batch["actions"].to(torch.float32)
        group_size = int(self.cfg.algorithm.group_size)
        action_chunk = int(self.cfg.actor.model.num_action_chunks)

        repeated_obs = repeat_obs_for_grpo(obs, group_size)
        repeated_gt_actions = gt_actions.repeat_interleave(group_size, dim=0)

        with torch.no_grad():
            self.model.eval()
            sampled_actions, result = self.model(
                forward_type=ForwardType.ACTION_SAMPLING,
                env_obs=repeated_obs,
                mode="train",
                compute_values=False,
            )

        sampled_actions = sampled_actions.to(torch.float32)
        prev_logprobs = result["prev_logprobs"].to(torch.float32)

        reward_mode = str(self.cfg.algorithm.offline_reward_fn).lower()
        if reward_mode == "mse":
            rewards, mse = compute_mse_rewards(sampled_actions, repeated_gt_actions)
            l1 = torch.zeros_like(rewards)
            gripper_match = torch.zeros_like(rewards)
        elif reward_mode == "l1":
            rewards, l1, gripper_match = compute_l1_rewards(
                sampled_actions,
                repeated_gt_actions,
            )
            mse = torch.zeros_like(rewards)
        else:
            raise ValueError(
                "Unsupported offline_reward_fn: "
                f"{self.cfg.algorithm.offline_reward_fn}. Expected one of ['mse', 'l1']."
            )

        rollout_epoch = int(self.cfg.algorithm.rollout_epoch)
        assert rollout_epoch == 1, (
            "OfflineOpenPIGRPOActor currently materializes exactly one rollout batch "
            f"per step, but got algorithm.rollout_epoch={rollout_epoch}."
        )

        self.rollout_batch = build_offline_rollout_batch(
            prev_logprobs=prev_logprobs,
            forward_inputs=result["forward_inputs"],
            rewards=rewards,
            action_chunk=action_chunk,
        )
        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

        group_scores = rewards.sum(dim=-1).view(-1, group_size)
        metrics = {
            "prompt_batch_size": float(gt_actions.shape[0]),
            "rollout_batch_size": float(repeated_gt_actions.shape[0]),
            "reward_mean": rewards.mean().item(),
            "reward_min": rewards.min().item(),
            "reward_max": rewards.max().item(),
            "mse_mean": mse.mean().item(),
            "l1_mean": l1.mean().item(),
            "gripper_match_rate": gripper_match.mean().item(),
            "reward_mode_is_mse": 1.0 if reward_mode == "mse" else 0.0,
            "reward_mode_is_l1": 1.0 if reward_mode == "l1" else 0.0,
            "group_score_std": group_scores.std(dim=-1).mean().item(),
        }
        return all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)

    @Worker.timer("compute_advantages_and_returns")
    def compute_advantages_and_returns(self) -> dict[str, float]:
        return super().compute_advantages_and_returns()
