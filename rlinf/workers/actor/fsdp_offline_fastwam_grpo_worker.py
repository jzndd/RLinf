import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.workers.actor.fsdp_offline_openpi_grpo_worker import (
    OfflineOpenPIGRPOActor,
    build_offline_rollout_batch,
    compute_l1_rewards,
    compute_mse_rewards,
    repeat_obs_for_grpo,
)


class OfflineFastWAMGRPOActor(OfflineOpenPIGRPOActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

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

        sampled_actions = torch.as_tensor(sampled_actions, dtype=torch.float32)
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
            "OfflineFastWAMGRPOActor currently materializes exactly one rollout batch "
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
