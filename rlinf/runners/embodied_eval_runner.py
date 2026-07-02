# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing
from pathlib import Path

import torch

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    def _dump_per_episode_flags(self, eval_metrics_list):
        if not eval_metrics_list:
            return

        reset_state_tensors = [
            metrics["reset_state_id"]
            for metrics in eval_metrics_list
            if "reset_state_id" in metrics
        ]
        if not reset_state_tensors:
            return

        success_key = None
        for candidate in ("success_once", ):
            if any(candidate in metrics for metrics in eval_metrics_list):
                success_key = candidate
                break
        if success_key is None:
            return

        success_tensors = [
            metrics[success_key] for metrics in eval_metrics_list if success_key in metrics
        ]
        if not success_tensors:
            return

        reset_state_ids = (
            torch.concat(reset_state_tensors)
            .detach()
            .cpu()
            .to(dtype=torch.int64)
            .tolist()
        )
        success_flags = (
            torch.concat(success_tensors)
            .detach()
            .cpu()
            .to(dtype=torch.int64)
            .tolist()
        )

        row_count = min(len(reset_state_ids), len(success_flags))
        if row_count == 0:
            return

        log_dir = Path(self.cfg.runner.logger.log_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        output_path = log_dir / "per_episode_flags.csv"
        with output_path.open("w", encoding="utf-8") as f:
            f.write("episode_idx,reset_state_id,success_flag\n")
            for idx in range(row_count):
                f.write(f"{idx},{reset_state_ids[idx]},{success_flags[idx]}\n")
        self.logger.info(f"Per-episode flags saved to: {output_path}")

    def init_workers(self):
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        rollout_handle.wait()
        env_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        self._dump_per_episode_flags(eval_metrics_list)
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        eval_metrics = self.evaluate()
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)

        self.metric_logger.finish()
