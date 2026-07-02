import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.runner_utils import check_progress

if TYPE_CHECKING:
    from rlinf.workers.actor.fsdp_offline_openpi_grpo_worker import (
        OfflineOpenPIGRPOActor,
    )


logger = logging.getLogger(__name__)


class OfflineEmbodiedGRPORunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: "OfflineOpenPIGRPOActor",
        run_timer: Optional[ScopedTimer] = None,
    ) -> None:
        self.cfg = cfg
        self.actor = actor
        self.run_timer = run_timer
        self.global_step = 0

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.num_steps_per_epoch = 1
        self.max_steps = self.cfg.runner.max_epochs
        if self.cfg.runner.max_steps >= 0:
            self.max_steps = min(self.max_steps, self.cfg.runner.max_steps)

    def _aggregate_numeric_metrics(self, metrics_list: list[dict] | dict | None) -> dict:
        if metrics_list is None:
            return {}
        if isinstance(metrics_list, dict):
            metrics_list = [metrics_list]

        merged_metrics = defaultdict(list)
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key].append(value)

        return {
            key: sum(values) / len(values)
            for key, values in merged_metrics.items()
            if values
        }

    def init_workers(self) -> None:
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.resume_dir
        if resume_dir is None:
            return

        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def run(self) -> None:
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            ncols=160,
        )

        for step in range(start_step, self.max_steps):
            self.actor.set_global_step(self.global_step)
            dataset_offline_eval_metrics = {}

            with self.timer("step"):
                rollout_handle: Handle = self.actor.prepare_rollout_batch()
                rollout_metrics = self._aggregate_numeric_metrics(rollout_handle.wait())

                adv_handle: Handle = self.actor.compute_advantages_and_returns()
                adv_metrics = self._aggregate_numeric_metrics(adv_handle.wait())

                training_handle: Handle = self.actor.run_training()
                training_metrics = self._aggregate_numeric_metrics(training_handle.wait())

                self.global_step += 1

                _, save_model, _ = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )
                if save_model:
                    self._save_checkpoint()
                    if bool(self.cfg.data.get("if_offline_eval", False)):
                        eval_handle: Handle = self.actor.run_dataset_offline_eval()
                        dataset_offline_eval_metrics = (
                            self._aggregate_numeric_metrics(eval_handle.wait())
                        )

            time_metrics = self.timer.consume_durations()
            time_metrics["prepare_rollout"] = rollout_handle.consume_duration()
            time_metrics["compute_advantages"] = adv_handle.consume_duration()
            time_metrics["training"] = training_handle.consume_duration()

            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            rollout_metrics = {f"rollout/{k}": v for k, v in rollout_metrics.items()}
            adv_metrics = {f"adv/{k}": v for k, v in adv_metrics.items()}
            training_metrics = {f"train/{k}": v for k, v in training_metrics.items()}
            dataset_offline_eval_metrics = {
                f"dataset_offline_eval/{k}": v
                for k, v in dataset_offline_eval_metrics.items()
            }

            self.metric_logger.log(time_metrics, step)
            self.metric_logger.log(rollout_metrics, step)
            self.metric_logger.log(adv_metrics, step)
            self.metric_logger.log(training_metrics, step)
            if dataset_offline_eval_metrics:
                self.metric_logger.log(dataset_offline_eval_metrics, step)

            logging_metrics = {}
            logging_metrics.update(time_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(adv_metrics)
            logging_metrics.update(training_metrics)
            logging_metrics.update(dataset_offline_eval_metrics)

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)

        self.metric_logger.finish()

    def _save_checkpoint(self) -> None:
        checkpoint_root = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
        )
        base_output_dir = os.path.join(
            checkpoint_root,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()
