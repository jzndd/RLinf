import json

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.runners.offline_embodied_grpo_runner import OfflineEmbodiedGRPORunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_offline_openpi_grpo_worker import OfflineOpenPIGRPOActor

mp.set_start_method("spawn", force=True)


def materialize_offline_dimensions(cfg):
    bootstrap_cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, bootstrap_cluster)
    actor_world_size = component_placement.get_world_size("actor")
    prompt_batch_size = int(cfg.data.prompt_batch_size_per_gpu)
    group_size = int(cfg.algorithm.group_size)
    action_chunk = int(cfg.actor.model.num_action_chunks)
    rollout_batch_size = prompt_batch_size * group_size * actor_world_size

    with open_dict(cfg):
        cfg.actor.global_batch_size = rollout_batch_size
        cfg.env.train.total_num_envs = rollout_batch_size
        cfg.env.train.group_size = group_size
        cfg.env.train.max_steps_per_rollout_epoch = action_chunk
        cfg.env.train.max_episode_steps = action_chunk
        cfg.env.eval.total_num_envs = actor_world_size
        cfg.env.eval.group_size = 1
        cfg.env.eval.max_steps_per_rollout_epoch = action_chunk
        cfg.env.eval.max_episode_steps = action_chunk
        cfg.rollout.pipeline_stage_num = 1
        cfg.rollout.model.model_path = cfg.actor.model.model_path
        cfg.rollout.model.precision = cfg.actor.model.precision

    return cfg


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_130_offline_grpo_openpi_pi05",
)
def main(cfg) -> None:
    cfg = materialize_offline_dimensions(cfg)
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_placement = component_placement.get_strategy("actor")
    actor_group = OfflineOpenPIGRPOActor.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=actor_placement,
    )

    runner = OfflineEmbodiedGRPORunner(
        cfg=cfg,
        actor=actor_group,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
