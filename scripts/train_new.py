# Copyright (c) 2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import yaml
from isaaclab_eureka.eureka import Eureka

HERE = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(HERE, "eureka_config.yaml")
TASK_CFG_DIR = os.path.join(HERE, "task_configs")

# The only keys we expect to change per run (run config should win on these)
RUN_KEYS = {
    "task",
    "num_parallel_runs",
    "mode",
    "max_eureka_iterations",
    "env_seed",
    "gpt_model",
    "temperature",
    "use_cache",
}

def load_yaml(path):
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def deep_update(dst, src):
    """Recursively update dict dst with src (in place) and return dst."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def overlay_keys(dst, src, keys):
    """Update only selected keys from src into dst."""
    for k in keys:
        if k in src:
            dst[k] = src[k]
    return dst

def args_to_dict(args_ns):
    d = vars(args_ns).copy()
    # argparse may set lists/strings; normalize if needed
    return d

def dict_to_namespace(d):
    return argparse.Namespace(**d)

def build_config(args_cli):
    """
    Merge order:
      1) start from CLI defaults
      2) overlay TASK config (stable, task-specific)
      3) overlay RUN config but ONLY for RUN_KEYS
    """
    cli_dict = args_to_dict(args_cli)
    run_cfg = load_yaml(CONFIG_PATH)

    # Task name comes from run config if present, else CLI
    task_name = run_cfg.get("task", cli_dict.get("task"))
    if not task_name:
        raise ValueError("No task specified. Set 'task' in eureka_config.yaml or via --task.")

    # Load task config if exists
    task_cfg_path = os.path.join(TASK_CFG_DIR, f"{task_name}.yaml")
    task_cfg = load_yaml(task_cfg_path)

    # 1) CLI defaults → 2) task overrides → 3) run overrides (only RUN_KEYS)
    merged = {}
    deep_update(merged, cli_dict)
    deep_update(merged, task_cfg)
    overlay_keys(merged, cli_dict, RUN_KEYS)

    # Normalize parameters_to_tune to a list
    ptt = merged.get("parameters_to_tune")
    if isinstance(ptt, str):
        merged["parameters_to_tune"] = [ptt]

    return dict_to_namespace(merged)

def main(args_cli):
    cfg = build_config(args_cli)

    eureka = Eureka(
        task=cfg.task,
        rl_library=cfg.rl_library,
        num_parallel_runs=cfg.num_parallel_runs,
        device=cfg.device,
        env_seed=cfg.env_seed,
        max_training_iterations=cfg.max_training_iterations,
        feedback_subsampling=cfg.feedback_subsampling,
        temperature=cfg.temperature,
        gpt_model=cfg.gpt_model,
        env_type=cfg.env_type,
        eureka_task=cfg.eureka_task,
        parameters_to_tune=cfg.parameters_to_tune,
        num_envs=cfg.num_envs,
        resume=getattr(cfg, "resume", None),
        use_cache=getattr(cfg, "use_cache", True),
        video=getattr(cfg, "video", False),
        random_start=getattr(cfg, "random_start", False),
        mode=getattr(cfg, "mode", "eureka"),
    )
    eureka.run(max_eureka_iterations=cfg.max_eureka_iterations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent with Eureka.")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-Direct-v0")
    parser.add_argument("--num_parallel_runs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--env_seed", type=int, default=42)
    parser.add_argument("--max_eureka_iterations", type=int, default=5)
    parser.add_argument("--max_training_iterations", type=int, default=100)
    parser.add_argument("--feedback_subsampling", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gpt_model", type=str, default="gpt-4")
    parser.add_argument("--rl_library", type=str, default="rsl_rl", choices=["rsl_rl", "rl_games", "skrl"])
    parser.add_argument("--env_type", type=str, default="")
    parser.add_argument("--eureka_task", type=str, default="reward_weight_tuning")
    parser.add_argument("--parameters_to_tune", nargs="+", default=[])
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--random_start", action="store_true")
    parser.add_argument("--mode", type=str, default="eureka")
    args_cli = parser.parse_args()
    main(args_cli)
