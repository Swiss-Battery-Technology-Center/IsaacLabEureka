#!/usr/bin/env python3
# Copyright (c) 2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import yaml
from isaaclab_eureka.eureka import Eureka

HERE = os.path.dirname(__file__)
DEFAULT_CONFIG_PATH = os.path.join(HERE, "eureka_config.yaml")
TASK_CFG_DIR = os.path.join(HERE, "task_configs")

# Keys that must *not* be overridden by task YAML (wrapper controls these)
PROTECTED_RUN_KEYS = {
    "task",
    "seed",
    "num_parallel_runs",
    "mode",
    "max_eureka_iterations",
    "gpt_model",
    "temperature",
    "use_cache",
    "resume",
    "random_start",
    "video",
    "rl_library",
    "env_type",
    "device",
}

def load_yaml(path: str) -> dict:
    if not (path and os.path.isfile(path)):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def deep_update(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def without_keys(d: dict, keys: set) -> dict:
    """Shallow copy of d without keys in `keys`."""
    return {k: v for k, v in (d or {}).items() if k not in keys}

def build_config(base_cfg_path: str) -> dict:
    # 1) Load wrapper-written base config (run-time knobs live here)
    base = load_yaml(base_cfg_path)
    if not base:
        raise ValueError(f"Config not found or empty: {base_cfg_path}")

    task_name = base.get("task")
    if not task_name:
        raise ValueError("Missing 'task' in eureka_config.yaml (wrapper should set this).")

    # 2) Overlay task-only fields from task YAML (cannot override protected run keys)
    task_cfg_path = os.path.join(TASK_CFG_DIR, f"{task_name}.yaml")
    task_cfg = load_yaml(task_cfg_path)
    merged = dict(base)
    deep_update(merged, without_keys(task_cfg, PROTECTED_RUN_KEYS))

    # Normalize parameters_to_tune to a list
    ptt = merged.get("parameters_to_tune")
    if isinstance(ptt, str):
        merged["parameters_to_tune"] = [ptt]

    return merged

def main():
    parser = argparse.ArgumentParser(
        description="Train an RL agent with Eureka. "
                    "Wrapper should write all run keys into eureka_config.yaml."
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                        help="Path to eureka_config.yaml (defaults to scripts/eureka_config.yaml).")
    args = parser.parse_args()

    cfg = build_config(args.config)

    # Construct Eureka with the merged config.
    # NOTE: Eureka now expects `seed`, not `env_seed`.
    eureka = Eureka(
        task=cfg["task"],
        rl_library=cfg.get("rl_library", "rsl_rl"),
        num_parallel_runs=cfg.get("num_parallel_runs", 1),
        device=cfg.get("device", "cuda"),
        env_seed=cfg.get("seed", 42),
        max_training_iterations=cfg.get("max_training_iterations", 100),
        feedback_subsampling=cfg.get("feedback_subsampling", 10),
        temperature=cfg.get("temperature", 1.0),
        gpt_model=cfg.get("gpt_model", "gpt-4"),
        env_type=cfg.get("env_type", ""),
        eureka_task=cfg.get("eureka_task", "reward_weight_tuning"),
        parameters_to_tune=cfg.get("parameters_to_tune", []),
        num_envs=cfg.get("num_envs", 256),
        resume=cfg.get("resume", None),
        use_cache=cfg.get("use_cache", True),
        video=cfg.get("video", False),
        random_start=cfg.get("random_start", False),
        mode=cfg.get("mode", "eureka"),
    )

    eureka.run(max_eureka_iterations=cfg.get("max_eureka_iterations", 5))

if __name__ == "__main__":
    main()
