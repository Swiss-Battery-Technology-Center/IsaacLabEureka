# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train an RL agent with Isaac Lab Eureka."""

import argparse
import os
from isaaclab_eureka.eureka import Eureka
import yaml
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "eureka_config.yaml")  # Ensure path is correct

def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def merge_args_with_yaml(args, yaml_config):
    """Merge command-line arguments with YAML config (YAML overrides CLI)."""
    merged_config = vars(args)  # Convert argparse Namespace to dictionary
    for key, value in yaml_config.items():
        merged_config[key] = value  # YAML overrides CLI arguments
    return argparse.Namespace(**merged_config)  # Convert back to Namespace

def main(args_cli):
    eureka = Eureka(
        task=args_cli.task,
        rl_library=args_cli.rl_library,
        num_parallel_runs=args_cli.num_parallel_runs,
        device=args_cli.device,
        env_seed=args_cli.env_seed,
        max_training_iterations=args_cli.max_training_iterations,
        feedback_subsampling=args_cli.feedback_subsampling,
        temperature=args_cli.temperature,
        gpt_model=args_cli.gpt_model,
        env_type=args_cli.env_type,
        task_type=args_cli.task_type,
        parameters_to_tune=args_cli.parameters_to_tune,
        warmstart = args_cli.warmstart,
        num_envs = args_cli.num_envs,
    )

    eureka.run(max_eureka_iterations=args_cli.max_eureka_iterations)


if __name__ == "__main__":
    yaml_config = load_yaml_config(CONFIG_PATH)
    parser = argparse.ArgumentParser(description="Train an RL agent with Eureka.")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-Direct-v0", help="Name of the task.")
    parser.add_argument(
        "--num_parallel_runs", type=int, default=1, help="Number of Eureka runs to execute in parallel."
    )
    parser.add_argument("--device", type=str, default="cuda", help="The device to run training on.")
    parser.add_argument("--env_seed", type=int, default=42, help="The random seed to use for the environment.")
    parser.add_argument("--max_eureka_iterations", type=int, default=5, help="The number of Eureka iterations to run.")
    parser.add_argument(
        "--max_training_iterations",
        type=int,
        default=100,
        help="The number of RL training iterations to run for each Eureka iteration.",
    )
    parser.add_argument(
        "--feedback_subsampling",
        type=int,
        default=10,
        help="The subsampling of the metrics given as feedack to the LLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Controls the randomness of the GPT output (0 is deterministic, 1 is highly diverse).",
    )
    parser.add_argument("--gpt_model", type=str, default="gpt-4", help="The GPT model to use.")
    parser.add_argument(
        "--rl_library",
        type=str,
        default="rsl_rl",
        choices=["rsl_rl", "rl_games", "skrl"],
        help="The RL training library to use.",
    )
    parser.add_argument("--env_type", type=str, default="", help="Type of IsaacLab env")
    parser.add_argument("--task_type", type=str, default="reward_weight_tuning", help="Eureka task type.")
    parser.add_argument("--parameters_to_tune", nargs="+", default=[], help="List of parameters to tune (for PPO tuning).")
    args_cli = parser.parse_args()
    
    args_cli = merge_args_with_yaml(args_cli, yaml_config)

    if isinstance(args_cli.parameters_to_tune, str):  # If a single string, convert to list
        args_cli.parameters_to_tune = [args_cli.parameters_to_tune]
        
    # Check parameter validity
    if os.name == "nt" and args_cli.num_parallel_runs > 1:
        print(
            "[WARNING]: Running with num_parallel_runs > 1 is not supported on Windows. Setting num_parallel_runs = 1."
        )
        args_cli.num_parallel_runs = 1

    # Run the main function
    main(args_cli)
