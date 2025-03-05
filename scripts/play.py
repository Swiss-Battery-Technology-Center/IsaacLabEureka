# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play an RL agent with Isaac Lab Eureka."""

import argparse
import math
import torch
import os
from rl_games.common.player import BasePlayer

from isaaclab_eureka.utils import get_freest_gpu



def main(args_cli):
    """Create the environment for the task."""
    from isaaclab.app import AppLauncher

    # parse args from cmdline
    device = args_cli.device
    task = args_cli.task
    checkpoint = args_cli.checkpoint

    # parse device
    if device == "cuda":
        device_id = get_freest_gpu()
        device = f"cuda:{device_id}"

    # launch app
    app_launcher = AppLauncher(headless=args_cli.headless, device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg: DirectRLEnvCfg = parse_env_cfg(task)
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env = gym.make(task, cfg=env_cfg)

    """Run the inferencing of the task."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    if args_cli.rl_library == "rsl_rl":
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

        agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = device

        # checkpoint path
        log_root_path = os.path.join("logs", "rl_runs", "rsl_rl_eureka", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        try:
            if args_cli.checkpoint:
                # Use absolute path if given
                resume_path = args_cli.checkpoint
                if not os.path.isabs(resume_path):
                    resume_path = os.path.abspath(resume_path)  # Convert relative to absolute
                if not os.path.exists(resume_path):
                    raise FileNotFoundError(f"Provided checkpoint not valid {resume_path}")
            else:
                # Find the most recent model in the directory
                from isaaclab_tasks.utils import get_checkpoint_path
                print("[INFO] No checkpoint provided, using most recent model.")
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

                if resume_path is None or not os.path.exists(resume_path):
                    raise FileNotFoundError(f"Cannot find most recent model at: {resume_path}")

            print(f"[INFO] Using checkpoint: {resume_path}")

            # Load the environment and model
            env = RslRlVecEnvWrapper(env)
            ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            ppo_runner.load(resume_path)

        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            exit(1)  # Exit the script if no checkpoint is found
        except Exception as e:
            print(f"[ERROR] Unexpected error while loading checkpoint: {e}")
            exit(1)
        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # reset environment
        obs, _ = env.get_observations()
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, _, _ = env.step(actions)

    elif args_cli.rl_library == "rl_games":
        from rl_games.common import env_configurations, vecenv
        from rl_games.common.algo_observer import IsaacAlgoObserver
        from rl_games.torch_runner import Runner

        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

        agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
        
        # parse checkpoint path
        log_root_path = os.path.join("logs", "rl_runs", "rl_games_eureka", agent_cfg["params"]["config"]["name"])
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        # find checkpoint
        try:
            # Find checkpoint
            if args_cli.checkpoint is None:
                # Specify directory for logging runs
                from isaaclab_tasks.utils import get_checkpoint_path
                run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
                checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
                checkpoint = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
            else:
                checkpoint = args_cli.checkpoint  # Fixed reference to checkpoint

            # Validate checkpoint path
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"[ERROR] Checkpoint file not found at: {checkpoint}")

            # Update agent configuration
            agent_cfg["params"]["load_checkpoint"] = True
            agent_cfg["params"]["load_path"] = checkpoint

            print(f"[INFO] Using checkpoint: {checkpoint}")

        except FileNotFoundError as e:
            print(e)
            exit(1) 
            
        agent_cfg["params"]["config"]["device"] = device
        agent_cfg["params"]["config"]["device_name"] = device
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        env = RlGamesVecEnvWrapper(env, device, clip_obs, clip_actions)

        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        # set number of actors into agent config
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        # create runner from rl-games
        runner = Runner(IsaacAlgoObserver())
        runner.load(agent_cfg)

        # obtain the agent from the runner
        agent: BasePlayer = runner.create_player()
        agent.restore(checkpoint)
        agent.reset()

        # reset environment
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        # required: enables the flag for batched observations
        _ = agent.get_batch_size(obs, 1)
        # initialize RNN states if used
        if agent.is_rnn:
            agent.init_rnn()
        # simulate environment
        # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
        #   attempt to have complete control over environment stepping. However, this removes other
        #   operations such as masking that is used for multi-agent learning by RL-Games.
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # convert obs to agent format
                obs = agent.obs_to_torch(obs)
                # agent stepping
                actions = agent.get_action(obs, is_deterministic=True)
                # env stepping
                obs, _, dones, _ = env.step(actions)

                # perform operations for terminated episodes
                if len(dones) > 0:
                    # reset rnn state for terminated episodes
                    if agent.is_rnn and agent.states is not None:
                        for s in agent.states:
                            s[:, dones, :] = 0.0
    elif args_cli.rl_library == "skrl":
        import time
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
            
            # specify directory for logging experiments (load checkpoint)
        log_root_path = os.path.join("logs", "rl_runs", "skrl_eureka", experiment_cfg["agent"]["experiment"]["directory"])
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        # get checkpoint path

        if args_cli.checkpoint:
            resume_path = os.path.abspath(args_cli.checkpoint)
        else:
            from isaaclab_tasks.utils import get_checkpoint_path
            resume_path = get_checkpoint_path(
                log_root_path, run_dir=f'.*Run.*',other_dirs=["checkpoints"]
            )
        try:
            dt = env.physics_dt
        except AttributeError:
            dt = env.unwrapped.physics_dt
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        env = SkrlVecEnvWrapper(env)
        # configure and instantiate the skrl runner
        # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
        
        from skrl.utils.runner.torch import Runner
        runner = Runner(env, experiment_cfg)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        # set agent to evaluation mode
        runner.agent.set_running_mode("eval")

        # reset environment
        obs, _ = env.reset()
        timestep = 0
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()

            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                # - multi-agent (deterministic) actions
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                # - single-agent (deterministic) actions
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                # env stepping
                obs, _, _, _, _ = env.step(actions)

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent with Eureka.")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-Direct-v0", help="Name of the task.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--device", type=str, default="cuda", help="The device to run training on.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Absolute path to model checkpoint.")
    parser.add_argument(
        "--rl_library",
        type=str,
        default="rsl_rl",
        choices=["rsl_rl", "rl_games", "skrl"],
        help="The RL training library to use.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Force display off at all times.",
    )
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
    args_cli = parser.parse_args()

    # Run the main function
    main(args_cli)
