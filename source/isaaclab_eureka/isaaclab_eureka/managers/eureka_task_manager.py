# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import math
import multiprocessing
import os
import traceback
import types
import ast
import copy
from contextlib import nullcontext
from datetime import datetime
from typing import Literal
from isaaclab_eureka import EUREKA_ROOT_DIR
from isaaclab_eureka.utils import MuteOutput, get_freest_gpu
from isaaclab.utils.io.yaml import dump_yaml

TEMPLATE_REWARD_STRING = """
from {module_name} import *
import torch

def _get_rewards(self):
    rewards_oracle = self._get_rewards_oracle()
    rewards_eureka, rewards_dict = self._get_rewards_eureka()
    self._eureka_episode_sums["eureka_total_rewards"] += rewards_eureka
    self._eureka_episode_sums["oracle_total_rewards"] += rewards_oracle
    for key in rewards_dict.keys():
        if key not in self._eureka_episode_sums:
            self._eureka_episode_sums[key] = torch.zeros(self.num_envs, device=self.device)
        self._eureka_episode_sums[key] += rewards_dict[key]
    return rewards_eureka
"""


# Insert the logic to log the eureka episode sums.
TEMPLATE_RESET_STRING = """
from {module_name} import *

@torch.inference_mode()
def _reset_idx(self, env_ids):
    if env_ids is None or len(env_ids) == self.num_envs:
        env_ids = torch.arange(self.num_envs, device=self.device)
    extras = dict()
    # This needs to happen before self._reset_idx_original(env_ids) because it will reset buffers that might be needed
    {success_metric}
    self._reset_idx_original(env_ids)
    if not "log" in self.extras:
        self.extras["log"] = dict()
    for key in self._eureka_episode_sums.keys():
        episodic_sum_avg = torch.mean(self._eureka_episode_sums[key][env_ids])
        extras["Eureka/"+key] = episodic_sum_avg / self.max_episode_length_s
        self._eureka_episode_sums[key][env_ids] = 0.0
    self.extras["log"].update(extras)
"""
# Insert the logic to track success metric
MANAGER_BASED_RESET_STRING = """
from {module_name} import *

@torch.inference_mode()
def _reset_idx(self, env_ids):
    if env_ids is None or len(env_ids) == self.num_envs:
        env_ids = torch.arange(self.num_envs, device=self.device)
    extras = dict()
    # This needs to happen before self._reset_idx_original(env_ids) because it will reset buffers that might be needed
    {success_metric}
    self._reset_idx_original(env_ids)
    if not "log" in self.extras:
        self.extras["log"] = dict()
    if hasattr(self, "_eureka_episode_sums"):
        for key in self._eureka_episode_sums.keys():
            episodic_sum_avg = torch.mean(self._eureka_episode_sums[key][env_ids])
            extras["Eureka/"+key] = episodic_sum_avg / self.max_episode_length_s
            self._eureka_episode_sums[key][env_ids] = 0.0
    self.extras["log"].update(extras)
"""


class EurekaTaskManager:
    """Manages the set-up and training of a task using LLM-generated reward functions.

    It takes an existing IsaacLab task and inserts the Eureka-generated reward function or configuration into it. The
    rewards that are already defined in the task are kept to serve as an oracle signal.
    """

    def __init__(
        self,
        task: str,
        rl_library: Literal["rsl_rl", "rl_games", "skrl"] = "rsl_rl",
        num_processes: int = 1,
        device: str = "cuda",
        env_seed: int = 42,
        max_training_iterations: int = 100,
        success_metric_string: str = "",
        env_type: str = "",
        task_type: str = "",
        parameters_to_tune: list[str] = [],
        warmstart: bool = False,
        num_envs: int = 1024,
    ):
        """Initialize the task manager. Each process will create an independent training run.

        Args:
            task: The name of the task to train.
            rl_library: The RL library to use for training.
            num_processes: The number of processes to use for training.
            device: The device to run training on.
            env_seed: The seed to use for the environment.
            max_training_iterations: The maximum number of training iterations.
            success_metric_string: A string that represents an expression to calculate the success metric for the task.
        """
        self._task = task
        self._rl_library = rl_library
        self._num_processes = num_processes
        self._device = device
        self._max_training_iterations = max_training_iterations
        self._success_metric_string = success_metric_string
        self._env_seed = env_seed
        self._env_type = env_type
        if self._success_metric_string:
            self._success_metric_string = (
                "extras['Eureka/success_metric'] = " + self._success_metric_string
            )
        self._task_type = task_type
        self._parameters_to_tune = parameters_to_tune
        self._warmstart = warmstart
        self._num_envs = num_envs
        self._processes = dict()
        # Used to communicate the reward functions to the processes
        self._rewards_queues = [
            multiprocessing.Queue() for _ in range(self._num_processes)
        ]
        # Used to communicate the observations method to the main process
        self._observations_queue = multiprocessing.Queue()
        # Used to communicate the results of the training runs to the main process
        self._results_queue = multiprocessing.Queue()
        self._initial_tuning_queue = multiprocessing.Queue()
        # Used to get weights of reward terms in manager based
        # self._weights_queues = [multiprocessing.Queue() for _ in range(self._num_processes)]
        # Used to signal the processes to terminate
        self.termination_event = multiprocessing.Event()
        for idx in range(self._num_processes):
            if self.is_manager_based():
                p = multiprocessing.Process(
                    target=self._worker_manager_based,
                    args=(idx, self._rewards_queues[idx]),
                )
            else:
                p = multiprocessing.Process(
                    target=self._worker, args=(idx, self._rewards_queues[idx])
                )
            self._processes[idx] = p
            p.start()

        # Fetch the observations, not needed for weight tuning
        self._get_observations_as_string = None
        self._get_initial_tuning_as_string = None
        if not self.is_manager_based():
            self._get_observations_as_string = self._observations_queue.get()
        else:
            self._get_initial_tuning_as_string = self._initial_tuning_queue.get()

    @property
    def get_observations_method_as_string(self) -> str:
        """The _get_observations method of the environment as a string."""
        return (
            self._get_observations_as_string
            if self._get_observations_as_string
            else "Not available for manager-based environments."
        )

    def is_manager_based(self):
        return self._env_type == "manager_based"

    def close(self):
        """Close the task manager and clean up the processes."""
        self.termination_event.set()
        # Send a stop signal to the processes
        for rewards_queue in self._rewards_queues:
            rewards_queue.put("Stop")
        for process in self._processes.values():
            process.join()

    def train(self, get_rewards_method_as_string: list[str]) -> list[dict]:
        """Train the task with the specified reward functions.

        Note: The methods must have the following signature "_get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]".

        Args:
            get_rewards_method_as_string: A list of get_rewards methods as strings. The length of the list must match
            the number of processes.
        Returns:
            A list of dictionaries containing the results of each training run. The dictionary contains the following
            keys:
                - "success": True if the training was successful, False otherwise.
                - "log_dir": The directory where the training logs are stored if the training succeeded.
                - "exception": The exception message if the training failed.
        """
        if get_rewards_method_as_string:
            # get_rewards_method_as_string is empty list [] at iter 0 of weight tuning, in which case we skip this part
            if len(get_rewards_method_as_string) != self._num_processes:
                raise ValueError(
                    f"Number of reward methods in the list ({len(get_rewards_method_as_string)}) does not match the number"
                    f" of processes ({self._num_processes})."
                )

            # Set the reward functions in each process
            for idx, rewards_queue in enumerate(self._rewards_queues):
                rewards_queue.put(get_rewards_method_as_string[idx])

        results = [None] * self._num_processes
        # Wait for each process to finish and collect the results
        for _ in range(self._num_processes):
            idx, result = self._results_queue.get()
            results[idx] = result

        return results

    def _worker(self, idx: int, rewards_queue: multiprocessing.Queue):
        """The worker function that runs the training of the task.

        Args:
            idx: The index of the worker.
            rewards_queue: The queue to receive the reward function from the main process
        """

        self._idx = idx
        while not self.termination_event.is_set():
            if not hasattr(self, "_env"):
                self._create_environment()

                # Fetch the environment's _get_observations method and send it to the main process
                if self._idx == 0 and not hasattr(self, "_observation_string"):
                    self._observation_string = inspect.getsource(
                        self._env.unwrapped._get_observations
                    )
                    self._observations_queue.put(self._observation_string)

            # Insert the reward function into the environment and run the training
            reward_func_string = rewards_queue.get()
            if isinstance(reward_func_string, str) and reward_func_string.startswith(
                "def _get_rewards_eureka(self)"
            ):
                try:
                    self._prepare_eureka_environment(reward_func_string)
                    # Only print the output of process 0
                    context = MuteOutput() if self._idx > 0 else nullcontext()
                    with context:
                        # Run training and send result to main process
                        self._run_training()
                    result = {"success": True, "log_dir": self._log_dir}
                except Exception as e:
                    result = {"success": False, "exception": str(e)}
                    print(traceback.format_exc())

            else:
                result = {
                    "success": False,
                    "exception": (
                        "The reward function must be a string that starts with 'def _get_rewards_eureka(self)'."
                    ),
                }

            self._results_queue.put((self._idx, result))
        # Clean up
        print(f"[INFO]: Run {self._idx} terminated.")
        self._env.close()
        self._simulation_app.close()

    def _create_environment(self):
        from isaaclab.app import AppLauncher

        if self._device == "cuda":
            device_id = get_freest_gpu()
            self._device = f"cuda:{device_id}"
        app_launcher = AppLauncher(headless=True, device=self._device)
        self._simulation_app = app_launcher.app

        import isaaclab_tasks  # noqa: F401
        from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
        from isaaclab_tasks.utils import parse_env_cfg
        import gymnasium as gym

        # Load fresh env config from registry (avoids curriculum side effects)
        if self._env_type == "manager_based":
            env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(self._task)
        else:
            env_cfg: DirectRLEnvCfg = parse_env_cfg(self._task)

        env_cfg.sim.device = self._device
        env_cfg.seed = self._env_seed
        env_cfg.scene.num_envs = self._num_envs  # ensure consistency
        self._original_env_cfg = copy.deepcopy(env_cfg)
        # Create new env
        self._env = gym.make(self._task, cfg=env_cfg)

    def _reset_all_envs(self):
        if not hasattr(self, "_env") or self._env is None:
            return

        import torch, copy

        env = self._env.unwrapped
        reset_env_ids = torch.arange(env.num_envs, device=env.device)

        # Reset state
        env.recorder_manager.record_pre_reset(reset_env_ids)
        env._reset_idx(reset_env_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()
        if env.sim.has_rtx_sensors() and env.cfg.rerender_on_reset:
            env.sim.render()
        env.recorder_manager.record_post_reset(reset_env_ids)
        env.common_step_counter = 0
        env.episode_length_buf[:] = 0

        # Restore original reward cfg
        original_cfg = self._original_env_cfg
        try:
            from isaaclab.managers import RewardManager, EventManager

            env.reward_manager = RewardManager(original_cfg.rewards, env)
            env.event_manager = EventManager(original_cfg.events, env)
        except Exception as e:
            print(f"[ERROR] Failed to make a new reward manager: {e}")

        print(f"[INFO] Reset all envs and restored original cfg.")

    def _prepare_eureka_environment(self, get_rewards_method_as_string: str):
        """Prepare the environment for training with the Eureka-generated reward function.

        It renames the original reward function to _get_rewards_oracle, adds the Eureka-generated reward function to the
        environment, and sets the environment's _get_rewards method to a template method that calls both the Eureka and
        oracle reward functions. It also sets the environment's _reset_idx method to a template method that updates the
        episodic sum of the Eureka-generated rewards.
        """
        import torch

        env = self._env.unwrapped
        namespace = {}
        # Check if the environment has already been prepared
        if not hasattr(env, "_get_rewards_eureka"):
            # rename the environment's original reward function to _get_rewards_oracle
            env._get_rewards_oracle = env._get_rewards
            # rename to environment's initial reset function to _reset_idx_original
            env._reset_idx_original = env._reset_idx
            # set the _get_rewards method to the template method
            template_reward_string_with_module = TEMPLATE_REWARD_STRING.format(
                module_name=env.__module__
            )
            exec(template_reward_string_with_module, namespace)
            setattr(
                env, "_get_rewards", types.MethodType(namespace["_get_rewards"], env)
            )
            # set the _reset_idx method to the template method
            template_reset_string_with_success_metric = TEMPLATE_RESET_STRING.format(
                module_name=env.__module__, success_metric=self._success_metric_string
            )
            # hack: can't enable inference with rl_games
            if self._rl_library == "rl_games":
                template_reset_string_with_success_metric = (
                    template_reset_string_with_success_metric.replace(
                        "@torch.inference_mode()", ""
                    )
                )
            exec(template_reset_string_with_success_metric, namespace)
            setattr(env, "_reset_idx", types.MethodType(namespace["_reset_idx"], env))

        # Add the GPT generated reward function to the environment
        get_rewards_method_as_string = (
            f"from {env.__module__} import * \nimport torch\n"
            + get_rewards_method_as_string
        )
        exec(get_rewards_method_as_string, namespace)
        setattr(
            env,
            "_get_rewards_eureka",
            types.MethodType(namespace["_get_rewards_eureka"], env),
        )

        # Prepare the reward sum buffers
        env._eureka_episode_sums = dict()
        env._eureka_episode_sums["eureka_total_rewards"] = torch.zeros(
            env.num_envs, device=env.device
        )
        env._eureka_episode_sums["oracle_total_rewards"] = torch.zeros(
            env.num_envs, device=env.device
        )

    def _worker_manager_based(self, idx: int, rewards_queue: multiprocessing.Queue):
        self._idx = idx
        self._has_sent_initial_tuning = False
        self._eureka_iter = 0
        while not self.termination_event.is_set():
            if not hasattr(self, "_env"):
                self._create_environment()
                if self._idx == 0 and not self._has_sent_initial_tuning:
                    if self._task_type == "reward_weight_tuning":
                        self._initial_tuning_as_string = self.get_initial_tuning()
                        self._initial_tuning_queue.put(self._initial_tuning_as_string)
                    if self._task_type == "ppo_tuning":
                        self._initial_tuning_as_string = self._get_initial_ppo_params()
                        self._initial_tuning_queue.put(self._initial_tuning_as_string)
                    self._has_sent_initial_tuning = True

            # uses new weights from llm
            # if weight string was not properly formatted, llm manager will return ""
            # do exception handling here
            new_weights_string = rewards_queue.get()
            if (
                len(new_weights_string) > 0
            ):  # if not empty string, weight string is properly formatted
                try:
                    if self._task_type == "reward_weight_tuning":
                        self._prepare_eureka_environment_reset_weights(
                            new_weights_string
                        )
                    self._prepare_eureka_environment_reset_idx()
                    context = MuteOutput() if self._idx > 0 else nullcontext()
                    with context:
                        if self._task_type == "ppo_tuning":
                            llm_param_names = ast.literal_eval(
                                new_weights_string
                            ).keys()
                            if set(self._parameters_to_tune) != set(llm_param_names):
                                raise Exception(
                                    f"Parameter names {llm_param_names} in suggested tuning do not match the correct parameter names: {self._parameters_to_tune}"
                                )
                            self._ppo_param_string = new_weights_string
                        self._run_training()
                        self._reset_all_envs()
                    result = {"success": True, "log_dir": self._log_dir}
                except Exception as e:
                    result = {"success": False, "exception": str(e)}
                    print(traceback.format_exc())
            else:
                result = {
                    "success": False,
                    "exception": (
                        "Your string response must follow the syntax. {'term1_name': 0.5, 'term2_name': 1.0}"
                        "It should be a string representation of dictionary,"
                        "whose keys are the term names as string and values are floats."
                        "term names should be enlosed with single quotes"
                    ),
                }
            # construct {term_name : weight} dictionary
            prev_weights_dict = {
                name: cfg.weight
                for name, cfg in zip(
                    self._env.unwrapped.reward_manager._term_names,
                    self._env.unwrapped.reward_manager._term_cfgs,
                )
            }
            result["prev_weights_str"] = repr(prev_weights_dict)
            self._eureka_iter += 1
            self._results_queue.put((self._idx, result))
        # Clean up
        print(f"[INFO]: Run {self._idx} terminated.")
        self._env.close()
        self._simulation_app.close()

    def get_initial_tuning(self):
        rm = self._env.unwrapped.reward_manager
        initial_weights_dict = {
            name: cfg.weight for name, cfg in zip(rm._term_names, rm._term_cfgs)
        }
        return repr(initial_weights_dict)

    def _prepare_eureka_environment_reset_weights(self, new_weights_string: str):
        # runs only for reward weight tuning
        # ppo tuning comes later
        if self._task_type != "reward_weight_tuning":
            return
        import torch

        env = self._env.unwrapped
        new_weights_dict = ast.literal_eval(new_weights_string)
        for name, weight in new_weights_dict.items():
            try:
                cfg = env.reward_manager.get_term_cfg(name)
                cfg.weight = weight
                env.reward_manager.set_term_cfg(name, cfg)
            except Exception as e:
                print(f"Error setting weight for {name} : {e}")

    def _prepare_eureka_environment_reset_idx(self):
        # runs only once for each process
        import torch

        env = self._env.unwrapped
        namespace = {}
        if not hasattr(env, "_reset_idx_original"):
            env._reset_idx_original = env._reset_idx
            template_reset_string_with_success_metric = (
                MANAGER_BASED_RESET_STRING.format(
                    module_name=env.__module__,
                    success_metric=self._success_metric_string,
                )
            )
            # hack: can't enable inference with rl_games
            if self._rl_library == "rl_games":
                template_reset_string_with_success_metric = (
                    template_reset_string_with_success_metric.replace(
                        "@torch.inference_mode()", ""
                    )
                )
            exec(template_reset_string_with_success_metric, namespace)
            setattr(env, "_reset_idx", types.MethodType(namespace["_reset_idx"], env))

    def _get_initial_ppo_params(self):
        """Get the agent configuration for the task."""
        import time

        time.sleep(10)
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        if self._rl_library == "rsl_rl":
            from rsl_rl.runners import OnPolicyRunner
            from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

            agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(
                self._task, "rsl_rl_cfg_entry_point"
            )
            agent_cfg_dict = agent_cfg.to_dict()
        elif self._rl_library == "skrl":
            import skrl
            from skrl.utils.runner.torch import Runner
            from isaaclab_rl.skrl import SkrlVecEnvWrapper

            # for skrl it is already a dict
            agent_cfg_dict = load_cfg_from_registry(self._task, "skrl_cfg_entry_point")
        else:
            raise Exception(f"PPO tuning supports only rsl_rl and skrl")
        initial_tuning = {}
        for param_path in self._parameters_to_tune:
            try:
                value = agent_cfg_dict
                for key in param_path.split("."):
                    value = value[key]  # Traverse the dictionary
                initial_tuning[param_path] = value  # Store as 'key1.key2.key3': value
            except KeyError:
                raise KeyError(
                    f"Parameter {'.'.join(param_path)} not found in agent_cfg"
                )
        return repr(initial_tuning)

    def _run_training(
        self, framework: Literal["rsl_rl", "rl_games", "skrl"] = "rsl_rl"
    ):
        """Run the training of the task."""
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        if self._rl_library == "rsl_rl":
            from rsl_rl.runners import OnPolicyRunner

            from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

            agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(
                self._task, "rsl_rl_cfg_entry_point"
            )
            agent_cfg.device = self._device
            agent_cfg.max_iterations = self._max_training_iterations

            log_root_path = os.path.join(
                "logs", "eureka", "rsl_rl", agent_cfg.experiment_name, self._task_type
            )
            log_root_path = os.path.abspath(log_root_path)
            if self._warmstart:
                log_root_path = os.path.join(log_root_path, "warmstart")
            else:
                log_root_path = os.path.join(log_root_path, "randstart")
            print(f"[INFO] Logging experiment in directory: {log_root_path}")
            # specify directory for logging runs: {time-stamp}_{run_name}
            log_dir = (
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                + f"_Run-{self._idx}_iter-{self._eureka_iter}"
            )
            if agent_cfg.run_name:
                log_dir += f"_{agent_cfg.run_name}"
            self._log_dir = os.path.join(log_root_path, log_dir)

            env = RslRlVecEnvWrapper(self._env)
            agent_cfg_dict = agent_cfg.to_dict()
            if self._task_type == "ppo_tuning":
                # update the agent configuration with the ppo tuning parameters
                ppo_tuning_dict = ast.literal_eval(self._ppo_param_string)
                for param_path, new_value in ppo_tuning_dict.items():
                    keys = param_path.split(".")
                    d = agent_cfg_dict
                    for key in keys[:-1]:
                        d = d[key]
                    d[keys[-1]] = new_value
            runner = OnPolicyRunner(
                env, agent_cfg_dict, log_dir=self._log_dir, device=agent_cfg.device
            )
            dump_yaml(os.path.join(self._log_dir, "params", "agent.yaml"), agent_cfg_dict)
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=True,
            )

        elif self._rl_library == "rl_games":
            from rl_games.common import env_configurations, vecenv
            from rl_games.common.algo_observer import IsaacAlgoObserver
            from rl_games.torch_runner import Runner

            from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

            agent_cfg = load_cfg_from_registry(self._task, "rl_games_cfg_entry_point")
            agent_cfg["params"]["config"]["max_epochs"] = self._max_training_iterations
            agent_cfg["params"]["config"]["device"] = self._device
            agent_cfg["params"]["config"]["device_name"] = self._device
            # specify directory for logging experiments
            log_root_path = os.path.join(
                "logs",
                "eureka",
                "rl_games",
                agent_cfg["params"]["config"]["name"],
                self._task_type,
            )
            log_root_path = os.path.abspath(log_root_path)
            if self._warmstart:
                log_root_path = os.path.join(log_root_path, "warmstart")
            else:
                log_root_path = os.path.join(log_root_path, "randstart")
            print(f"[INFO] Logging experiment in directory: {log_root_path}")
            # specify directory for logging runs
            log_dir = (
                agent_cfg["params"]["config"].get(
                    "full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                )
                + f"_Run-{self._idx}_iter-{self._eureka_iter}"
            )
            # set directory into agent config
            # logging directory path: <train_dir>/<full_experiment_name>
            agent_cfg["params"]["config"]["train_dir"] = log_root_path
            agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
            # Update the log directory to the tensorboard file
            self._log_dir = os.path.join(log_root_path, log_dir, "summaries")
            clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
            clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
            env = RlGamesVecEnvWrapper(self._env, self._device, clip_obs, clip_actions)

            vecenv.register(
                "IsaacRlgWrapper",
                lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(
                    config_name, num_actors, **kwargs
                ),
            )
            env_configurations.register(
                "rlgpu",
                {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
            )

            # set number of actors into agent config
            agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
            # create runner from rl-games
            runner = Runner(IsaacAlgoObserver())
            runner.load(agent_cfg)
            # reset the agent and env
            runner.reset()
            # train the agent
            runner.run({"train": True, "play": False, "sigma": None})
        elif self._rl_library == "skrl":
            import skrl
            from skrl.utils.runner.torch import Runner
            from isaaclab_rl.skrl import SkrlVecEnvWrapper

            # max iterations for training
            agent_cfg = load_cfg_from_registry(self._task, "skrl_cfg_entry_point")
            agent_cfg["trainer"]["timesteps"] = (
                self._max_training_iterations * agent_cfg["agent"]["rollouts"]
            )
            agent_cfg["trainer"]["close_environment_at_exit"] = False

            # set the agent and environment seed from command line
            # note: certain randomization occur in the environment initialization so we set the seed here
            agent_cfg["seed"] = self._env_seed
            agent_cfg["device"] = self._device

            # specify directory for logging experiments
            log_root_path = os.path.join(
                "logs",
                "eureka",
                "skrl",
                agent_cfg["agent"]["experiment"]["directory"],
                self._task_type,
            )
            log_root_path = os.path.abspath(log_root_path)
            if self._warmstart:
                log_root_path = os.path.join(log_root_path, "warmstart")
            else:
                log_root_path = os.path.join(log_root_path, "randstart")
            print(f"[INFO] Logging experiment in directory: {log_root_path}")
            # specify directory for logging runs: {time-stamp}_{run_name}
            log_dir = (
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                + f"_Run-{self._idx}_iter-{self._eureka_iter}"
            )

            print(f"Exact experiment name requested from command line {log_dir}")
            if agent_cfg["agent"]["experiment"]["experiment_name"]:
                log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
            # set directory into agent config
            agent_cfg["agent"]["experiment"]["directory"] = log_root_path
            agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
            # update log_dir
            log_dir = os.path.join(log_root_path, log_dir)
            self._log_dir = log_dir
            # wrap around environment for skrl
            env = SkrlVecEnvWrapper(
                self._env
            )  # ml_framework="torch", wrapper="isaaclab" by default
            if self._task_type == "ppo_tuning":
                # update the agent configuration with the ppo tuning parameters
                ppo_tuning_dict = ast.literal_eval(self._ppo_param_string)
                for param_path, new_value in ppo_tuning_dict.items():
                    keys = param_path.split(".")
                    d = agent_cfg
                    for key in keys[:-1]:
                        d = d[key]
                    d[keys[-1]] = new_value
            # configure and instantiate the skrl runner
            # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
            runner = Runner(env, agent_cfg)
            dump_yaml(os.path.join(self._log_dir, "params", "agent.yaml"), agent_cfg)
            # run training
            runner.run()
        else:
            raise Exception(f"framework {framework} is not supported yet.")
