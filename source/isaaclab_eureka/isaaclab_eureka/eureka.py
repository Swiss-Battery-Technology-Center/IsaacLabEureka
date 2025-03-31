# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from typing import Literal

from isaaclab_eureka import EUREKA_ROOT_DIR
from isaaclab_eureka.config import (
    DIRECT_WORKFLOW_INITIAL_PROMPT,
    DIRECT_WORKFLOW_TASK_PROMPT,
    TASK_FAILURE_FEEDBACK_PROMPT,
    TASK_SUCCESS_POST_FEEDBACK_PROMPT,
    TASK_SUCCESS_PRE_FEEDBACK_PROMPT,
    TASKS_CFG,
    TASK_SUCCESS_REWARD_NAME_DICT,
    MANAGER_BASED_WEIGHT_TUNING_TASK_PROMPT,
    MANAGER_BASED_PPO_TUNING_TASK_PROMPT,
    MANAGER_BASED_WEIGHT_TUNING_INITIAL_PROMPT,
    MANAGER_BASED_PPO_TUNING_INITIAL_PROMPT,
    MULTIPLE_SUGGESTIONS_EXAMPLE,
    MULTIPLE_SUGGESTIONS_INSTRUCTION,
    WEIGHT_TUNING_TASK_FAILURE_FEEDBACK_PROMPT,
    WEIGHT_TUNING_TASK_SUCCESS_POST_FEEDBACK_PROMPT,
    WEIGHT_TUNING_TASK_SUCCESS_PRE_FEEDBACK_PROMPT,
    PPO_TUNING_TASK_FAILURE_FEEDBACK_PROMPT,
    PPO_TUNING_TASK_SUCCESS_PRE_FEEDBACK_PROMPT,
    PPO_TUNING_TASK_SUCCESS_POST_FEEDBACK_PROMPT
)
from isaaclab_eureka.managers import EurekaTaskManager, LLMManager
from isaaclab_eureka.utils import load_tensorboard_logs

class Eureka:
    """Orchestrates the training of the RL agent using the LLM."""

    def __init__(
        self,
        task: str,
        device: str = "cuda",
        env_seed: int = 42,
        rl_library: Literal["rsl_rl", "rl_games", "skrl"] = "rsl_rl",
        max_training_iterations: int = 100,
        feedback_subsampling: int = 10,
        temperature: float = 1.0,
        gpt_model: str = "gpt-4",
        num_parallel_runs: int = 1,
        env_type: str = "",
        task_type: str = "reward_weight_tuning",
        parameters_to_tune: list[str] = [],
        warmstart:bool = False,
    ):
        """Initialize the Eureka class.

        Args:

            task: The task to train the agent on.
            device: The device to run the training on.
            env_seed: The seed to use for the environment
            rl_library: The RL library to use for training.
            max_training_iterations: The maximum number of training iterations for the RL agent.
            feedback_subsampling: The subsampling of the metrics given as feedack to the LLM.
            temperature: The temperature to use for the GPT model.
            gpt_model: The GPT model to use.
            num_parallel_runs: The number of runs to execute in parallel.
        """

        # Load the task description and success metric
        if task in TASKS_CFG:
            task_description = TASKS_CFG[task]["description"]
            success_metric_string = TASKS_CFG[task].get("success_metric")
            self._success_metric_to_win = TASKS_CFG[task].get("success_metric_to_win")
            self._success_metric_tolerance = TASKS_CFG[task].get("success_metric_tolerance")
            print(success_metric_string)
        else:
            raise ValueError(
                f"Task configuration for {task} not found in the `TASKS_CFG` dictionary in config/tasks.py."
            )

        self._task_description = task_description
        self._feedback_subsampling = feedback_subsampling
        self._num_processes = num_parallel_runs if env_type == "manager_based" else 1
        self._success_metric_string = success_metric_string
        self.task_success_reward_name = TASK_SUCCESS_REWARD_NAME_DICT[task]
        print("[INFO]: Setting up the LLM Manager...")
        self._llm_manager = LLMManager(
                gpt_model=gpt_model,
                num_suggestions=self._num_processes,
                temperature=temperature,
                system_prompt=self._get_system_prompt(env_type, task_type),
                env_type=env_type,
                task_type=task_type,
                parameters_to_tune=parameters_to_tune,
            )
        # this should apply to all manager based, so fix later
        if self._num_processes > 1:
            self._llm_manager.append_to_system_prompt(MULTIPLE_SUGGESTIONS_INSTRUCTION.format(num_parallel_runs=num_parallel_runs) + MULTIPLE_SUGGESTIONS_EXAMPLE)
        
        print("[INFO]: Setting up the Task Manager...")
        self._task_manager = EurekaTaskManager(
            task=task,
            device=device,
            env_seed=env_seed,
            rl_library=rl_library,
            num_processes=self._num_processes,
            max_training_iterations=max_training_iterations,
            success_metric_string=success_metric_string,
            env_type=env_type,
            task_type=task_type,
            parameters_to_tune=parameters_to_tune,
            warmstart=warmstart,
        )

        # Logging
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if warmstart:
            self._log_dir = os.path.join(EUREKA_ROOT_DIR, "logs", "eureka", task, task_type, "warmstart", timestamp)
        else:
            self._log_dir = os.path.join(EUREKA_ROOT_DIR, "logs", "eureka", task, task_type,"randstart", timestamp)
        os.makedirs(self._log_dir)
        self._tensorboard_writer = TensorboardSummaryWriter(log_dir=self._log_dir, flush_secs=10)
    
    
    def run(self, max_eureka_iterations: int):
        if self._task_manager.is_manager_based():
            self.run_manager_based(max_eureka_iterations)
        else:
            self.run_direct(max_eureka_iterations)

    def run_direct(self, max_eureka_iterations: int):
        """Run the Eureka training loop.

        Args:
            max_eureka_iterations: The maximum number of Eureka iterations to run.
        """
        # Initial prompts
        user_prompt = DIRECT_WORKFLOW_TASK_PROMPT.format(
            task_description=self._task_description,
            success_metric_to_win=self._success_metric_to_win,
            get_observations_method_as_string=self._task_manager.get_observations_method_as_string,
        )
        # The assistant prompt is used to feed the previous LLM output back into the LLM
        assistant_prompt = None

        # The best run across all iterations
        best_run_results = {"success_metric": None}

        for iter in range(max_eureka_iterations):
            print(f"\n{'#' * 20} Running Eureka Iteration {iter} {'#' * 20} \n")
            # Generate the GPT reward methods
            llm_outputs = self._llm_manager.prompt(user_prompt=user_prompt, assistant_prompt=assistant_prompt)
            gpt_reward_method_strings = llm_outputs["reward_strings"]
            # Log the llm outputs
            for idx, gpt_reward_method_string in enumerate(gpt_reward_method_strings):
                self._tensorboard_writer.add_text(f"Run_{idx}/raw_llm_output", llm_outputs["raw_outputs"][idx], iter)
            # Train the RL agent
            results = self._task_manager.train(gpt_reward_method_strings)
            # Evaluate the results
            results, best_run_results, best_run_idx = self.evaluate_results(results, llm_outputs, best_run_results, gpt_reward_method_strings)

            self._log_iteration_results(iter, results)

            if (
                best_run_results["success_metric"] is not None
                and np.abs(best_run_results["success_metric"] - self._success_metric_to_win)
                < self._success_metric_tolerance
            ):
                print(f"Task solved with success metric: {best_run_results['success_metric']}")
                break

            assistant_prompt = results[best_run_idx]["assistant_prompt"]
            user_prompt = results[best_run_idx]["user_prompt"]

        self._log_final_results(best_run_results, max_eureka_iterations)
        # Close the task manager
        self._task_manager.close()

    def run_manager_based(self, max_eureka_iterations: int):
        """Run the Eureka training loop.

        Args:
            max_eureka_iterations: The maximum number of Eureka iterations to run.
        """
        # Initial prompts
        # temporary fix, bc with weight tuning the initial prompt is not used
        context_code_string = self.read_env_source_code(self._task_manager._task)
        ppo_algo_code_string = self.read_ppo_source_code(self._task_manager._rl_library)
        if self._task_manager._task_type == "reward_weight_tuning":
            self._llm_manager.feed_context_code(context_code_string)
        if self._task_manager._task_type == "ppo_tuning":
            self._llm_manager.feed_context_code(ppo_algo_code_string)
        if self._task_manager._task_type == "ppo_tuning":
            user_prompt = MANAGER_BASED_PPO_TUNING_TASK_PROMPT.format(
                task_description=self._task_description,
                success_metric_to_win=self._success_metric_to_win,
            )
            user_prompt += " Here is the initial configuration of ppo parameters\n"
        else:
            user_prompt = MANAGER_BASED_WEIGHT_TUNING_TASK_PROMPT.format(
                task_description=self._task_description,
                success_metric_to_win=self._success_metric_to_win,
            )
            user_prompt += " Here is the initial configuration of reward term weights\n"
        if self._task_manager._get_initial_tuning_as_string is not None:
            user_prompt += self._task_manager._get_initial_tuning_as_string
        # user_prompt += context_code_string
        # The assistant prompt is used to feed the previous LLM output back into the LLM
        assistant_prompt = self._task_manager._get_initial_tuning_as_string

        # The best run across all iterations
        best_run_results = {"success_metric": None}
        gpt_weight_strings = []
        llm_outputs=None
        for iter in range(max_eureka_iterations):
            print(f"\n{'#' * 20} Running Eureka Iteration {iter} {'#' * 20} \n")
            # Get new weights from LLM, from iter==1
            llm_outputs = self._llm_manager.prompt_weights(user_prompt=user_prompt, assistant_prompt=assistant_prompt)
            gpt_weight_strings = llm_outputs["weight_strings"]
            if iter == 0 and self._task_manager._warmstart:
                # if warmstart, overwrite gpt_weight_strings with the initial tuning string
                for i in range(len(gpt_weight_strings)):
                    gpt_weight_strings[i] = self._task_manager._get_initial_tuning_as_string
            # Log the llm outputs
            for idx, gpt_reward_method_string in enumerate(gpt_weight_strings):
                self._tensorboard_writer.add_text(f"Run_{idx}/raw_llm_output", gpt_reward_method_string, iter)
            # Train the RL agent
            results = self._task_manager.train(gpt_weight_strings)
            # bad fix, gpt_weight_strings is empty at iter==0 so use prev_weights_str instead
            # Evaluate the results
            # llm_outputs["raw_outputs"] == [response.message.content], a list of length 1
            # llm_outputs["raw_outputs"][0] is the raw response string
            results, best_run_results, best_run_idx = self.evaluate_results_weight_tuning(results, llm_outputs, best_run_results, gpt_weight_strings)

            self._log_iteration_results(iter, results)

            if (
                best_run_results["success_metric"] is not None
                and np.abs(best_run_results["success_metric"] - self._success_metric_to_win)
                < self._success_metric_tolerance
            ):
                print(f"Task solved with success metric: {best_run_results['success_metric']}")
                break

            assistant_prompt = results[best_run_idx]["assistant_prompt"]
            user_prompt = results[best_run_idx]["user_prompt"]

        self._log_final_results(best_run_results, iter)
        # Close the task manager
        self._task_manager.close()

    def _get_eureka_task_feedback(self, log_dir: str, feedback_subsampling: int) -> tuple[str, float, float]:
        """Get the feedback for the Eureka task.

        Args:
            log_dir: The directory where the tensorboard logs are stored.
            feedback_subsampling: The subsampling of the metrics' trajectories.
        Returns:
            A tuple containing the feedback string, the maximum of the success metric, and the correlation between the oracle and GPT rewards.
        """
        if self._task_manager.is_manager_based():
            return self._get_eureka_task_feedback_manager_based(log_dir, feedback_subsampling)
        
        data = load_tensorboard_logs(log_dir)
        # Compute correlation between the oracle and GPT rewards
        eureka_rewards = np.array(
            next((data[key] for key in data if key.endswith("Eureka/eureka_total_rewards")), None)
        )
        oracle_rewards = np.array(
            next((data[key] for key in data if key.endswith("Eureka/oracle_total_rewards")), None)
        )
        # Sometimes, the tensorboard logging is not complete, we take the minimum length between the two buffers
        min_length = min(eureka_rewards.shape[0], oracle_rewards.shape[0])
        rewards_correlation = np.corrcoef(eureka_rewards[:min_length], oracle_rewards[:min_length])[0, 1]

        success_metric_max = None
        # Make a summary of each plot in the tensorboard logs
        total_feed_back_string = ""
        for metric_name, metric_data in data.items():
            if "Eureka/" in metric_name:
                # Remove the first two data points as they are usually outliers
                # some metrics like 'quat_reward' have less than 2 entries?
                if len(metric_data) > 2:
                    metric_data = metric_data[2:]
                metric_name = metric_name.split("Eureka/", 1)[-1]
                metric_min = min(metric_data)
                metric_max = max(metric_data)
                metric_mean = sum(metric_data) / len(metric_data)
                # Best metric is the one closest to the target
                metric_best = metric_data[np.abs(np.array(metric_data) - self._success_metric_to_win).argmin()]
                if metric_name == "success_metric":
                    metric_name = "task_score"
                    success_metric_max = metric_best
                data_string = [f"{data:.2f}" for data in metric_data[::feedback_subsampling]]
                feedback_string = (
                    f"{metric_name}: {data_string}, Min: {metric_min:.2f}, Max: {metric_max:.2f}, Mean:"
                    f" {metric_mean:.2f} \n"
                )
                if "Eureka/success_metric" in data and metric_name == "Eureka/oracle_total_rewards":
                    # If success metric is available, we do not provide the oracle feedback
                    feedback_string = ""
                total_feed_back_string += feedback_string

        total_feed_back_string += f"\nThe desired task_score to win is: {self._success_metric_to_win:.2f}\n"
        return total_feed_back_string, success_metric_max, rewards_correlation
    
    def _get_eureka_task_feedback_manager_based(self, log_dir: str, feedback_subsampling: int) -> tuple[str, float, float]:
        """Get the feedback for the Eureka task.

        Args:
            log_dir: The directory where the tensorboard logs are stored.
            feedback_subsampling: The subsampling of the metrics' trajectories.
        Returns:
            A tuple containing the feedback string, the maximum of the success metric, and the correlation between the oracle and GPT rewards.
        """
        
        data = load_tensorboard_logs(log_dir)

        success_metric_max = None
        # overwrite Eureka/success_metric data with task specific reward term
        def find_matching_key(data, suffix):
            """Find the correct key in `data` that ends with the given suffix."""
            for key in data.keys():
                if key.endswith(suffix):
                    return key
            raise KeyError(f"Could not find a key ending with '{suffix}' in TensorBoard logs.")

        # Find the correct keys dynamically
        success_metric_key = find_matching_key(data, "Eureka/success_metric")
        reward_term_key = find_matching_key(data, "Episode_Reward/" + TASK_SUCCESS_REWARD_NAME_DICT[self._task_manager._task])

        # Overwrite Eureka/success_metric with the correct reward term data
        data[success_metric_key] = data[reward_term_key]
        # TODO really need to clean this up
        total_feed_back_string = ""
        if self._task_manager._task_type == "reward_weight_tuning":
            # for reward weight tuning, only track reward term and success rate history
            for metric_name, metric_data in data.items():
                if "Episode_Reward/" in metric_name:
                    # Remove the first two data points as they are usually outliers
                    # some metrics like 'quat_reward' have less than 2 entries?
                    if len(metric_data) > 2:
                        metric_data = metric_data[2:]
                    metric_name = metric_name.split("Episode_Reward/", 1)[-1]
                    metric_min = min(metric_data)
                    metric_max = max(metric_data)
                    metric_mean = sum(metric_data) / len(metric_data)
                    data_string = [f"{data:.2f}" for data in metric_data[::feedback_subsampling]]
                    feedback_string = (
                        f"{metric_name}: {data_string}, Min: {metric_min:.2f}, Max: {metric_max:.2f}, Mean:"
                        f" {metric_mean:.2f} \n"
                    )
                    if "Eureka/success_metric" in data and metric_name == "Eureka/oracle_total_rewards":
                        # If success metric is available, we do not provide the oracle feedback
                        feedback_string = ""
                    total_feed_back_string += feedback_string
                elif "Eureka/" in metric_name:
                    if len(metric_data) > 2:
                        metric_data = metric_data[2:]
                    metric_min = min(metric_data)
                    metric_max = max(metric_data)
                    metric_mean = sum(metric_data) / len(metric_data)
                    # Best metric is the one closest to the target
                    if "success_metric" in metric_name:
                        metric_name = "task_score"
                        metric_best = metric_data[np.abs(np.array(metric_data) - self._success_metric_to_win).argmin()]
                        success_metric_max = metric_best
                    data_string = [f"{data:.2f}" for data in metric_data[::feedback_subsampling]]
                    feedback_string = (
                        f"{metric_name}: {data_string}, Min: {metric_min:.2f}, Max: {metric_max:.2f}, Mean:"
                        f" {metric_mean:.2f} \n"
                    )
                    total_feed_back_string += feedback_string
        if self._task_manager._task_type == "ppo_tuning":
            # for ppo tuning, just give everything
            # not truncating metric names, it will be
            # Episode_Reward/reward_name, Eureka/success_metric, etc.
            for metric_name, metric_data in data.items():
                if len(metric_data) > 2:
                    metric_data = metric_data[2:]
                metric_min = min(metric_data)
                metric_max = max(metric_data)
                metric_mean = sum(metric_data) / len(metric_data)
                # Best metric is the one closest to the target
                if "Eureka/success_metric" in metric_name:
                    metric_name = "task_score"
                    metric_best = metric_data[np.abs(np.array(metric_data) - self._success_metric_to_win).argmin()]
                    success_metric_max = metric_best
                data_string = [f"{data:.2f}" for data in metric_data[::feedback_subsampling]]
                feedback_string = (
                    f"{metric_name}: {data_string}, Min: {metric_min:.2f}, Max: {metric_max:.2f}, Mean:"
                    f" {metric_mean:.2f} \n"
                )
                if "Eureka/success_metric" in data and metric_name == "Eureka/oracle_total_rewards":
                    # If success metric is available, we do not provide the oracle feedback
                    feedback_string = ""
                total_feed_back_string += feedback_string

        total_feed_back_string += f"\nThe desired task_score to win is: {self._success_metric_to_win:.2f}\n"
        return total_feed_back_string, success_metric_max, 0

    def _log_iteration_results(self, iter: int, results: list):
        """Log the results of the iteration."""
        for idx, result in enumerate(results):
            print(f"{'*' * 20} Iteration {iter} / Process: {idx} {'*' * 20}")
            if result["success"]:
                print(f"Training successful with the following metrics:\n{result['eureka_task_feedback']}")
                print(f"Reward correlation with oracle rewards: {result['rewards_correlation']}")
            else:
                print(f"Training failed with the following exception:\n{result['exception']}\n")

        # write the iterations results to file
        with open(f"{self._log_dir}/eureka_iterations.txt", "a") as f:
            for idx, result in enumerate(results):
                if iter == 0:
                    if self._task_manager._warmstart:
                        f.write(f"Using Warmstart\n")
                    else:
                        f.write(f"Using Randstart\n")
                f.write(f"{'#' * 20} Iteration: {iter} {'#' * 20}\n\n")
                f.write(f"{'*' * 20} Run: {idx} {'*' * 20}\n")
                f.write(f"- GPT reward method {result['assistant_prompt']}\n")
                if not (iter == 0 and self._task_manager._warmstart):
                    f.write(f"GPT reasoning:\n{result['raw_llm_output']}\n")
                if result["success"]:
                    f.write(f"Training successful with the following metrics:\n{result['eureka_task_feedback']}\n")
                    f.write(f"Reward correlation with oracle rewards:\n{result['rewards_correlation']}\n")
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric", result["success_metric_max"], iter)
                else:
                    f.write(f"Training failed with the following exception:\n{result['exception']}\n")
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric", 0.0, iter)
                self._tensorboard_writer.add_text(f"Run_{idx}/run_feedback", result["user_prompt"], iter)
                f.write("\n")

    def _log_final_results(self, best_run_results: dict, iter):
        """Log the final results of the Eureka run."""
        output = ""
        if self._task_manager._warmstart:
            output += f"Using Warmstart\n"
        else:   
            output += f"Using Randstart\n"
        if best_run_results["success_metric"] is not None:
            output += f"- Success metric: {best_run_results['success_metric']}\n"
            output += f"- GPT reward method: {best_run_results['gpt_reward_method']}\n"
            output += f"- Task metrics:\n{best_run_results['task_feedback']}\n"
        else:
            output += "- No successful training run\n"
        
        output += f"For {self._task_manager._env_type}, {self._num_processes} parallel prompting, {iter} eureka iterations."
        output += f"Total token usage: {self._llm_manager._total_tokens}\n"
        output += f"Input token usage: {self._llm_manager._total_query_tokens}\n"
        output += f"Output token usage: {self._llm_manager._total_response_tokens}\n"
        charge = self._llm_manager._total_query_tokens / (10**6)*self._llm_manager._input_token_price + self._llm_manager._total_response_tokens / (10**6)*self._llm_manager._output_token_price
        output += f"Price: {charge:.3f}$\n"

        print(f"Final output: {output}")
        with open(f"{self._log_dir}/eureka_final_result.txt", "w") as f:
            f.write(output)
    
    def evaluate_results(self, results, llm_outputs, best_run_results, gpt_reward_method_strings):
        iter_best_success_metric = None
        best_run_idx = 0

        for idx, result in enumerate(results):
            if not result["success"]:
                user_feedback_prompt = TASK_FAILURE_FEEDBACK_PROMPT.format(traceback_msg=result["exception"])
            else:
                # Compute the performance metrics
                eureka_task_feedback, success_metric_max, rewards_correlation = self._get_eureka_task_feedback(
                    result["log_dir"], self._feedback_subsampling
                )

                # Generate the user feedback prompt
                user_feedback_prompt = (
                    TASK_SUCCESS_PRE_FEEDBACK_PROMPT.format(feedback_subsampling=self._feedback_subsampling)
                    + eureka_task_feedback
                    + TASK_SUCCESS_POST_FEEDBACK_PROMPT
                )

                # Store the results
                results[idx]["eureka_task_feedback"] = eureka_task_feedback
                results[idx]["success_metric_max"] = success_metric_max
                results[idx]["rewards_correlation"] = rewards_correlation

                # Check the best performing metric, determined by the minimum distance from the win target
                if success_metric_max is not None and (
                    iter_best_success_metric is None
                    or np.abs(success_metric_max - self._success_metric_to_win)
                    < np.abs(iter_best_success_metric - self._success_metric_to_win)
                ):
                    # Store the best run for this iteration
                    iter_best_success_metric = success_metric_max
                    best_run_idx = idx

                    # Store the best metric across all iterations
                    if best_run_results["success_metric"] is None or (
                        np.abs(iter_best_success_metric - self._success_metric_to_win)
                        < np.abs(best_run_results["success_metric"] - self._success_metric_to_win)
                    ):
                        best_run_results["success_metric"] = iter_best_success_metric
                        best_run_results["gpt_reward_method"] = gpt_reward_method_strings[idx]
                        best_run_results["task_feedback"] = eureka_task_feedback

            # Add the prompts
            results[idx]["user_prompt"] = user_feedback_prompt
            results[idx]["assistant_prompt"] = llm_outputs["raw_outputs"][idx] if llm_outputs else results[0]["prev_weights_string"]
        return results, best_run_results, best_run_idx
    
    def evaluate_results_weight_tuning(self, results, llm_outputs, best_run_results, gpt_reward_method_strings):
        iter_best_success_metric = None
        best_run_idx = 0

        for idx, result in enumerate(results):
            if not result["success"]:
                if self._task_manager._task_type == "reward_weight_tuning":
                    user_feedback_prompt = WEIGHT_TUNING_TASK_FAILURE_FEEDBACK_PROMPT.format(traceback_msg=result["exception"])
                if self._task_manager._task_type == "ppo_tuning":
                    user_feedback_prompt = PPO_TUNING_TASK_FAILURE_FEEDBACK_PROMPT.format(traceback_msg=result["exception"])
            else:
                # Compute the performance metrics
                eureka_task_feedback, success_metric_max, rewards_correlation = self._get_eureka_task_feedback_manager_based(
                    result["log_dir"], self._feedback_subsampling
                )

                # Generate the user feedback prompt
                if self._task_manager._task_type == "reward_weight_tuning":
                    user_feedback_prompt = (
                        WEIGHT_TUNING_TASK_SUCCESS_PRE_FEEDBACK_PROMPT.format(feedback_subsampling=self._feedback_subsampling)
                        + eureka_task_feedback
                        + WEIGHT_TUNING_TASK_SUCCESS_POST_FEEDBACK_PROMPT
                    )
                if self._task_manager._task_type == "ppo_tuning":
                    user_feedback_prompt = (
                        PPO_TUNING_TASK_SUCCESS_PRE_FEEDBACK_PROMPT.format(feedback_subsampling=self._feedback_subsampling)
                        + eureka_task_feedback
                        + PPO_TUNING_TASK_SUCCESS_POST_FEEDBACK_PROMPT
                    )

                # Store the results
                results[idx]["eureka_task_feedback"] = eureka_task_feedback
                results[idx]["success_metric_max"] = success_metric_max
                results[idx]["rewards_correlation"] = rewards_correlation

                # Check the best performing metric, determined by the minimum distance from the win target
                if success_metric_max is not None and (
                    iter_best_success_metric is None
                    or np.abs(success_metric_max - self._success_metric_to_win)
                    < np.abs(iter_best_success_metric - self._success_metric_to_win)
                ):
                    # Store the best run for this iteration
                    iter_best_success_metric = success_metric_max
                    best_run_idx = idx

                    # Store the best metric across all iterations
                    if best_run_results["success_metric"] is None or (
                        np.abs(iter_best_success_metric - self._success_metric_to_win)
                        < np.abs(best_run_results["success_metric"] - self._success_metric_to_win)
                    ):
                        best_run_results["success_metric"] = iter_best_success_metric
                        best_run_results["gpt_reward_method"] = gpt_reward_method_strings[idx]
                        best_run_results["task_feedback"] = eureka_task_feedback

            # Add the prompts
            results[idx]["user_prompt"] = user_feedback_prompt
            results[idx]["assistant_prompt"] = gpt_reward_method_strings[idx]
            results[idx]["raw_llm_output"] = llm_outputs["raw_outputs"][idx]
        return results, best_run_results, best_run_idx
    
    def _get_system_prompt(self, env_type, task_type) -> str:
        """Determine the appropriate system prompt based on env_type and task_type."""

        prompts = {
            ("manager_based", "reward_weight_tuning"): MANAGER_BASED_WEIGHT_TUNING_INITIAL_PROMPT + f'\n NEVER change reward weight for {self.task_success_reward_name} term. This term is used to compute task success metric. Changing its weight is the same as cheating the task success metric.',
            ("manager_based", "ppo_tuning"): MANAGER_BASED_PPO_TUNING_INITIAL_PROMPT,
            ("direct", "reward_weight_tuning"): DIRECT_WORKFLOW_INITIAL_PROMPT,
        }

        # Default to a generic prompt if no specific case is found
        return prompts[(env_type, task_type)]
    
    

    def read_env_source_code(self, task):
        """Recursively read all .py files in the mdp and task-specific directory inside IsaacLab."""
        import re
        base_dir = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sbtc/manager_based"

        # 🔹 Extract task type dynamically (e.g., SBTC-Lift-Cube-Franka-OSC-v0 → sbtc_lift)
        match = re.search(r"SBTC-([A-Za-z]+)", task)
        task_type = match.group(1).lower() if match else ""
        task_folder = f"sbtc_{task_type}"  # Convert to folder format: "sbtc_lift", "sbtc_reach", etc.

        # 🔹 Define directories to read: always mdp + detected task folder
        directories_to_read = [os.path.join(base_dir, "mdp")]

        task_dir = os.path.join(base_dir, task_folder)

        try:
            if not os.path.exists(task_dir):
                raise FileNotFoundError(f"⚠ ERROR: Task directory '{task_folder}' does not exist in {base_dir}")
            directories_to_read.append(task_dir)
        except FileNotFoundError as e:
            print(str(e))  
            return ""  # Return empty string to avoid breaking execution

        collected_code = ["Here is directory structure and source code of each file to help you understand my IsaacLab environment.\n"]

        for directory in directories_to_read:
            # Extract only the last directory name for readability (e.g., "mdp", "sbtc_lift")
            relative_dir_name = os.path.basename(directory)
            collected_code.append(f"Directory: {relative_dir_name}")

            # Walk through all Python files
            for root, _, files in os.walk(directory):
                py_files = sorted([f for f in files if f.endswith(".py") and f != "__init__.py"])  # 🔹 Ignore __init__.py
                if not py_files:
                    continue  # Skip empty directories

                # Show folder structure (relative to base_dir)
                collected_code.append(f"\n{relative_dir_name}")
                collected_code.append("├── " + "\n├── ".join(py_files))  # Show directory tree

                # Read Python files
                for file in py_files:
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()

                    # Append file contents with clear formatting
                    collected_code.append(f"\n---- {file} ----\n{file_content}\n")

        return "\n".join(collected_code)
    
    def read_ppo_source_code(self, rl_library: str) -> str:
        """Read the PPO source code of the given RL library and return it as a string."""
        ppo_paths = {
            "rsl_rl": "/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.10/site-packages/rsl_rl/algorithms/ppo.py",
            "skrl": "/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.10/site-packages/skrl/agents/torch/ppo/ppo.py",
        }

        if rl_library not in ppo_paths:
            raise ValueError(f"Unsupported RL library: {rl_library}. Choose from {list(ppo_paths.keys())}")

        ppo_file_path = ppo_paths[rl_library]
        intro=f"Here is the source code of the PPO algorithm in {rl_library} library. Try to understand what each hyperparameter does.\n"
        try:
            with open(ppo_file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            return intro + source_code
        except FileNotFoundError:
            raise FileNotFoundError(f"PPO source code not found at {ppo_file_path}")
