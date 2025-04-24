# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import numpy as np
import os
import logging
import textwrap
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from typing import Literal
import traceback
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
    WEIGHT_TUNING_TASK_CRASH_FEEDBACK_PROMPT,
    WEIGHT_TUNING_TASK_FORMAT_ERROR_FEEDBACK_PROMPT,
    WEIGHT_TUNING_TASK_SUCCESS_POST_FEEDBACK_PROMPT,
    WEIGHT_TUNING_TASK_SUCCESS_PRE_FEEDBACK_PROMPT,
    PPO_TUNING_TASK_CRASH_FEEDBACK_PROMPT,
    PPO_TUNING_TASK_FORMAT_ERROR_FEEDBACK_PROMPT,
    PPO_TUNING_TASK_SUCCESS_PRE_FEEDBACK_PROMPT,
    PPO_TUNING_TASK_SUCCESS_POST_FEEDBACK_PROMPT,
)
from isaaclab_eureka.managers import EurekaTaskManager, LLMManager
from isaaclab_eureka.utils import load_tensorboard_logs, TrainingStatus


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
        eureka_task: str = "reward_weight_tuning",
        parameters_to_tune: list[str] = [],
        warmstart: bool = False,
        num_envs: int = 1024,
        resume: dict = {'enabled':False, 'resume_path': ""},
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
        # logging comes first
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if warmstart:
            self._log_dir = os.path.join(
                EUREKA_ROOT_DIR,
                "logs",
                "eureka",
                task,
                eureka_task,
                "warmstart",
                timestamp,
            )

        else:
            self._log_dir = os.path.join(
                EUREKA_ROOT_DIR,
                "logs",
                "eureka",
                task,
                eureka_task,
                "randstart",
                timestamp,
            )
        os.makedirs(self._log_dir)
        self._tensorboard_writer = TensorboardSummaryWriter(
            log_dir=self._log_dir, flush_secs=10
        )
        init_eureka_logger(self._log_dir)
        logging.info("Eureka initializing.")

        # Load the task description and success metric
        if task in TASKS_CFG:
            task_description = TASKS_CFG[task]["description"]
            success_metric_string = TASKS_CFG[task].get("success_metric")
            self._success_metric_to_win = TASKS_CFG[task].get("success_metric_to_win")
            self._success_metric_tolerance = TASKS_CFG[task].get(
                "success_metric_tolerance"
            )
            print(success_metric_string)
        else:
            raise ValueError(
                f"Task configuration for {task} not found in the `TASKS_CFG` dictionary in config/tasks.py."
            )

        self._task_description = task_description
        self._feedback_subsampling = feedback_subsampling
        self._num_processes = num_parallel_runs if env_type == "manager_based" else 1
        self._success_metric_string = success_metric_string
        if env_type == "manager_based":
            self.task_success_reward_name = TASK_SUCCESS_REWARD_NAME_DICT[task]
        else:
            self.task_success_reward_name = None
        print("[INFO]: Setting up the LLM Manager...")
        logging.info("Setting up the LLM Manager...")
        self._llm_manager = LLMManager(
            gpt_model=gpt_model,
            num_suggestions=1,
            temperature=temperature,
            env_type=env_type,
            eureka_task=eureka_task,
            parameters_to_tune=parameters_to_tune,
        )
        # set system prompt
        self._llm_manager.append_system_prompt(self._get_system_prompt(env_type=env_type, eureka_task=eureka_task))
        if self._num_processes > 1:
            self._llm_manager.append_to_system_prompt(
                MULTIPLE_SUGGESTIONS_INSTRUCTION.format(
                    num_parallel_runs=num_parallel_runs
                )
                + MULTIPLE_SUGGESTIONS_EXAMPLE
            )
        self._resume = resume
        print("[INFO]: Setting up the Task Manager...")
        logging.info("Setting up the Task Manager...")
        self._task_manager = EurekaTaskManager(
            task=task,
            device=device,
            env_seed=env_seed,
            rl_library=rl_library,
            num_processes=self._num_processes,
            max_training_iterations=max_training_iterations,
            success_metric_string=success_metric_string,
            env_type=env_type,
            eureka_task=eureka_task,
            parameters_to_tune=parameters_to_tune,
            warmstart=warmstart,
            num_envs=num_envs,
        )


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
            llm_outputs = self._llm_manager.prompt(
                user_prompt=user_prompt, assistant_prompt=assistant_prompt
            )
            gpt_reward_method_strings = llm_outputs["reward_strings"]
            # Log the llm outputs
            for idx, gpt_reward_method_string in enumerate(gpt_reward_method_strings):
                self._tensorboard_writer.add_text(
                    f"Run_{idx}/raw_llm_output", llm_outputs["raw_outputs"][idx], iter
                )
            # Train the RL agent
            results = self._task_manager.train(gpt_reward_method_strings)
            # Evaluate the results
            results, best_run_results, best_run_idx = self.evaluate_results(
                results, llm_outputs, best_run_results, gpt_reward_method_strings
            )

            self._log_iteration_results(iter, results)

            if (
                best_run_results["success_metric"] is not None
                and np.abs(
                    best_run_results["success_metric"] - self._success_metric_to_win
                )
                < self._success_metric_tolerance
            ):
                print(
                    f"Task solved with success metric: {best_run_results['success_metric']}"
                )
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
        #  context_code_string = self.read_env_source_code_brute(self._task_manager._task)
        smart_context_code_string = self._task_manager._context_code_string
        success_metric_code_string = self.read_success_metric_code()

        if self._task_manager._eureka_task == "reward_weight_tuning":
            self._llm_manager.append_user_prompt(smart_context_code_string)
            print("APPENDING CONTEXT CODE COMPLETE")
            logging.info("APPENDING CONTEXT CODE COMPLETE")

        if self._task_manager._eureka_task == "ppo_tuning":
            ppo_algo_code_string = self.read_ppo_source_code(self._task_manager._rl_library) 
            self._llm_manager.append_user_prompt(ppo_algo_code_string)
            print("APPENDING PPO CODE COMPLETE")
            logging.info("APPENDING PPO CODE COMPLETE")

        self._llm_manager.append_user_prompt(success_metric_code_string)
        print("APPENDING SUCCESS METRIC CODE COMPLETE")
        logging.info("APPENDING SUCCESS METRIC CODE COMPLETE")

        if self._resume["enabled"]:
            prev_iterations_txt = self.read_prev_iterations()   
            self._llm_manager.append_user_prompt(prev_iterations_txt)
            print("RESUME == TRUE, APPENDING PREVIOUS ITERATIONS COMPLETE")
            logging.info("RESUME == TRUE, APPENDING PREVIOUS ITERATIONS COMPLETE")
        print("APPENDING ALL COMPLETE")
        logging.info("APPENDING ALL COMPLETE")

        # this part was for calling the LLM the first time
        # However, now that we first run the training with initial tuning as in the code,
        # and then call the LLM with training results, we don't need this anymore

        # if self._task_manager._eureka_task == "ppo_tuning":
        #     user_prompt = MANAGER_BASED_PPO_TUNING_TASK_PROMPT.format(
        #         task_description=self._task_description,
        #         success_metric_to_win=self._success_metric_to_win,
        #     )
        #     user_prompt += " Here is the initial configuration of ppo parameters\n"
        # else:
        #     user_prompt = MANAGER_BASED_WEIGHT_TUNING_TASK_PROMPT.format(
        #         task_description=self._task_description,
        #         success_metric_to_win=self._success_metric_to_win,
        #     )
        #     user_prompt += " Here is the initial configuration of reward term weights\n"
        # if self._task_manager._get_initial_tuning_as_string is not None:
        #     user_prompt += self._task_manager._get_initial_tuning_as_string
        # user_prompt += context_code_string     
        #  assistant_prompt = self._task_manager._get_initial_tuning_as_string

        # The best run across all iterations
        best_run_results = {"success_metric": None}
        gpt_weight_strings = [None]*self._num_processes
        llm_outputs = None
        raw_output = None
        try: 
            for iter in range(max_eureka_iterations):

                print(f"\n{'#' * 20} Running Eureka Iteration {iter} {'#' * 20} \n")
                logging.info(f"Running Eureka Iteration {iter}")
                # WORKAROUND
                # For 0th iteration, we just use initial tuning from the code
                # only one process will train, the other processes will intentionally skip training
                # save raw output as assistant prompt ASAP, otherwise it might get lost and not save on eureka_conversation.txt
                if iter == 0:
                    if self._resume["enabled"]:
                        # 0th iter but resume, so call llm
                        print("RESUME == TRUE, CALLING LLM FROM 0TH ITER")
                        logging.info("RESUME == TRUE, CALLING LLM FROM 0TH ITER")
                        self._llm_manager.append_user_prompt(f"Based on the previous iterations, give me {self._num_processes} new suggestions.")
                        llm_outputs = self._llm_manager.call_llm()
                        print('LLM MANAGER CALLING COMPLETE')
                        logging.info('LLM MANAGER CALLING COMPLETE')
                        gpt_weight_strings = llm_outputs["weight_strings"]
                        raw_output = llm_outputs["raw_output"]
                        self._llm_manager.append_assistant_prompt(raw_output)
                    else:
                        # 0th iter fresh start, populate gpt_weight_strings with initial tuning
                        print("FRESH START, POPULATING GPT WEIGHT STRINGS")
                        logging.info("FRESH START, POPULATING GPT WEIGHT STRINGS")
                        SUGGESTION_PROMPT = f"Here is current configuration. Give me {self._num_processes} suggestions to start evolutionary search with, including current configuration as the first suggestion. You are encouraged to make wild guesses, since this will be the first generation in evolutioary search.\n"
                        self._llm_manager.append_user_prompt(SUGGESTION_PROMPT + self._task_manager._get_initial_tuning_as_string)
                        llm_outputs = self._llm_manager.call_llm()
                        print('LLM MANAGER CALLING COMPLETE')
                        logging.info('LLM MANAGER CALLING COMPLETE')    
                        gpt_weight_strings = llm_outputs["weight_strings"]
                        raw_output = llm_outputs["raw_output"]
                        self._llm_manager.append_assistant_prompt(raw_output)
                else:           
                    # Get new weights from LLM, from iter==1
                    print('LLM MANAGER CALLING START')
                    logging.info('LLM MANAGER CALLING START')
                    # 1st iter: user_prompt is generated from 0th iter training result
                    self._llm_manager.append_user_prompt(user_prompt)
                    llm_outputs = self._llm_manager.call_llm()
                    print('LLM MANAGER CALLING COMPLETE')
                    logging.info('LLM MANAGER CALLING COMPLETE')
                    gpt_weight_strings = llm_outputs["weight_strings"]
                    raw_output = llm_outputs["raw_output"]
                    self._llm_manager.append_assistant_prompt(raw_output)
                # Do I need this part?
                # for idx, gpt_reward_method_string in enumerate(gpt_weight_strings):
                #     self._tensorboard_writer.add_text(
                #         f"Run_{idx}/raw_llm_output", gpt_reward_method_string, iter
                #     )
                # Train the RL agent
                print('STARTING TASK MANAGER TRAIN')
                logging.info('STARTING TASK MANAGER TRAIN')
                print(f"GPT WEIGHT STRINGS: \n{gpt_weight_strings}")
                logging.info(f"GPT WEIGHT STRINGS: \n{gpt_weight_strings}")
                results = self._task_manager.train(gpt_weight_strings)
                print('TASK MANAGER TRAIN COMPLETE')
                logging.info('TASK MANAGER TRAIN COMPLETE')
                # bad fix, gpt_weight_strings is empty at iter==0 so use prev_weights_str instead
                # Evaluate the results
                # llm_outputs["raw_outputs"] is the raw response string

                results, best_run_results, best_run_idx = (
                    self.evaluate_results_weight_tuning(
                        results, best_run_results
                    )
                )
                logging.info(f"Evaluation complete")
                self._log_iteration_results(iter, results, raw_output)
                logging.info(f"Iteration results saved")

                if (
                    best_run_results["success_metric"] is not None
                    and np.abs(
                        best_run_results["success_metric"] - self._success_metric_to_win
                    )
                    < self._success_metric_tolerance
                ):
                    print(
                        f"Task solved with success metric: {best_run_results['success_metric']}"
                    )
                    break

                
                user_prompt = (results[best_run_idx]["user_prompt"] +
                                MULTIPLE_SUGGESTIONS_INSTRUCTION.format(num_parallel_runs=self._num_processes)
                                + MULTIPLE_SUGGESTIONS_EXAMPLE)
        except Exception as e:
            print(f"An error occurred during the Eureka training loop {iter}:")
            print(e)
            traceback.print_exc()
            logging.error(f"An error occurred during the Eureka training loop {iter}:\n{traceback.format_exc()}") 

            
            # Handle the error as needed
                
        finally:
            self._log_final_results(best_run_results, iter)
            self._log_conversation()
            # Close the task manager
            print("CLOSING TASK MANAGER")
            logging.info("CLOSING TASK MANAGER")
            self._task_manager.close()

    def _get_eureka_task_feedback(
        self, log_dir: str, feedback_subsampling: int
    ) -> tuple[str, float, float]:
        """Get the feedback for the Eureka task.

        Args:
            log_dir: The directory where the tensorboard logs are stored.
            feedback_subsampling: The subsampling of the metrics' trajectories.
        Returns:
            A tuple containing the feedback string, the maximum of the success metric, and the correlation between the oracle and GPT rewards.
        """
        if self._task_manager.is_manager_based():
            return self._get_eureka_task_feedback_manager_based(
                log_dir, feedback_subsampling
            )

        data = load_tensorboard_logs(log_dir)
        # Compute correlation between the oracle and GPT rewards
        eureka_rewards = np.array(
            next(
                (
                    data[key]
                    for key in data
                    if key.endswith("Eureka/eureka_total_rewards")
                ),
                None,
            )
        )
        oracle_rewards = np.array(
            next(
                (
                    data[key]
                    for key in data
                    if key.endswith("Eureka/oracle_total_rewards")
                ),
                None,
            )
        )
        # Sometimes, the tensorboard logging is not complete, we take the minimum length between the two buffers
        min_length = min(eureka_rewards.shape[0], oracle_rewards.shape[0])
        rewards_correlation = np.corrcoef(
            eureka_rewards[:min_length], oracle_rewards[:min_length]
        )[0, 1]

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
                metric_best = metric_data[
                    np.abs(np.array(metric_data) - self._success_metric_to_win).argmin()
                ]
                if metric_name == "success_metric":
                    metric_name = "task_score"
                    success_metric_max = metric_best
                data_string = [
                    f"{data:.2f}" for data in metric_data[::feedback_subsampling]
                ]
                feedback_string = (
                    f"{metric_name}: {data_string}, Min: {metric_min:.2f}, Max: {metric_max:.2f}, Mean:"
                    f" {metric_mean:.2f} \n"
                )
                if (
                    "Eureka/success_metric" in data
                    and metric_name == "Eureka/oracle_total_rewards"
                ):
                    # If success metric is available, we do not provide the oracle feedback
                    feedback_string = ""
                total_feed_back_string += feedback_string

        total_feed_back_string += (
            f"\nThe desired task_score to win is: {self._success_metric_to_win:.2f}\n"
        )
        return total_feed_back_string, success_metric_max, rewards_correlation

    def _get_eureka_task_feedback_manager_based(
        self, log_dir: str, feedback_subsampling: int
    ) -> tuple[str, float, float]:
        """Get the feedback for the Eureka task.

        Args:
            log_dir: The directory where the tensorboard logs are stored.
            feedback_subsampling: The subsampling of the metrics' trajectories.
        Returns:
            A tuple containing the feedback string, the maximum of the success metric, and the correlation between the oracle and GPT rewards.
        """

        # filtered data from tensorboard logs
        # data is a list of tuples (key, value), in order
        data = self.extract_relevant_metrics(log_dir=log_dir)

        success_metric_max = None

        def find_actual_iterations(data, suffix):
            """Find the correct key in `data` that ends with the given suffix."""
            for key, value in data:
                if key.endswith(suffix):
                    return len(value)
            raise KeyError(
                f"Could not find a key ending with '{suffix}' in TensorBoard logs."
            )

        actual_training_iterations = find_actual_iterations(data, "Eureka/success_metric")
        adaptive_feedback_subsampling = actual_training_iterations // feedback_subsampling
        # TODO really need to clean this up
        total_feed_back_string = ""
        # just give everything, hope that llm can figure it out
        for metric_name, metric_data in data:
            if len(metric_data) > 2:
                metric_data = metric_data[2:]
            metric_min = min(metric_data)
            metric_max = max(metric_data)
            metric_mean = sum(metric_data) / len(metric_data)
            # Best metric is the one closest to the target
            if "Eureka/success_metric" in metric_name:
                metric_best = metric_data[
                    np.abs(
                        np.array(metric_data) - self._success_metric_to_win
                    ).argmin()
                ]
                success_metric_max = metric_best
            data_string = [
                f"{data:.2f}" for data in metric_data[::adaptive_feedback_subsampling]
            ]
            feedback_string = (
                f"{metric_name}: {data_string}, Min: {metric_min:.2f}, Max: {metric_max:.2f}, Mean:"
                f" {metric_mean:.2f} \n"
            )
            if (
                "Eureka/success_metric" in data
                and metric_name == "Eureka/oracle_total_rewards"
            ):
                # If success metric is available, we do not provide the oracle feedback
                feedback_string = ""
            total_feed_back_string += feedback_string

        total_feed_back_string += (
            f"\nThe desired Eureka/success_metric to win is: {self._success_metric_to_win:.2f}\n"+
            f"Other fields under Eureka/ are intermediate values used to compute Eureka/success_metric, as given in compute_success_metric() function.\n"
        )
        total_feed_back_string += f"The agent is trained using {self._task_manager._rl_library} library.\n"
        total_feed_back_string += f"Each metric was sampled at every {adaptive_feedback_subsampling} learning iterations.\n"
        total_feed_back_string += f"Max learning iteration was set to {self._task_manager._max_training_iterations}, and the actual training had {actual_training_iterations} learning iterations.\n"
        total_feed_back_string += f"Recall that there may be curriculums acting on individual reward terms.\n"
        return total_feed_back_string, success_metric_max, 0

    def _log_iteration_results(self, iter: int, results: list, raw_output: str):
        """Log the results of the iteration."""
        # for idx, result in enumerate(results):
        #     print(f"{'*' * 20} Iteration {iter} / Process: {idx} {'*' * 20}")
        #     if result["success"]:
        #         print(
        #             f"Training successful with the following metrics:\n{result['eureka_task_feedback']}"
        #         )
        #         print(
        #             f"Reward correlation with oracle rewards: {result['rewards_correlation']}"
        #         )
        #     else:
        #         print(
        #             f"Training failed with the following exception:\n{result['exception']}\n"
        #         )

        # write the iterations results to file
        with open(f"{self._log_dir}/eureka_iterations.txt", "a") as f:
            if iter == 0:
                if self._resume["enabled"]:
                    f.write(f"Resuming from previous iterations\n")
                    f.write(f"{self._resume['prev_iterations_path']}\n\n")
                elif self._task_manager._warmstart:
                    f.write(f"Using Warmstart\n\n")
                else:
                    f.write(f"Using Randstart\n\n")
            if raw_output is not None:
                f.write(f"{'#' * 20} Iteration: {iter} {'#' * 20}\n\n")
                wrapped_output = textwrap.fill(raw_output, width=150)
                f.write(f"GPT reasoning:\n{wrapped_output}\n\n")
            for idx, result in enumerate(results):
                f.write(f"{'#' * 20} Iteration: {iter} {'#' * 20}\n\n")
                f.write(f"{'*' * 20} Process: {idx} {'*' * 20}\n")
                if result["success"] == TrainingStatus.SUCCESS:
                    f.write(f"- Configuration: \n{result['prev_config']}\n")
                    f.write(
                        f"Training successful with the following metrics:\n{result['eureka_task_feedback']}\n"
                    )
                    # f.write(
                    #     f"Reward correlation with oracle rewards:\n{result['rewards_correlation']}\n"
                    # )
                    self._tensorboard_writer.add_scalar(
                        f"Process_{idx}/success_metric", result["success_metric_max"], iter
                    )
                    self._tensorboard_writer.add_text(
                    f"Process_{idx}/run_feedback", result["user_prompt"], iter
                    )
                if result["success"] == TrainingStatus.CRASH:
                    f.write(f"- Configuration: \n{result['prev_config']}\n")
                    f.write(
                        f"Training crashed with the following exception:\n{result['exception']}\n"
                    )
                    f.write(
                        f"Training metrics are:\n{result['eureka_task_feedback']}\n"
                    )
                    self._tensorboard_writer.add_scalar(
                        f"Process_{idx}/success_metric", result["success_metric_max"], iter
                    )
                    self._tensorboard_writer.add_text(
                    f"Process_{idx}/run_feedback", result["user_prompt"], iter
                    )
                if result["success"] == TrainingStatus.FORMAT_ERROR:
                    f.write(
                        f"Wrong string format, training didn't run:\n{result['exception']}\n"
                    )
                f.write("\n")

    def _log_final_results(self, best_run_results: dict, iter):
        """Log the final results of the Eureka run."""
        output = ""
        if self._resume["enabled"]:
            output += f"Resuming from previous iterations\n"
            output += f"{self._resume['prev_iterations_path']}\n"
        elif self._task_manager._warmstart:
            output += f"Using Warmstart\n"
        else:
            output += f"Using Randstart\n"
        if best_run_results["success_metric"] is not None:
            output += f"- Success metric: {best_run_results['success_metric']}\n"
            output += f"- GPT config: {best_run_results['gpt_reward_method']}\n"
            output += f"- Task metrics:\n{best_run_results['task_feedback']}\n"
        else:
            output += "- No successful training run\n"

        output += f"For {self._task_manager._env_type}, {self._num_processes} parallel prompting, {iter} eureka iterations."
        output += f"Total token usage: {self._llm_manager._total_tokens}\n"
        output += f"Input token usage: {self._llm_manager._total_query_tokens}\n"
        output += f"Output token usage: {self._llm_manager._total_response_tokens}\n"
        charge = (
            self._llm_manager._total_query_tokens
            / (10**6)
            * self._llm_manager._input_token_price
            + self._llm_manager._total_response_tokens
            / (10**6)
            * self._llm_manager._output_token_price
        )
        output += f"Price: {charge:.3f}$\n"

        print(f"Final output: {output}")
        with open(f"{self._log_dir}/eureka_final_result.txt", "w") as f:
            f.write(output)
    
    def _log_conversation(self):
        with open(f"{self._log_dir}/eureka_conversation.txt", "w") as f:
            for chat in self._llm_manager._prompts:
                f.write(f"{chat['role']}: \n")
                f.write(f"{chat['content']}\n\n")


    def evaluate_results(
        self, results, llm_outputs, best_run_results, gpt_reward_method_strings
    ):
        iter_best_success_metric = None
        best_run_idx = 0

        for idx, result in enumerate(results):
            if not result["success"]:
                user_feedback_prompt = TASK_FAILURE_FEEDBACK_PROMPT.format(
                    traceback_msg=result["exception"]
                )
            else:
                # Compute the performance metrics
                eureka_task_feedback, success_metric_max, rewards_correlation = (
                    self._get_eureka_task_feedback(
                        result["log_dir"], self._feedback_subsampling
                    )
                )

                # Generate the user feedback prompt
                user_feedback_prompt = (
                    TASK_SUCCESS_PRE_FEEDBACK_PROMPT.format(
                        feedback_subsampling=self._feedback_subsampling
                    )
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
                        < np.abs(
                            best_run_results["success_metric"]
                            - self._success_metric_to_win
                        )
                    ):
                        best_run_results["success_metric"] = iter_best_success_metric
                        best_run_results["gpt_reward_method"] = (
                            gpt_reward_method_strings[idx]
                        )
                        best_run_results["task_feedback"] = eureka_task_feedback

            # Add the prompts
            results[idx]["user_prompt"] = user_feedback_prompt
            results[idx]["assistant_prompt"] = (
                llm_outputs["raw_outputs"][idx]
                if llm_outputs
                else results[0]["prev_weights_string"]
            )
        return results, best_run_results, best_run_idx

    def evaluate_results_weight_tuning(
        self, results, best_run_results
    ):
        iter_best_success_metric = None
        best_run_idx = 0

        for idx, result in enumerate(results):

            if "log_dir" in result: 
                # There is training data
                eureka_task_feedback, success_metric_max, rewards_correlation = (
                    self._get_eureka_task_feedback_manager_based(
                        result["log_dir"], self._feedback_subsampling
                    )
                )
            else:
                # training didn't even run, set success metric to -inf
                eureka_task_feedback = ""
                success_metric_max = -np.inf
                rewards_correlation = 0

            # SUCCESS, CRASH, FORMAT_ERROR, SKIPPED
            if result["success"] == TrainingStatus.SUCCESS:
                if self._task_manager._eureka_task == "reward_weight_tuning":
                    user_feedback_prompt = (
                        WEIGHT_TUNING_TASK_SUCCESS_PRE_FEEDBACK_PROMPT.format(
                            feedback_subsampling=self._feedback_subsampling
                        )
                        + "Using this configuration: \n" +result["prev_config"] + "\n\n"
                        + eureka_task_feedback
                        + WEIGHT_TUNING_TASK_SUCCESS_POST_FEEDBACK_PROMPT
                    )
                if self._task_manager._eureka_task == "ppo_tuning":
                    user_feedback_prompt = (
                        PPO_TUNING_TASK_SUCCESS_PRE_FEEDBACK_PROMPT.format(
                            feedback_subsampling=self._feedback_subsampling
                        )
                        + result["prev_config"] + "\n\n"
                        + eureka_task_feedback
                        + PPO_TUNING_TASK_SUCCESS_POST_FEEDBACK_PROMPT
                    )
            if result["success"] == TrainingStatus.CRASH:
                if self._task_manager._eureka_task == "reward_weight_tuning":
                    user_feedback_prompt = (
                        WEIGHT_TUNING_TASK_CRASH_FEEDBACK_PROMPT 
                        + result["exception"] 
                        + result["prev_config"] + "\n\n"
                        + eureka_task_feedback
                    )
                
                if self._task_manager._eureka_task == "ppo_tuning":
                    user_feedback_prompt = (
                        PPO_TUNING_TASK_CRASH_FEEDBACK_PROMPT 
                        + result["exception"] 
                        + result["prev_config"] + "\n\n"
                        + eureka_task_feedback
                    )
            if result["success"] == TrainingStatus.FORMAT_ERROR:
                if self._task_manager._eureka_task == "reward_weight_tuning":
                    user_feedback_prompt = (
                        WEIGHT_TUNING_TASK_FORMAT_ERROR_FEEDBACK_PROMPT 
                        + result["exception"] 
                        + f'\nThis is your previous suggestion: {result["prev_config"]}\n'
                        + f"However, the string must follow the format: {self._task_manager._get_initial_tuning_as_string}\n"
                    )
                
                if self._task_manager._eureka_task == "ppo_tuning":
                    user_feedback_prompt = (
                        PPO_TUNING_TASK_FORMAT_ERROR_FEEDBACK_PROMPT 
                        + result["exception"] 
                        + f'\nThis is your previous suggestion: {result["prev_config"]}\n'
                        + f"However, the string must follow the format: {self._task_manager._get_initial_tuning_as_string}\n"
                    )

                # Store the results
            results[idx]["eureka_task_feedback"] = eureka_task_feedback
            results[idx]["success_metric_max"] = success_metric_max
            results[idx]["rewards_correlation"] = rewards_correlation
            results[idx]["user_prompt"] = user_feedback_prompt
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
                < np.abs(
                    best_run_results["success_metric"]
                    - self._success_metric_to_win
                )
            ):
                best_run_results["success_metric"] = iter_best_success_metric
                best_run_results["gpt_reward_method"] = (
                    results[idx]["prev_config"]
                )
                best_run_results["task_feedback"] = eureka_task_feedback

            
        return results, best_run_results, best_run_idx

    def _get_system_prompt(self, env_type, eureka_task) -> str:
        """Determine the appropriate system prompt based on env_type and task_type."""

        prompts = {
            (
                "manager_based",
                "reward_weight_tuning",
            ): MANAGER_BASED_WEIGHT_TUNING_INITIAL_PROMPT,
            
            ("manager_based", "ppo_tuning"): MANAGER_BASED_PPO_TUNING_INITIAL_PROMPT,
            ("direct", "reward_weight_tuning"): DIRECT_WORKFLOW_INITIAL_PROMPT,
        }

        # Default to a generic prompt if no specific case is found
        return prompts[(env_type, eureka_task)] + f"The task is: {self._task_description}"

    def read_env_source_code_brute(self, task):
        """Recursively read all .py files in the mdp and task-specific directory inside IsaacLab."""
        import re

        base_dir = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/sbtc_tasks/manager_based"

        # 🔹 Extract task type dynamically (e.g., SBTC-Lift-Cube-Franka-OSC-v0 → sbtc_lift)
        match = re.search(r"SBTC-([A-Za-z]+)", task)
        rl_task_type = match.group(1).lower() if match else ""
        task_folder = f"sbtc_{rl_task_type}"  # Convert to folder format: "sbtc_lift", "sbtc_reach", etc.

        # 🔹 Define directories to read: always mdp + detected task folder
        directories_to_read = [os.path.join(base_dir, "mdp")]

        task_dir = os.path.join(base_dir, task_folder)

        try:
            if not os.path.exists(task_dir):
                raise FileNotFoundError(
                    f"⚠ ERROR: Task directory '{task_folder}' does not exist in {base_dir}"
                )
            directories_to_read.append(task_dir)
        except FileNotFoundError as e:
            print(str(e))
            return ""  # Return empty string to avoid breaking execution

        collected_code = [
            "Here is directory structure and source code of each file to help you understand my IsaacLab environment.\n"
        ]

        for directory in directories_to_read:
            # Extract only the last directory name for readability (e.g., "mdp", "sbtc_lift")
            relative_dir_name = os.path.basename(directory)
            collected_code.append(f"Directory: {relative_dir_name}")

            # Walk through all Python files
            for root, _, files in os.walk(directory):
                py_files = sorted(
                    [f for f in files if f.endswith(".py") and f != "__init__.py"]
                )  # 🔹 Ignore __init__.py
                if not py_files:
                    continue  # Skip empty directories

                # Show folder structure (relative to base_dir)
                collected_code.append(f"\n{relative_dir_name}")
                collected_code.append(
                    "├── " + "\n├── ".join(py_files)
                )  # Show directory tree

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
            raise ValueError(
                f"Unsupported RL library: {rl_library}. Choose from {list(ppo_paths.keys())}"
            )

        ppo_file_path = ppo_paths[rl_library]
        intro = f"Here is the source code of the PPO algorithm in {rl_library} library. Try to understand what each hyperparameter does.\n"
        try:
            with open(ppo_file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            return intro + source_code
        except FileNotFoundError:
            raise FileNotFoundError(f"PPO source code not found at {ppo_file_path}")
        
    def read_prev_iterations(self) -> str:
        """Read the previous iterations of the task and return them as a string."""
        prev_iterations_path = self._resume["prev_iterations_path"]
        try:
            with open(prev_iterations_path, "r", encoding="utf-8") as f:
                prev_iterations = f.read()
            intro="Previous eureka run ended abruptly. Here is the information from the previous run.\n"
            return intro + prev_iterations
        except FileNotFoundError:
            raise FileNotFoundError(f"Previous iterations file not found at {prev_iterations_path}")
    

    def read_success_metric_code(self) -> str:
        """Read the success metric code and return it as a string."""
        success_metric_path = f"/workspace/isaaclab/_isaaclab_eureka/source/isaaclab_eureka/isaaclab_eureka/success_metric/{self._task_manager._rl_task_type}.py"

        try:
            with open(success_metric_path, "r", encoding="utf-8") as f:
                success_metric_code = f.read()
            intro = f"This is how we define success metric for the given task: {self._task_description}.\n"
            return success_metric_code
        except FileNotFoundError:
            raise FileNotFoundError(f"Success metric code not found at {success_metric_path}")
        
    def extract_relevant_metrics(self, log_dir) -> list[tuple[str, list]]:
        import re
        data = load_tensorboard_logs(log_dir)

        # Define patterns and preferred order
        if self._task_manager._eureka_task == "reward_weight_tuning":
            ordered_patterns = [
                r".*Eureka/success_metric$",
                r".*Eureka/.+",
                r".*Episode_Reward/.+",
                r".*Reward / Total reward.+",
                r".*Reward / Instantaneous reward.+",
                r".*Train/mean_reward$",
            ]
        elif self._task_manager._eureka_task == "ppo_tuning":
            ordered_patterns = [
                r".*Eureka/success_metric$",
                r".*Loss\s*/\s*.+",
                r".*Policy\s*/\s*.+",
                r".*Learning rate",
                r".*Train/mean_reward$",
                r".*Train/mean_episode_length$",
                r".*Reward / Total reward.+",
                r".*Reward / Instantaneous reward.+",
            ]
        else:
            raise ValueError(
                f"Unsupported eureka task: {self._task_manager._eureka_task}. Choose from ['reward_weight_tuning', 'ppo_tuning']"
            )
        filtered_data = []
        seen = set()

        # Apply each pattern in order
        for pattern in ordered_patterns:
            for key in sorted(data.keys()):
                if key not in seen and re.match(pattern, key):
                    filtered_data.append((key, data[key]))
                    seen.add(key)

        return filtered_data
    

def init_eureka_logger(log_dir: str, filename: str = "eureka_debug.log", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    # If logger is already set up, don't reconfigure
    if logging.getLogger().hasHandlers():
        return

    logging.basicConfig(
        filename=log_path,
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w",  # overwrite on each run; change to "a" to append
    )


