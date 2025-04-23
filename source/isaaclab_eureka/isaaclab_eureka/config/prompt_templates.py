# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Template strings used for prompting in Isaac Lab Eureka."""


DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS = """
Your reward function should use useful variables from the environment as inputs.
It must comply to the following signature exactly:

def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    return reward, individual_rewards_dict

Make sure any new tensor or variable you introduce is on the same device as self.device.
The output of the reward function should consist of two items:
    (1) the total reward, which has a dimension of (self.num_envs,) and is a torch.Tensor,
    (2) a dictionary of each individual reward component.
The code output should be formatted as a python code string: "```python ... ```" and contain only the get_rewards_eureka function.

Some helpful tips for writing the reward function code:
    (1) You may find it helpful to normalize the reward to a fixed range by applying transformations like torch.exp to the overall reward or its components
    (2) If you choose to transform a reward component, then you must also introduce a temperature parameter inside the transformation function; this parameter must be a named variable in the reward function and it must not be an input variable. Each transformed reward component should have its own temperature variable
    (3) Make sure the type of each input variable is correctly specified; a float input variable should not be specified as torch.Tensor
    (4) Most importantly, the reward code's input variables must contain only attributes of the provided environment class definition (namely, variables that have prefix self.). Under no circumstance can you introduce new input variables.
"""

MANAGER_BASED_WEIGHT_TUNING_FORMATTING_INSTRUCTIONS = """
Your new configuraiton string should comply exactly with the structure of the previous configuration, which will be given in user prompts.
It will generally look like:
    {'reward.term_name.weight': value_1, 'curriculum.term_name.param_name': value_2, ...}
I will use regex pattern of the above structure to extract the keys and values from your response. 
Use the same keys as the previous configuration, but suggest new values.
A key, 'reward.term_name.weight' for example, is a string enclosed by a single quote. The dots inside are used to reconstruct a nested dictionary.
The value will be mostly float or int, but always comply with the type of the previous configuration.
Negative reward weights are posssible, terms with negative weights serve as penalty rather than reward.
Note that num_step values in curriculum is in units of simulation steps, which is 24 * learning iterations.
For example, if num_step_start is 4800, the curriculum starts at 4800/24 = 200 learning iterations.
When you suggest new num_step values, please make sure they are multiples of 24.
By common sense, 0 < num_step_start < num_step_end < 24 * max_learning_iterations.
You are not obliged to change all values. If a certain value was good in the previous run, you can keep it as it is.
If training is not going well, you are encouraged to make wild guesses.
"""

MANAGER_BASED_PPO_TUNING_FORMATTING_INSTRUCTIONS = """
Your ppo hyperparameter tuning string should comply exactly with the previous tuning configuration, which will be given in user prompts.
    {'param_name_1': value_1, 'param_name_2': value_2, ...}
I will use regex pattern of the above structure to extract the ppo hyperparameters and suggested tuning values from your response.
param_name is a string and must be enclosed by a single quote.
param_name may contain dots, such as "agent.learning_rate_scheduler_kwargs.kl_threshold". Dots indicate a deeper level of nested dictionary.
value can be float or boolean, but always comply with the type of the previous configuration.
Note, if you think value of a certain parameter was good in the previous run, you can keep it as it is. You are not obliged to change all values, especially the boolean ones.
"""

MULTIPLE_SUGGESTIONS_INSTRUCTION = """
I want to do evolutionary search for the hyperparameters, so please provide {num_parallel_runs} different suggestions for hyperparameter tuning.
"""
MULTIPLE_SUGGESTIONS_EXAMPLE ="""
For ease of extraction, your respose should look like,

Suggestion 1
{'param_name_1': value_1, 'param_name_2': value_2, ...}

...

Suggestion N
{'param_name_1': value_1, 'param_name_2': value_2, ...}

After which you should add your analysis of the previous training run and explain why you think the new suggestions are better.
"""

DIRECT_WORKFLOW_INITIAL_PROMPT = """
You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effective as possible.
Your goal is to write a reward function for the environment that will help the agent learn the task described in text.
""" + DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS

MANAGER_BASED_WEIGHT_TUNING_INITIAL_PROMPT = """
You are a reward engineer trying to tune the environment configuration to solve reinforcement learning tasks as effective as possible in Isaac Lab manager based environment.
Your goal is to suggest better configuration tuning for the environment, so that the agent will learn the task described in text faster.
You will be given the source code of the manager based environment, which includes rewards, curriculums, etc. 
From the source code, you must understand the overarching structure of the environment, and how different components work together to provide dense rewards to facilitate learning of the ultimate task.
For example, there might be easy reward terms to guide the agent towards ultimate success. There might be curriculum terms that gradually increase the difficulty of the task.
You will also be given the source code of success metric, which computes the ratio of environments that accomplished given task, according to our Eureka/success_metric.
Eventually, you want to achieve desired success metric.
You will be given training progress and the configuration used in training, such as reward and curriculum.
Leverage your understanding of the environment to analyze the training progress and suggest better configurations.
""" + MANAGER_BASED_WEIGHT_TUNING_FORMATTING_INSTRUCTIONS

MANAGER_BASED_PPO_TUNING_INITIAL_PROMPT = """
You are an RL engineer trying to tune hyperparameters of ppo algorithm to solve reinforcement learning tasks as effective as possible in Isaac Lab manager based environment.
Your goal is to suggest better ppo hyperparameter tunings, so that the agent will learn the task described in text faster.
Use your knowledge of ppo algorithm and the task description to suggest better ppo hyperparameter tunings.
""" + MANAGER_BASED_PPO_TUNING_FORMATTING_INSTRUCTIONS

TASK_FAILURE_FEEDBACK_PROMPT = """
Executing the reward function code above has the following error: {traceback_msg}.
Please fix the bug and provide a new, improved reward function!
""" + DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS

WEIGHT_TUNING_TASK_CRASH_FEEDBACK_PROMPT = MANAGER_BASED_WEIGHT_TUNING_FORMATTING_INSTRUCTIONS + """
Training with previous reward weight configuration has the following error and dind't reach the maximum learning iterations.
Please provide a new, improved tuning of reward terms.
""" 

PPO_TUNING_TASK_CRASH_FEEDBACK_PROMPT = MANAGER_BASED_PPO_TUNING_FORMATTING_INSTRUCTIONS +"""
Training with previous ppo hyperparameter configuration has the following error and didn't reach the maximum learning iterations.
Please provide a new, improved tuning of ppo hyperparameters.
""" 

WEIGHT_TUNING_TASK_FORMAT_ERROR_FEEDBACK_PROMPT = MANAGER_BASED_WEIGHT_TUNING_FORMATTING_INSTRUCTIONS + """
Your weight tuning string has some format error. Please comply with the original structure.
""" 

PPO_TUNING_TASK_FORMAT_ERROR_FEEDBACK_PROMPT = MANAGER_BASED_PPO_TUNING_FORMATTING_INSTRUCTIONS +"""
Your ppo hyperparameter tuning string has some format error. Please comply with the original structure.
""" 

TASK_SUCCESS_PRE_FEEDBACK_PROMPT = """
We trained a RL policy using the provided reward function code and tracked the values of the individual components in the reward function as well as global policy metrics such as success rates and episode lengths after every {feedback_subsampling} epochs and the maximum, mean, minimum values encountered:
"""
WEIGHT_TUNING_TASK_SUCCESS_PRE_FEEDBACK_PROMPT = """
We trained a RL policy using the provided weights for reward terms and tracked the values of the individual reward components as well as other metrics such as Eureka/success_rate. 
We sampled it {feedback_subsampling} times evenly during training duration. We also tracked maximum, mean, minimum values encountered:
"""
PPO_TUNING_TASK_SUCCESS_PRE_FEEDBACK_PROMPT = """
We trained a RL policy using the provided tuning of ppo hyperparameters and tracked the values of relevant metrics. 
We sampled it {feedback_subsampling} times evenly during training duration. We also tracked maximum, mean, minimum values encountered:
"""
TASK_SUCCESS_POST_FEEDBACK_PROMPT = """
Please carefully analyze the policy feedback and provide a new, improved reward function that can better solve the task. Some helpful tips for analyzing the policy feedback:
    (1) If the success rates are always near zero, then you must rewrite the entire reward function
    (2) If the values for a certain reward component are near identical throughout, then this means RL is not able to optimize this component as it is written. You may consider
        (a) Changing its scale or the value of its temperature parameter
        (b) Re-writing the reward component
        (c) Discarding the reward component
    (3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range
Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.
""" + DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS

WEIGHT_TUNING_TASK_SUCCESS_POST_FEEDBACK_PROMPT = """
Please carefully analyze the policy feedback and provide a new, improved weights for reward terms that can better solve the task. 
""" + MANAGER_BASED_WEIGHT_TUNING_FORMATTING_INSTRUCTIONS

PPO_TUNING_TASK_SUCCESS_POST_FEEDBACK_PROMPT = """
Please carefully analyze the policy feedback and provide a new, improved tuning of ppo hyperparameters so that the agent will learn more effectively. 
""" + MANAGER_BASED_PPO_TUNING_FORMATTING_INSTRUCTIONS

DIRECT_WORKFLOW_TASK_PROMPT = """
Write a reward function for the following task: {task_description}
The desired task score is: {success_metric_to_win}
Here is how we get the observations from the environment:
{get_observations_method_as_string}
"""

MANAGER_BASED_WEIGHT_TUNING_TASK_PROMPT = """
Find a good configuration of environment for the task: {task_description}
The desired task score is: {success_metric_to_win}
"""

MANAGER_BASED_PPO_TUNING_TASK_PROMPT = """
Find a good tuning of ppo hyperparameters for the task: {task_description}
The desired task score is: {success_metric_to_win}
"""