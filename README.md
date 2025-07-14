## Eureka parameter tuning for IsaacLab manager based

Eureka is a flexibe, automated tuning framework for RL tasks in IsaacLab manager based environment. It provides source code context and training to LLM, and LLM suggests new tunings which are injected back to IsaacLab. After iterations, Eureka returns the best tuning: one that achieves highest success metric. LLM api calls are free. Currently it runs on SBTC tasks(`unscrew`, `lift`) and various default IsaacLab tasks(`cartpole`, `humanoid`, `velocity`, `franka reach`) but can be extended to new tasks. It supports `rsl_rl` and `skrl` library, can tune reward/curriculum/ppo parameters.

- Installation

    Go to root repo directory ``/workspace/isaaclab/_isaaclab_eureka`` of SBTC container.
    ```
    isaaclab --python -m pip install -e source/isaaclab_eureka
    ```

- Save your api key

    Get an api key from [openrouter](https://openrouter.ai/). 
    Go to api_keys folder, create `.env.api_keys` and save your key.
    
    ```
    OPENROUTER_API_KEY=your_api_key
    ```

- How to run

    set relevant parameters in `scripts/eureka_config.yaml` and run `scripts/train.py`.

    - Let `max_eureka_iterations` = X, `num_parallel_runs`=Y.
    - Eureka will run X iterations. At each iteration, Eureka will get Y different tunings and train Y different policies.
    - In the end it selects the best one from X*Y policies.
    - Most of the time, you only need to change `task`, `random_start`, `max_eureka_iterations`, `num_parallel_runs`, `max_training_iterations` and `parameters_to_tune`
    - for rl library, `rsl_rl` is fully tested, `skrl` is implemented but not tested
    - for eureka\_task, `reward_weight_tuning` supports reward weight tuning and curriculum tuning. Use `ppo_tuning` exclusively for ppo hyperparameter tuning.
    - in `parameters_to_tune`, give a list of parameters you want to tune, in nested structure. You can only tune reward/curriculum or ppo, not both jointly. Parameters should be given in format similar to below:
        - `reward.progress.weight`
        - `curriculum.reset_robot_joints.performance_low`
        - `algorithm.use_clipped_value_loss`
        - `agent.entropy_loss_scale`
    
    Note, the parameters should be task and rl library specific.
    
    There are example yaml files, such as `examples` folder or `ppo_tuning_ex.yaml`

- Results
    - inside `logs`, it saves 
        - `eureka_conversation.txt`: all inputs and outputs of LLM queries
        - `eureka_iterations.txt`: summary of each policy
        - `eureka_final_result.txt`: summary of the best policy and token usage. Disregard `price`, we are using free model.
    - the actual trained policies and tensorboard data are saved in `logs/rl_runs`

- Miscellaneous
    - Eureka reads source code from specific folder directory to provide LLM with context. Therefore, any newly added task must comply with either default IsaacLab style or SBTC style(in terms of folder names, file names, directory structure, etc)
    - In detail, instead of giving the raw source code to LLM(which may be too overwhelming), we use a summary of source code(Also LLM-generated). With `use_cache` in `eureka_config.yaml`, you can decide whether to reuse previous summary or regenerate summary(in case you changed prompting strategy)
    - For each task, you must define a success metric in `success_metric` folder. It allows to have a metric that is independent from reward values which are influenced by tuning. This `compute_success_metric(self, env_ids)` function is dynamically executed and attached to IsaacLab env instance, so you can write it in a similar way to how you write reward functions. Return a dictionary of any metrics including `success_metric` that you want to track, they will all be saved in tensorboard under `Eureka/`
    - For default IsaacLab tasks, you must additionally create a mapping between task name and alias: "Isaac-Cartpole-v0" and "cartpole" for example. It should be done in 'ENV_ID_TO_RL_TASK' of `eureka_task_mmanager.py` and `tasks.py`. For SBTC tasks it is done automatically.
    - Add new task to `tasks.py`. Disregard the `success_metric` field, just set it to zero. It was used by original Eureka implementation but you could only give one line of string, which is not enough to define complex success metric.
    - If Eureka run crashed for some reason, you can resume by giving path to previous `eureka_iterations.txt` in `eureka_config.yaml`, although not really used in practice.


## Overview(from original Eureka)

This repository is an implementation of *[Eureka](https://github.com/eureka-research/Eureka): Human-Level Reward Design via Coding Large Language Models* in Isaac Lab.
It prompts an LLM to discover and tune reward functions automatically for your specific task.

We support the native Openai and the Azure Openai APIs.

## Installation

- Make sure that you have either an [Openai API](https://platform.openai.com/api-keys) or [Azure Openai API](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line%2Cpython-new&pivots=programming-language-python) key.

- Install Isaac Lab, see the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

- Using a python interpreter that has Isaac Lab installed, install Isaac Lab Eureka
    ```
    python -m pip install -e source/isaaclab_eureka
    ```

## Running Isaac Lab Eureka

Run Eureka from the root repo directory ``IsaacLabEureka``.

The Openai API key has to be exposed to the script via an environment variable. We follow the Openai API convention and use ``OPENAI_API_KEY``, ``AZURE_OPENAI_API_KEY``, and ``AZURE_OPENAI_ENDPOINT``.

### Running with the Openai API

<details open>
<summary>Linux</summary>

```
OPENAI_API_KEY=your_key python scripts/train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=100 --rl_library="rl_games"
```
</details>

<details>
<summary>Windows</summary>

**Powershell**
```
$env:OPENAI_API_KEY="your_key"
python scripts\train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=100 --rl_library="rl_games"
```

**Command line**
```
set OPENAI_API_KEY=your_key
python scripts\train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=100 --rl_library="rl_games"
```
</details>

### Running with the Azure Openai API

<details open>
<summary>Linux</summary>

```
AZURE_OPENAI_API_KEY=your_key AZURE_OPENAI_ENDPOINT=azure_endpoint_url python scripts/train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=100 --rl_library="rl_games"
```
</details>

<details>
<summary>Windows</summary>

**Powershell**
```
$env:AZURE_OPENAI_API_KEY="your_key"
$env:AZURE_OPENAI_ENDPOINT="azure_endpoint_url"
python scripts\train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=100 --rl_library="rl_games"
```

**Command line**
```
set AZURE_OPENAI_API_KEY=your_key
set AZURE_OPENAI_ENDPOINT=azure_endpoint_url
python scripts\train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=100 --rl_library="rl_games"
```
</details>

### Running Eureka Trained Policies

For each Eureka run, logs for the Eureka iterations are available under ``IsaacLabEureka/logs/eureka``.
This directory holds files containing the output from each Eureka iteration, as well as output and metrics
of the final Eureka results for the task. The tensorboard log also contains a Text tab which shows the raw LLM output
and the provided feedback at every iteration.

In addition, trained policies during the Eureka run are saved under ``IsaacLabEureka/logs/rl_runs``.
This directory contains checkpoints for each valid Eureka run, similar to the checkpoints available
when training with Isaac Lab.

To run inference on an Eureka-trained policy, locate the path to the desired checkpoint and run the ``scripts/play.py`` script.

For RSL RL, run:

```
    python scripts/play.py --task=Isaac-Cartpole-Direct-v0 --checkpoint=/path/to/desired/checkpoint.pt --num_envs=20 --rl_library="rsl_rl"
```

For RL-Games, run:

```
    python scripts/play.py --task=Isaac-Cartpole-Direct-v0 --checkpoint=/path/to/desired/checkpoint.pth --num_envs=20 --rl_library="rl_games"
```

### Limitations

- Isaac Lab Eureka currently only supports tasks implemented in the direct-workflow style, basing off of the ``DirectRLEnv`` class.
Available examples can be found in the [task config](source/isaaclab_eureka/isaaclab_eureka/config/tasks.py). Following the ``DirectRLEnv``
interface, we assume each task has the observation function implemented in a method named ``_get_observations()``.
- Currently, only RSL RL and RL-Games libraries are supported.
- Due to limitations of multiprocessing on Windows, running with argument ``num_parallel_runs`` > 1 is not supported on Windows.
- When running with ``num_parallel_runs > 1`` on a single-GPU machine, training will run in parallel in the background and CPU and memory usage will increase.
- Best policy is selected based on the ``success_metric`` defined for the task. For best performance, make sure to define an accurate success metric in the task config to guide the reward function generation process.
- During the reward generation process, the LLM may generate code that introduces syntax or logical errors during the training process. In such case, the error message will be propagated to the output and the Eureka iteration will be skipped.


## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```
