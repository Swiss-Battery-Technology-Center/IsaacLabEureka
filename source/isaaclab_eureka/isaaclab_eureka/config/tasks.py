# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

TASKS_CFG = {
    "Isaac-Cartpole-Direct-v0": {
        "description": "balance a pole on a cart so that the pole stays upright",
        "success_metric": "self.episode_length_buf[env_ids].float().mean() / self.max_episode_length",
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.01,
    },
    "Isaac-Cartpole-v0": {
        "description": "balance a pole on a cart so that the pole stays upright",
        "success_metric": "self.episode_length_buf[env_ids].float().mean() / self.max_episode_length",
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.01,
    },
    "Isaac-Humanoid-v0": {
        "description": "the humanoid walks forward in the target direction as far as possible",
        "success_metric": "0",
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.01,
    },
    "Isaac-Velocity-Flat-Anymal-B-v0": {
        "description": "Track a velocity command on flat terrain with the Anymal B robot",
        "success_metric": "0",
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.01,
    },
    "Isaac-Reach-Franka-v0":{
        "description": "Move the end-effector to a sampled target position and orientation.",
        "success_metric": "0",
        "success_metric_to_win": 1, # now success metric is independent of reward weights!
        "success_metric_tolerance": 0.001,
    },
    "Isaac-Lift-Cube-Franka-v0":{
        "description": "Use Franka arm to lift an object and bring it to a target position in air.",
        "success_metric": "0",
        "success_metric_to_win": 1, # now success metric is independent of reward weights!
        "success_metric_tolerance": 0.001,
    },
    "Isaac-Open-Drawer-Franka-v0": {
        "description": "Franka arm approaches drawer handle, grasps it, and opens the drawer",
        "success_metric": "0",
        "success_metric_to_win": 1,
        "success_metric_tolerance": 0.01,
    },
    "SBTC-Lift-Cube-Franka-OSC-v0":{
        "description": "Use Franka arm to lift an object and bring it to a target position in air.",
        "success_metric": "0",
        "success_metric_to_win": 1, # now success metric is independent of reward weights!
        "success_metric_tolerance": 0.001,
    },
    "SBTC-Unscrew-Franka-OSC-v0":{
        "description": "Use Franka arm to approach a screw, engage and unscrew it.",
        "success_metric": "0",
        "success_metric_to_win":1,
        "success_metric_tolerance": 0.001,
    },

}


"""Configuration for the tasks supported by Isaac Lab Eureka.

`TASKS_CFG` is a dictionary that maps task names to their configuration. Each task configuration
is a dictionary that contains the following keys:

- `description`: A description of the task.
- `success_metric`: A Python expression that computes the success metric for the task.
- `success_metric_to_win`: The threshold for the success metric to win the task and stop.
- `success_metric_tolerance`: The tolerance for the success metric to consider the task successful.
"""
