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
    "Isaac-Quadcopter-Direct-v0": {
        "description": (
            "bring the quadcopter to the target position: self._desired_pos_w, while making sure it flies smoothly"
        ),
        "success_metric": (
            "torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1).mean()"
        ),
        "success_metric_to_win": 0.0,
        "success_metric_tolerance": 0.2,
    },
    "Isaac-Factory-NutThread-Direct-v0": {
        "description": "thread a nut onto a bolt",
        "success_metric": ("(torch.logical_and(torch.linalg.vector_norm(self.target_held_base_pos[:, :2] - self.held_base_pos[:, :2], dim=1) < 0.0025,(self.held_base_pos[:, 2] - self.target_held_base_pos[:, 2]) < (self.cfg_task.fixed_asset_cfg.thread_pitch * 0.375)).float().mean())"
        ),
        "success_metric_to_win": 0.4,
        "success_metric_tolerance": 0.1,
    },
    "Isaac-Open-Drawer-Franka-v0": {
        "description": "Franka arm approaches drawer handle, grasps it, and opens the drawer",
        "success_metric": "self.scene.articulations['cabinet'].data.joint_pos[env_ids].float().mean()",
        "success_metric_to_win": 0.4,
        "success_metric_tolerance": 0.1,
    },
    "SBTC-Lift-Cube-Franka-OSC-v0":{
        "description": "Use Franka arm to lift an object and bring it to a target position in air. Task is considered to be successful if the distance between the object and the target position is less than threshold.",
        "success_metric": "0",
        "success_metric_to_win": 9, # because curriculum will increase object_goal_tracking by 10 times
        "success_metric_tolerance": 1,
    },
        "SBTC-Unscrew-Franka-OSC-v0":{
        "description": "Use Franka arm to approach a screw, engage and unscrew it. Task is considered to be successful if robot arm end effector engages/clicks with the screw.",
        "success_metric": "0",
        "success_metric_to_win": 0.9,
        "success_metric_tolerance": 0.1,
    },

}
TASK_SUCCESS_REWARD_NAME_DICT = {"SBTC-Lift-Cube-Franka-OSC-v0":"object_goal_tracking",
                                 "SBTC-Unscrew-Franka-OSC-v0":"screw_engaged"}



"""Configuration for the tasks supported by Isaac Lab Eureka.

`TASKS_CFG` is a dictionary that maps task names to their configuration. Each task configuration
is a dictionary that contains the following keys:

- `description`: A description of the task.
- `success_metric`: A Python expression that computes the success metric for the task.
- `success_metric_to_win`: The threshold for the success metric to win the task and stop.
- `success_metric_tolerance`: The tolerance for the success metric to consider the task successful.
"""
