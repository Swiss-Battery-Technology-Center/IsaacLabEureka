import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

def compute_success_metric(self, env_ids):
    # Params
    term_cfg = self.reward_manager.get_term_cfg("object_goal_tracking")
    std = term_cfg.params["std"]
    minimal_height = term_cfg.params["minimal_height"]
    command_name = term_cfg.params["command_name"]
    robot_name = term_cfg.params.get("robot_cfg", SceneEntityCfg("robot")).name
    object_name = term_cfg.params.get("object_cfg", SceneEntityCfg("object")).name

    robot = self.scene[robot_name]
    obj = self.scene[object_name]
    command = self.command_manager.get_command(command_name)

    # Goal pose in world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b
    )

    # Distance to goal
    distance = torch.norm(des_pos_w - obj.data.root_pos_w[:, :3], dim=1)

    is_lifted = torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

    dist_score = 1 - torch.tanh(distance / std)

    # === 3. Combine ===
    success = 0.5*is_lifted + 0.5* dist_score

    return {
        "success_metric": success.mean(),
        "lift_score": is_lifted.mean(),
        "dist_score": dist_score.mean(),
        'distance': distance.mean(),
    }
