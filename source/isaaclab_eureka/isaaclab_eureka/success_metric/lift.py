# task success metric for lift task
import torch
from isaaclab.utils.math import combine_frame_transforms
def compute_success_metric(self, env_ids):
    obj = self.scene["object"]
    robot = self.scene["robot"]
    command = self.command_manager.get_command("object_pose")

    command_pos_b = command[:, :3]
    command_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        command_pos_b,
    )
    object_pos_w = obj.data.root_pos_w[:, :3]

    dist = torch.norm(object_pos_w[env_ids] - command_pos_w[env_ids], dim=1)
    std = self.reward_manager.get_term_cfg("object_goal_tracking").params["std"]
    min_height = self.reward_manager.get_term_cfg("object_goal_tracking").params["minimal_height"]
    is_lifted = object_pos_w[env_ids, 2] > min_height
    is_close = (torch.tanh(dist/std) < 0.05) # is 5cm threshold good?
    success = torch.logical_and(is_close, is_lifted)

    return {
        "success_metric": success.float().mean(),
        "is_close": is_close.float().mean(),
        "is_lifted": is_lifted.float().mean(),
        "min_height": min_height,
        "dist": dist,
    }
