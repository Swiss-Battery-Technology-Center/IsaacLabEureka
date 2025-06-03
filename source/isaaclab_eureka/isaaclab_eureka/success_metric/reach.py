import torch
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

def compute_success_metric(self, env_ids):
    # Load config
    pos_term_cfg = self.reward_manager.get_term_cfg("end_effector_position_tracking")
    command_name = pos_term_cfg.params["command_name"]
    asset_cfg = pos_term_cfg.params["asset_cfg"]
    asset_name = asset_cfg.name

    # Access robot end-effector
    asset = self.scene[asset_name]
    command = self.command_manager.get_command(command_name)

    # Desired position in world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)

    # Current end-effector position
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]

    # Position similarity (Gaussian kernel)
    pos_error = torch.norm(curr_pos_w - des_pos_w, dim=1)
    pos_sigma = 0.1  # You can adjust this
    position_is_close = torch.exp(-0.5 * (pos_error / pos_sigma)**2)

    # Desired orientation in world frame
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)

    # Current end-effector orientation
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]

    # Orientation similarity (Gaussian kernel on quat error magnitude)
    orient_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    orient_sigma = 0.05  # You can adjust this
    orientation_is_close = torch.exp(-0.5 * (orient_error / orient_sigma)**2)

    # Overall success metric: average of position and orientation similarities
    success = 0.5 * (position_is_close + orientation_is_close)

    return {
        "success_metric": success.mean(),
        "position_close": position_is_close,
        "orientation_close": orientation_is_close,
    }
