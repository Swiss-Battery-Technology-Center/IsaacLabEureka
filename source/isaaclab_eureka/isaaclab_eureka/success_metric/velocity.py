import torch

def compute_success_metric(self, env_ids):
    # Config
    command_name = self.reward_manager.get_term_cfg("track_lin_vel_xy_exp").params["command_name"]
    asset_cfg = self.reward_manager.get_term_cfg("track_lin_vel_xy_exp").params["asset_cfg"]
    std_lin = self.reward_manager.get_term_cfg("track_lin_vel_xy_exp").params["std"]
    std_ang = self.reward_manager.get_term_cfg("track_ang_vel_z_exp").params["std"]

    # Get asset
    asset = self.scene[asset_cfg.name]

    # Compute linear velocity similarity (xy-plane)
    lin_vel_error = torch.sum(
        torch.square(self.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    lin_similarity = torch.exp(-lin_vel_error / std_lin**2)

    # Compute angular velocity similarity (yaw/z)
    ang_vel_error = torch.square(
        self.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2]
    )
    ang_similarity = torch.exp(-ang_vel_error / std_ang**2)

    # Success metric: average of linear and angular tracking
    success_metric = 0.5 * (lin_similarity + ang_similarity)

    # Average over all environments
    return success_metric.mean()
