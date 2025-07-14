import torch

def compute_success_metric(self, env_ids):
    # Config
    term_lin_cfg = self.reward_manager.get_term_cfg("track_lin_vel_xy_exp")
    term_ang_cfg = self.reward_manager.get_term_cfg("track_ang_vel_z_exp")

    command_name = term_lin_cfg.params["command_name"]
    std_lin = term_lin_cfg.params["std"]
    std_ang = term_ang_cfg.params["std"]
    asset_cfg = term_lin_cfg.params.get("asset_cfg", None)

    # Default to "robot" if not specified
    asset_name = asset_cfg.name if asset_cfg else "robot"
    asset = self.scene[asset_name]  # ✅ Correct way to retrieve the robot

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

    # Multiply similarities (range [0, 1])
    success_metric = (lin_similarity + ang_similarity)/2

    return {
        "success_metric": success_metric.mean(),
        "linear_similarity": lin_similarity.mean(),
        "angular_similarity": ang_similarity.mean(),
    }
