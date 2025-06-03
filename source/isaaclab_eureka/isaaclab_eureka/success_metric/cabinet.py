import torch
from isaaclab_tasks.manager_based.manipulation.cabinet.mdp.rewards import align_grasp_around_handle
def compute_success_metric(self, env_ids):
    # The asset config for the cabinet drawer
    asset_cfg = self.reward_manager.get_term_cfg("open_drawer_bonus").params["asset_cfg"]

    # Get the drawer joint position (shape: [num_envs])
    drawer_pos = self.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]

    # Define the target/maximum opening position for normalization
    target_drawer_pos = 0.4  # adjust based on your environment

    # Clip to [0,1] to ensure success metric is bounded
    success_metric = torch.clamp(drawer_pos / target_drawer_pos, 0.0, 1.0)

    # Optional: Only consider it "successfully open" if the grasp pose is valid
    grasp_valid = align_grasp_around_handle(self).float()
    success_metric = success_metric * grasp_valid

    return success_metric.mean()  # average across all environments
