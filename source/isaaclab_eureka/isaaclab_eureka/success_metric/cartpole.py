from isaaclab.utils.math import wrap_to_pi
import torch

def compute_success_metric(self, env_ids):
    asset_cfg = self.reward_manager.get_term_cfg("pole_pos").params["asset_cfg"]
    target = self.reward_manager.get_term_cfg("pole_pos").params["target"]  # scalar or 1-element list
    asset = self.scene[asset_cfg.name]

    # Wrap current joint pos to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])  # shape: (num_envs,)

    # Compute wrapped angle error
    angle_error = wrap_to_pi(joint_pos - target)  # shape: (num_envs,)

    # Cosine similarity, scaled to [0, 1]
    success_rate = (torch.cos(angle_error) + 1.0) / 2.0  # shape: (num_envs,)

    return {"success_metric": success_rate.mean()}  # average across all envs
