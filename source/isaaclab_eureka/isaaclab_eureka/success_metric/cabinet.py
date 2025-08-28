import torch

def compute_success_metric(self, env_ids):
    """
    Success = drawer joint position > 0.30 (fully open).
    No intermediate values returned.
    """
    idx = env_ids if env_ids is not None else slice(None)

    # Try to read the same joint used by the reward term
    try:
        asset_cfg = self.reward_manager.get_term_cfg("open_drawer_bonus").params["asset_cfg"]
        drawer = self.scene[asset_cfg.name]
        drawer_pos = drawer.data.joint_pos[idx, asset_cfg.joint_ids[0]]
    except Exception:
        # Fallback: use the cabinet's first joint (assumes 'drawer_top_joint' is index 0)
        drawer = self.scene["cabinet"]
        drawer_pos = drawer.data.joint_pos[idx, 0]

    success = drawer_pos > 0.30
    return {"success_metric": success.float().mean()}
