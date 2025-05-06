# task success metric for unscrew task
import torch
def compute_success_metric(self, env_ids):
    object = self.scene["object"]
    ee_frame = self.scene["ee_frame"]

    # 🔹 Retrieve current reward term parameters
    screw_engaged_cfg = self.reward_manager.get_term_cfg("screw_engaged")
    offset = screw_engaged_cfg.params["offset"]
    sigma = screw_engaged_cfg.params["sigma"]
    
    # 
    object_offset_w = torch.tensor(
        (0.0, 0.0, 0.0), device=object.data.root_pos_w.device
    )

    # 🔹 Positions
    object_pos_w = object.data.root_pos_w + object_offset_w             # (num_envs, 3)
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]                   # (num_envs, 3)

    # 🔹 Distance between EE and screw position
    dist = torch.norm(ee_pos_w[env_ids] - object_pos_w[env_ids], dim=1)

    step = (torch.tanh((dist - offset) / sigma) + 1.0) * 0.5

    # 🔹 Binary success: ee is within offset distance to screw.
    success = 1 - step
    return {"success_metric": success.mean(), 
            "dist": dist.mean(), 
            'object_offset': object_offset_w.mean()}