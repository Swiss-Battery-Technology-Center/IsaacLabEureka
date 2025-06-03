import torch

def compute_success_metric(self, env_ids):
    asset = self.scene["robot"]
    # Retrieve the target position
    _target_pos = self.reward_manager.get_term_cfg("progress").params["target_pos"]
    target_pos = torch.tensor(_target_pos, device=self.device)
    
    # Compute progress in x-direction (assuming world x is first coordinate)
    current_x = asset.data.root_pos_w[:, 0]
    target_x = target_pos[0]
    
    # Calculate the progress ratio
    progress_ratio = current_x / target_x
    
    # Average over *all* environments, not just env_ids
    success_metric = progress_ratio.mean()
    
    return {"success_metric":success_metric}
