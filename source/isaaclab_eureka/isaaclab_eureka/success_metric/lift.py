import torch
from isaaclab.utils.math import combine_frame_transforms

def compute_success_metric(self, env_ids):
    """
    Success = object is lifted above minimal_height AND close to commanded goal position.
    Also returns intermediate metrics for debugging/plots.
    """

    # --- 1) Locate the reward term config (robust to naming) ---
    term_cfg = self.reward_manager.get_term_cfg("object_goal_tracking")
    params = term_cfg.params
    # Required params (with safe defaults if missing)
    std = float(params.get("std", 0.05))  # position kernel width used in reward
    minimal_height = float(params.get("minimal_height", 0.08))
    command_name = params.get("command_name", "goal_pose")

    # Entity cfgs (default to conventional names if not provided)
    robot_cfg = params.get("robot_cfg", params.get("asset_cfg", None))
    object_cfg = params.get("object_cfg", None)
    robot_name = getattr(robot_cfg, "name", "robot")
    object_name = getattr(object_cfg, "name", "object")

    robot = self.scene[robot_name]
    obj = self.scene[object_name]

    # Optional env filtering
    idx = env_ids if env_ids is not None else slice(None)

    # --- 2) Compute desired world-frame goal position (like the reward term) ---
    command = self.command_manager.get_command(command_name)              # (N, >=3)
    des_pos_b = command[:, :3]                                           # (N, 3) commanded pos in robot frame
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w[idx], robot.data.root_quat_w[idx], des_pos_b[idx]
    )  # -> world-frame desired pos for each env in idx

    # --- 3) Distances & lifted condition ---
    obj_pos_w = obj.data.root_pos_w[idx]                                 # (M, 3)
    dists = torch.norm(des_pos_w - obj_pos_w, dim=1)                     # (M,)
    heights = obj_pos_w[:, 2]                                            # (M,)

    is_lifted = heights > minimal_height
    # Use the same scale as the reward kernel for "close to goal"
    dist_threshold = 2*std
    is_close = dists < dist_threshold

    # --- 4) Binary success & a smooth tracking score (for diagnostics) ---
    success = is_lifted & is_close
    tracking_score = (1.0 - torch.tanh(dists / std)) * is_lifted.float()  # ∈ [0,1], gated by lifted

    # --- 5) Aggregate and return ---
    # Note: keep values as tensors so your logger can handle them directly.
    return {
        "success_metric": success.float().mean(),
        "is_lifted_rate": is_lifted.float().mean(),
        "is_close_rate": is_close.float().mean(),
        "tracking_score_mean": tracking_score.mean(),
        "distance_to_goal/mean": dists.mean(),
        "height/mean": heights.mean(),
    }
