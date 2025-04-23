from isaaclab_eureka.utils import load_tensorboard_logs

# log_dir = "/workspace/isaaclab/_isaaclab_eureka/logs/rl_runs/skrl_eureka/sbtc_franka_unscrew_osc/reward_weight_tuning/randstart/2025-04-19_14-15-33_Run-0_iter-0"
log_dir = "/workspace/isaaclab/_isaaclab_eureka/logs/rl_runs/rsl_rl_eureka/sbtc_franka_unscrew_osc/reward_weight_tuning/randstart/2025-04-19_16-33-04_Run-0_iter-0"
data = load_tensorboard_logs(log_dir)

import re

def extract_relevant_metrics(data: dict, tuning_type: str = "reward_weight") -> list[tuple[str, list]]:
    import re
    assert tuning_type in {"reward_weight", "ppo_tuning"}, "Invalid tuning type"

    # Define patterns and preferred order
    if tuning_type == "reward_weight":
        ordered_patterns = [
            r".*Eureka/success_metric$",
            r".*Eureka/.+",
            r".*Episode_Reward/.+",
            r".*Reward / Total reward.+",
            r".*Reward / Instantaneous reward.+",
            r".*Train/mean_reward$",
        ]
    else:
        ordered_patterns = [
            r".*Eureka/success_metric$",
            r".*Loss\s*/\s*.+",
            r".*Policy\s*/\s*.+",
            r".*Learning rate",
            r".*Train/mean_reward$",
            r".*Train/mean_episode_length$",
            r".*Reward / Total reward.+",
            r".*Reward / Instantaneous reward.+",
        ]

    result = []
    seen = set()

    # Apply each pattern in order
    for pattern in ordered_patterns:
        for key in sorted(data.keys()):
            if key not in seen and re.match(pattern, key):
                result.append((key, data[key]))
                seen.add(key)

    return result


filtered_data = extract_relevant_metrics(data, tuning_type="reward_weight")

for name, values in filtered_data:
    print(f"{name}: len {len(values)}")