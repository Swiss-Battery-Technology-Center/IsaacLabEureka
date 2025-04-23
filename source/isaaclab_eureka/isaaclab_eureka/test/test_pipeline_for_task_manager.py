

import os
import time
from isaaclab_eureka.managers import EurekaTaskManager

# === Choose task and parameters ===
TASK = "SBTC-Lift-Cube-Franka-OSC-v0"  # change to match your task
EUREKA_TASK = "reward_weight_tuning"  
ENV_TYPE = "manager_based"
NUM_PROCESSES = 1
SUCCESS_STRING = ""  # unused for manager_based, but required arg
DUMMY_REWARD = ""  # not needed for this test, but required arg
NUM_ENVS=128
MAX_TRAINING_ITERATIONS = 50  # short run just for test
RL_LIBRARY = "rsl_rl"
PARAMETERS_TO_TUNE = [
     "algorithm.learning_rate",
     "algorithm.entropy_coef",
     "algorithm.desired_kl",
     "algorithm.clip_param",
     "algorithm.use_clipped_value_loss",
     "policy.init_noise_std",
]
class TestEurekaTaskManager:
    def __init__(self):
        self.tm = EurekaTaskManager(
        task=TASK,
        eureka_task=EUREKA_TASK,
        env_type=ENV_TYPE,
        rl_library=RL_LIBRARY,  # or "skrl" / "rl_games" if preferred
        num_processes=NUM_PROCESSES,
        success_metric_string=SUCCESS_STRING,
        max_training_iterations=MAX_TRAINING_ITERATIONS,  # short run just for test
        parameters_to_tune=PARAMETERS_TO_TUNE,
        )

    # === TEST 1: Save read_env_source_code_smart() output ===    
    def test_read_env_source_code_smart(self):
        print("[INFO] Saving env source code context...")
        env_text = self.tm._context_code_string
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(SCRIPT_DIR, "test_outputs")
        os.makedirs(output_dir, exist_ok=True)
        # Save env context file inside it
        with open(os.path.join(output_dir, "env_context.txt"), "w") as f:
            f.write(env_text)
        print("[✅] Saved environment context to test_outputs/env_context.txt")

    # === TEST 2: exception handling with reward strings: empty, duplicate, format error...
    def test_multiple_reward_strings(self):
        print("[INFO] Running dummy training...")
        # one normal, one with duplicates, one with wrong key
        # result should be: one SUCCESS, another SUCCESS, one FORMAT_ERROR, one SKIPIPED
        REWARD_STRING_LIST = [
            "{'reward.ee_orientation_alignment.weight': 1.0, 'reward.reaching_object.weight': 1.0, 'reward.lifting_object.weight': 10.0, 'reward.object_goal_tracking.weight': 0.0, 'reward.object_goal_stabilization.weight': 0.0, 'reward.action_rate.weight': -0.001, 'reward.gripper_action_rate.weight': -0.001, 'curriculum.object_goal_tracking.weight': 100.0, 'curriculum.object_goal_tracking.num_steps': 0, 'curriculum.reach_speed_stabilization.weight': 10.0, 'curriculum.reach_speed_stabilization.num_steps': 4800, 'curriculum.action_rate_high.weight': -0.1, 'curriculum.action_rate_high.num_steps': 12000, 'curriculum.reset_robot_joints.num_steps_start': 0, 'curriculum.reset_robot_joints.num_steps_end': 12000, 'curriculum.scale_gripper_velocity_limit.num_steps_start': 0, 'curriculum.scale_gripper_velocity_limit.num_steps_end': 12000}"

        ]
        # "{'ee_verticality': 0.8, 'yaw_alignment': 0.8, 'pos_similarity_coarse': 0.6, 'pos_similarity_medium': 0.9, 'pos_similarity_fine': 2.0, 'screw_engaged': 1.0, 'screw_contact': 1.5, 'table_contact': -2.0, 'action_rate': -0.03, 'joint_vel': -0.05} # for unscrew
        # "{'ee_orientation_alignment':1.0, 'reaching_object' = 1.0, 'lifting_object':10.0, 'object_goal_tracking': 0, 'object_goal_stabilization': 0, 'action_rate': -0.001, 'action_rate': -0.001, 'gripper_action_rate': -0.001}" # for lift
        results, _ = self.tm.train(REWARD_STRING_LIST)
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(SCRIPT_DIR, "test_outputs")
        os.makedirs(output_dir, exist_ok=True)
        # Save env context file inside it
        with open(os.path.join(output_dir, "multiple_reward_strings.txt"), "w") as f:
            for i, res in enumerate(results):
                f.write(f"Process {i}: {repr(res)}\n")
        print("[✅] Saved multiple reward strings result to test_outputs/multiple_reward_strings.txt")

    def test_initial_tunings(self):
        initial_tuning_string = self.tm._get_initial_tuning_as_string
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(SCRIPT_DIR, "test_outputs")
        os.makedirs(output_dir, exist_ok=True)
        # Save env context file inside it
        with open(os.path.join(output_dir, "initial_tuning_as_string.txt"), "a") as f:
            f.write(f'{self.tm._eureka_task}, {self.tm._task}, {self.tm._rl_library}\n')
            f.write(f'{initial_tuning_string}\n')
        print("[✅] Saved initial tuning to txt")
    
    def close(self):
        self.tm.close()
        print("[INFO] Closed task manager.")
        print("[INFO] TestEurekaTaskManager closed.")


if __name__ == "__main__":
    tetm = TestEurekaTaskManager()
    # tetm.some_test_function()
    # tetm.test_initial_tunings()
    tetm.test_multiple_reward_strings()

    tetm.close()
