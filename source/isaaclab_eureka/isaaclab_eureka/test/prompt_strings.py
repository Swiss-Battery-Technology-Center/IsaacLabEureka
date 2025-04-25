REWARD_TUNING_SYSTEM_PROMPT = '''
You are a reward engineer trying to tune the environment configuration to solve reinforcement learning tasks as effective as possible in Isaac Lab manager based environment.
Your goal is to suggest better configuration tuning for the environment, so that the agent will learn the task described in text faster.
You will be given the source code of the manager based environment, which includes rewards, curriculums, etc. 
From the source code, you must understand the overarching structure of the environment, and how different components work together to provide dense rewards to facilitate learning of the ultimate task.
For example, there might be easy reward terms to guide the agent towards ultimate success. There might be curriculum terms that gradually increase the difficulty of the task.
You will also be given the source code of success metric, which computes the ratio of environments that accomplished given task, according to our Eureka/success_metric.
Eventually, you want to achieve desired success metric.
You will be given training progress and the configuration used in training, such as reward and curriculum.
Leverage your understanding of the environment to analyze the training progress and suggest better configurations.

Your new configuraiton string should comply exactly with the structure of the previous configuration.
It will generally look like:
    {'reward.term_name.weight': value_1, 'curriculum.term_name.param_name': value_2, ...}
I will use regex pattern of the above structure to extract the keys and values from your response. 
Use the same keys as the previous configuration, but suggest new values.
A key, 'reward.term_name.weight' for example, is a string enclosed by a single quote. The dots inside are used to reconstruct a nested dictionary.
The value will be mostly float or int, but always comply with the type of the previous configuration.
Negative reward weights are posssible, terms with negative weights serve as penalty rather than reward.
Note that num_step values in curriculum is in units of simulation steps, which is 24 * learning iterations.
For example, if num_step_start is 4800, the curriculum starts at 4800/24 = 200 learning iterations.
When you suggest new num_step values, please make sure they are multiples of 24.
By common sense, 0 < num_step_start < num_step_end < 24 * max_learning_iterations.
You are not obliged to change all values. If a certain value was good in the previous run, you can keep it as it is.
If training is not going well, you are encouraged to make wild guesses.
The task is: Use Franka arm to approach a screw, engage and unscrew it.
I want to do evolutionary search for the configurations, so please provide 2 different suggestions for configurations tuning.

For ease of extraction, your respose should look like,

Suggestion 1
{'param_name_1': value_1, 'param_name_2': value_2, ...}

...

Suggestion N
{'param_name_1': value_1, 'param_name_2': value_2, ...}

After which you should add your analysis of the previous training run and explain why you think the new suggestions are better.
'''

CONTEXT_CODE = '''
Here is environment source code

##### === SBTC UNSCREW ENV CONFIG === #####

#################################################################
# Copyright (c) 2024 SBTC Switzerland Innovation Park Biel Bienne
# Author: Özhan Özen
# Email: oezhan.oezen@sipbb.ch, sbtc@sipbb.ch
# Created: 2024-06-04
#################################################################

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab_tasks.sbtc_tasks.manager_based.mdp as mdp
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Scene definition
##


@configclass
class SBTCUnscrewSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robot base
    robot_base: AssetBaseCfg = MISSING  # type: ignore

    # robots
    robot: ArticulationCfg = MISSING  # type: ignore

    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING  # type: ignore

    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING  # type: ignore

    # object sensor: will be populated by agent env cfg
    object_frame: FrameTransformerCfg = MISSING  # type: ignore

    # contact sensor at the end-effector: will be populated by agent env cfg
    # This goes the observations
    contact_sensor_ee: ContactSensorCfg = MISSING  # type: ignore

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING  # type: ignore


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 3
        ee_position = ObsTerm(
            func=mdp.ee_position_in_robot_root_frame,
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )
        # 3
        ee_rotation = ObsTerm(
            func=mdp.ee_delta_rotation_in_robot_root_frame,
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )
        # 6
        ee_velocity = ObsTerm(
            func=mdp.ee_velocity_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING)},  # type: ignore
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        # 6
        raw_actions = ObsTerm(func=mdp.last_action, params={"action_name": "arm_action"})
        # 6
        processed_actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "arm_action"})
        # 3
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        # 1
        object_yaw = ObsTerm(
            func=mdp.object_delta_yaw_in_robot_root_frame,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # 3
        interaction_forces_ee = ObsTerm(
            func=mdp.interaction_forces_prim_net,
            params={
                "sensor_cfg": SceneEntityCfg("contact_sensor_ee"),
            },
            noise=Unoise(n_min=-1.0, n_max=1.0),
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        ee_position = ObsTerm(
            func=mdp.ee_position_in_robot_root_frame,
        )
        ee_rotation = ObsTerm(
            func=mdp.ee_delta_rotation_in_robot_root_frame,
        )
        ee_velocity = ObsTerm(
            func=mdp.ee_velocity_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING)},  # type: ignore
        )
        raw_actions = ObsTerm(func=mdp.last_action, params={"action_name": "arm_action"})
        processed_actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "arm_action"})
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
        )
        object_yaw = ObsTerm(
            func=mdp.object_delta_yaw_in_robot_root_frame,
        )
        interaction_forces_ee_filtered = ObsTerm(
            func=mdp.interaction_forces_prim_filteredprims,
            params={
                "sensor_cfg": SceneEntityCfg("contact_sensor_ee"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""

    reset_object_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "yaw": (-torch.pi / 6, torch.pi / 6),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="object"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.randomize_ee_pose_wrt_object,  # FIXME Check the following error: Hint: Use either 'joint_names' or 'joint_ids' to avoid confusion.
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING, body_names=MISSING),  # type: ignore
            "object_cfg": SceneEntityCfg("object", body_names="object"),
            "ref_ee_object_pos_offset": (0.0, 0.0, 0.05),
            "unoise_ee_pos": (0.04, 0.04, 0.04),
            "unoise_ee_axisangle": (0.2, 0.2, 0.2),
        },
    )

    randomize_pd_gains = EventTerm(
        func=mdp.randomize_osc_pd_gains,
        mode="reset",
        params={
            "term_name": "arm_action",
            "p_range": (50.0, 150.0),
            "d_range": (0.25, 1.75),
        },
    )

    randomize_joint_effort_deadzones = EventTerm(
        func=mdp.randomize_osc_joint_effort_deadzones,
        mode="reset",
        params={
            "term_name": "arm_action",
            "deadzone_range": (0.0, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    ee_verticality = RewTerm(
        func=mdp.reward_axis_similarity_vector_ee,
        params={"target_quat": (0.0, 1.0, 0.0, 0.0)},
        weight=1.0, #4.0
    )
    yaw_alignment = RewTerm(
        func=mdp.reward_yaw_similarity_object_ee,
        params={"angle_repeats": 6},
        weight=3.0, #1.0
    )
    pos_similarity_coarse = RewTerm(
        func=mdp.reward_position_similarity_squash_object_ee,
        params={
            "sigma": 5.0,
            "object_offset_w": (0.0, 0.0, 0.006),
        },
        weight=0.75, # 0.25
    )
    pos_similarity_medium = RewTerm(
        func=mdp.reward_position_similarity_squash_object_ee,
        params={
            "sigma": 50.0,
            "object_offset_w": (0.0, 0.0, 0.006),
        },
        weight=1,# 0.5
    )
    pos_similarity_fine = RewTerm(
        func=mdp.reward_position_similarity_squash_object_ee,
        params={
            "sigma": 100.0,
            "object_offset_w": (0.0, 0.0, 0.006),
        },
        weight=1.5, #0.75
    )
    screw_engaged = RewTerm(
        func=mdp.reward_position_similarity_step_tanh_object_ee,
        params={
            "offset": 0.002,
            "sigma": 0.0007,
            "object_offset_w": (0.0, 0.0, 0.006),
        },
        weight=1.0, #1.0
    )

    screw_contact = RewTerm(
        func=mdp.is_contact_prim_filteredprim,
        weight=0.5, # 0.1
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor_ee"),
            "filtered_body_id": 0,
            "threshold": 2.5,
            "axis_idx": 2,
        },
    )
    table_contact = RewTerm(
        func=mdp.is_contact_prim_filteredprim,
        weight=-1.0, # -10.0
        params={"sensor_cfg": SceneEntityCfg("contact_sensor_ee"), "filtered_body_id": 1, "threshold": 1.0},
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05) # -0.005
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001, # -0.005
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},  # type: ignore
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # num_steps = num_steps_per_env (24) * training_step_to_change (e.g., 200)

    # reset_robot_joints = CurrTerm(
    #     func=mdp.modify_ee_pose_reset_progressively,
    #     params={
    #         "term_name": "reset_robot_joints",
    #         "ref_ee_object_pos_offset_start": (0.0, 0.0, 0.05),
    #         "ref_ee_object_pos_offset_end": (0.0, 0.0, 0.21),
    #         "unoise_ee_pos_start": (0.04, 0.04, 0.04),
    #         "unoise_ee_pos_end": (0.2, 0.2, 0.2),
    #         "unoise_ee_axisangle_start": (0.2, 0.2, 0.2),
    #         "unoise_ee_axisangle_end": (0.2, 0.2, 0.2),
    #         "num_steps_start": 500 * 24,  # 1100
    #         "num_steps_end": 600 * 24,  # 1300
    #     },
    # )

    reset_robot_joints = CurrTerm(
        func=mdp.modify_ee_pose_reset_progressively_with_performance,
        params={
            "term_name": "reset_robot_joints",
            "ref_ee_object_pos_offset_start": (0.0, 0.0, 0.05),
            "ref_ee_object_pos_offset_end": (0.0, 0.0, 0.21),
            "unoise_ee_pos_start": (0.04, 0.04, 0.04),
            "unoise_ee_pos_end": (0.2, 0.2, 0.2),  # TODO Check if z range is too much (might collide with the table)
            "unoise_ee_axisangle_start": (0.2, 0.2, 0.2),
            "unoise_ee_axisangle_end": (0.2, 0.2, 0.2),
            "performance_reward_term_name": "screw_engaged",
            "performance_low": 0.5,
            "performance_high": 0.6,
            "num_steps_enable": 100 * 24, # 0
            "num_steps_disable": 900 * 24, # 600
        },
    )

    object_offset = CurrTerm(
        func=mdp.modify_reward_object_offset_progressively,
        params={
            "term_name_list": (
                "pos_similarity_coarse",
                "pos_similarity_medium",
                "pos_similarity_fine",
                "screw_engaged",
            ),
            "object_offset_w_start": (0.0, 0.0, 0.006),
            "object_offset_w_end": (0.0, 0.0, 0.0),
            "num_steps_start": 500 * 24,  # 600
            "num_steps_end": 600 * 24,  # 700
        },
    )

    randomize_joint_effort_deadzones = CurrTerm(
        func=mdp.modify_deadzone_reset_progressively,
        params={
            "term_name": "randomize_joint_effort_deadzones",
            "deadzone_range_start": (0.0, 0.1),
            "deadzone_range_end": (0.0, 0.5),
            "num_steps_start": 700 * 24, # 1000
            "num_steps_end": 800 * 24, # 1100
        },
    )

    # object_offset = CurrTerm(
    #     func=mdp.modify_reward_object_offset_progressively_with_performance,
    #     params={
    #         "term_name_list": (
    #             "pos_similarity_coarse",
    #             "pos_similarity_medium",
    #             "pos_similarity_fine",
    #             "screw_engaged",
    #         ),
    #         "object_offset_w_start": (0.0, 0.0, 0.006),
    #         "object_offset_w_end": (0.0, 0.0, 0.0),
    #         "performance_reward_term_name": "screw_engaged",
    #         "performance_low": 0.3,
    #         "performance_high": 0.5,
    #         "num_steps_enable": 800 * 24,
    #         "num_steps_disable": 2000 * 24,
    #     },
    # )


##
# Environment configuration
##


@configclass
class SBTCUnscrewEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: SBTCUnscrewSceneCfg = SBTCUnscrewSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        # Viewer settings
        self.viewer.eye = (0.125, 0.0, 0.05)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "object"

        # Factory settings ##################

        self.sim.physics_material = RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        )

        self.sim.physx.max_position_iteration_count = 192  # Important to avoid interpenetration.
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_max_num_partitions = 1  # Important for stable simulation.
        #######################################

        # Custom settings
        self.sim.physx.gpu_collision_stack_size = 2**31
        # self.sim.physx.enable_ccd = True  # TODO Check if this is needed
        # self.sim.physx.gpu_found_lost_pairs_capacity *= 1
        # self.sim.physx.gpu_found_lost_aggregate_pairs_capacity *= 1
        # self.sim.physx.gpu_total_aggregate_pairs_capacity *= 1
        # self.sim.physx.gpu_heap_capacity *= 1
        # self.sim.physx.gpu_temp_buffer_capacity *= 1
        # self.sim.physx.gpu_max_soft_body_contacts *= 1
        # self.sim.physx.gpu_max_particle_contacts *= 1



##### === REWARDS FUNCTIONS === #####

=== action_rate_l2 ===
# === isaaclab.envs.mdp.rewards.action_rate_l2 ===
def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)



=== is_contact_prim_filteredprim ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.rewards.is_contact_prim_filteredprim ===
def is_contact_prim_filteredprim(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    filtered_body_id: int,
    threshold: float = 1.0,
    axis_idx: int | slice | None = None,
) -> torch.Tensor:
    """Reward or penalize desired contacts as the number of forces that are above a threshold.

    Args:
        env: The environment instance.
        sensor_cfg: The configuration of the contact sensor.
        filtered_body_id: The index of the filtered body to extract the contact forces,
            according to the filter_prim_paths_expr argument of the sensor_cfg.
        threshold: The threshold for the contact force.
        axis_idx: The index or slice of the axes along which to check the contact force.
            Defaults to None (selects all axes).

    Returns:
        A tensor containing the reward/penalty for each environment instance.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    # Default axis_idx to slice(None) if None
    if axis_idx is None:
        axis_idx = slice(None)  # Select all axes
    contact_forces = contact_sensor.data.force_matrix_w[:, sensor_cfg.body_ids, filtered_body_id, axis_idx]  # type: ignore
    if isinstance(axis_idx, slice):
        # Handle slicing (e.g., slice(0, 3))
        contact_forces = torch.norm(contact_forces, dim=-1)
    is_contact = contact_forces > threshold
    return torch.sum(is_contact, dim=1)



=== joint_vel_l2 ===
# === isaaclab.envs.mdp.rewards.joint_vel_l2 ===
def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)



=== reward_axis_similarity_vector_ee ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.rewards.reward_axis_similarity_vector_ee ===
def reward_axis_similarity_vector_ee(
    env: ManagerBasedRLEnv,
    target_quat: tuple[float, float, float, float],
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for aligning a specific axis of end-effector orientation to a target quaternion.

    Args:
        env: The environment instance.
        target_quat: The target orientation as a quaternion (shape: [4] in w, x, y, z order).
        ee_frame_cfg: End-effector frame configuration.

    Returns:
        A tensor containing the alignment reward for each environment instance.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]

    target_quat_tensor = torch.tensor(target_quat, device=ee_quat_w.device).expand_as(ee_quat_w)

    similarity = sbtc_utils.axis_orientation_similarity(ee_quat_w, target_quat_tensor)
    return similarity

# === sbtc_utils.axis_orientation_similarity ===
def axis_orientation_similarity(ori1: torch.Tensor, ori2: torch.Tensor, axis: int = 2) -> torch.Tensor:
    """Normalized similarity with regard to a specific axis of two orientations.

    Args:
        ori1: Quaternion tensor (shape: [N, 4]) in [w, x, y, z] format.
        ori2: Quaternion tensor (shape: [N, 4]) in [w, x, y, z] format.
        axis: The axis to compute the similarity for.

    Returns:
        Between 0 and 1, where 0 is the minimum axis alignment and 1 is the maximum.
    """
    # Convert quaternions to rotation matrices
    R1 = matrix_from_quat(ori1)
    R2 = matrix_from_quat(ori2)

    # Extract the specific axis of both rotation matrices
    a1 = R1[:, axis, :]
    a2 = R2[:, axis, :]

    # Compute alignment similarity between 0 and 1
    dot_product = torch.sum(a1 * a2, dim=1)
    dot_product = torch.clamp(dot_product, -1, 1)
    alignment_error = 1 - dot_product
    alignment_error *= 0.5
    alignment_similarity = torch.clamp(1 - alignment_error, 0, 1)
    return alignment_similarity


=== reward_position_similarity_squash_object_ee ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.rewards.reward_position_similarity_squash_object_ee ===
def reward_position_similarity_squash_object_ee(
    env: ManagerBasedRLEnv,
    sigma: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_offset_w: Sequence[float] = (0.0, 0.0, 0.0),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the similarity between the object and the end-effector position using a squash kernel.

    Args:
        env: The environment instance.
        sigma: The smoothing factor a for the squash kernel.
        object_cfg: The configuration of the object.
        object_offset_w: The offset to apply to the object position.
        ee_frame_cfg: The configuration of the end-effector frame.

    Returns:
        The position similarity reward between 0 and 1.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w + torch.tensor(object_offset_w, device=object.data.root_pos_w.device)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    similarity = sbtc_utils.vector_similarity_squash(ee_w, object_pos_w, sigma)
    return similarity

# === sbtc_utils.vector_similarity_squash ===
def vector_similarity_squash(vec1: torch.Tensor, vec2: torch.Tensor, a: float, b: float = 0) -> torch.Tensor:
    """Compute a normalized similarity between two vectors.

    The difference is normalized by the smoothing factor and then passed through a squashing function.

    Args:
        vec1: The first tensor.
        vec2: The second tensor.
        a: The smoothing factor a.
        b: The smoothing factor b.

    Returns:
        Between 0 and 1, where 0 is the minimum similarity and 1 is the maximum.

    Note:
        Squash function has the shape r(x) = ( 1 / (exp(-ax) + b + exp(ax)) ) * (2 + b).
            It is modified from Appendix B of https://arxiv.org/pdf/2408.04587.
    """
    diff = torch.norm(vec1 - vec2, dim=1)
    return (1 / (torch.exp(a * diff) + b + torch.exp(-a * diff))) * (b + 2)


=== reward_position_similarity_step_tanh_object_ee ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.rewards.reward_position_similarity_step_tanh_object_ee ===
def reward_position_similarity_step_tanh_object_ee(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_offset_w: Sequence[float] = (0.0, 0.0, 0.0),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    offset: float = 0.002,
    sigma: float = 0.0007,
) -> torch.Tensor:
    """Reward the similarity between the object and the end-effector position with an offset step using a tanh kernel.

    The reward goes to 0 when the difference is higher than the offset, and it goes to one when lower than the offset.
    The default values are tune for the screw engagement (goes from 0 to 1 as the screw is engaged from the
    surface contact).

    Args:
        env: The environment instance.
        object_cfg: The configuration of the object.
        object_offset_w: The offset to apply to the object position.
        ee_frame_cfg: The configuration of the end-effector frame.
        offset: The offset for the step function (where the step will be at 50%).
        sigma: The smoothing factor for the tanh kernel.

    Returns:
        The position similarity reward between 0 and 1.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w + torch.tensor(object_offset_w, device=object.data.root_pos_w.device)
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    similarity = 1 - sbtc_utils.vector_difference_step_tanh(ee_pos_w, object_pos_w, offset, sigma)
    return similarity

# === sbtc_utils.vector_difference_step_tanh ===
def vector_difference_step_tanh(
    vec1: torch.Tensor, vec2: torch.Tensor, offset: float = 0.5, sigma: float = 0.01
) -> torch.Tensor:
    """Smooth step based on the difference between two vectors.

    The step goes from 0 to 1 as the difference between the two vectors increases.

    Args:
        vec1: The first tensor.
        vec2: The second tensor.
        offset: The offset for the step function (where the step will be at 50%).
        sigma: The smoothing factor (how steep the transition is).

    Returns:
        Value between 0 and 1.
    """
    diff = torch.norm(vec1 - vec2, dim=1)
    step = (torch.tanh((diff - offset) / sigma) + 1.0) * 0.5
    return step


=== reward_yaw_similarity_object_ee ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.rewards.reward_yaw_similarity_object_ee ===
def reward_yaw_similarity_object_ee(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    angle_repeats: int = 1,
    flip_ee: bool = True,
) -> torch.Tensor:
    """Reward the similarity between the object and the end-effector delta yaw angle.

    Args:
        env: The environment instance.
        object_cfg: The configuration of the object.
        ee_frame_cfg: The configuration of the end-effector frame.
        angle_repeats: The number of times the angle repeats (e.g., 6 for 60 degrees).
        flip_ee: Whether to flip the end-effector orientation upside down.

    Returns:
        The normalized yaw similarity between the object and the end-effector, between 0 and 1.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w
    object_quat_w = object.data.root_quat_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]

    unit_pos = torch.zeros_like(object_pos_w)
    unit_quad_obj = torch.zeros_like(object_quat_w)
    unit_quad_obj[:, 0] = 1.0
    if flip_ee:
        unit_quad_ee = torch.zeros_like(object_quat_w)
        unit_quad_ee[:, 1] = 1.0
    else:
        unit_quad_ee = unit_quad_obj

    _, object_deltayaw = compute_pose_error(unit_pos, unit_quad_obj, object_pos_w, object_quat_w)
    object_deltayaw = object_deltayaw[..., 2]

    _, ee_deltayaw = compute_pose_error(unit_pos, unit_quad_ee, ee_pos_w, ee_quat_w)
    ee_deltayaw = ee_deltayaw[..., 2]

    repeat_angle = (2.0 * torch.pi) / angle_repeats
    raw_diff = object_deltayaw - ee_deltayaw
    wrapped_diff = (raw_diff + (0.5 * repeat_angle)) % repeat_angle - (0.5 * repeat_angle)
    diff = torch.abs(wrapped_diff) / (0.5 * repeat_angle)
    similarity = 1 - diff

    return similarity




##### === EVENTS FUNCTIONS === #####

=== randomize_ee_pose_wrt_object ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.events.randomize_ee_pose_wrt_object ===
def randomize_ee_pose_wrt_object(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ref_ee_object_pos_offset: Sequence[float] = [0.0, 0.0, 0.0],
    ref_ee_quat: Sequence[float] = [0.0, 1.0, 0.0, 0.0],
    unoise_ee_pos: Sequence[float] = [0.05, 0.05, 0.05],
    unoise_ee_axisangle: Sequence[float] = [0.05, 0.05, 0.05],
    ik_iterations: int = 10,
    lambda_val: float = 0.01,
):
    """
    Reset the robot's joint positions using iterative inverse kinematics (IK) to align the end-effector
    with a target pose relative to a specified object.

    ATTENTION: The IK solver may fail if the robot's initial position (as defined in
    env.scene[asset_cfg.name].data.default_joint_pos) is too far from the target pose.
    Convergence is currently NOT checked in this reset function.

    Args:
        env: The environment.
        env_ids: The environment IDs.
        asset_cfg: Configuration for the robot. Make sure to set the body_names (end-effector) and the joint_names.
        object_cfg: Configuration for the target object.
        ref_ee_object_pos_offset: The offset from the object's body frame to the end-effector's reference pose.
        ref_ee_quat: The reference quaternion for the end-effector's orientation.
        unoise_ee_pos: The range for the end-effector's position noise.
        unoise_ee_axisangle: The range for the end-effector's orientation noise, as axis angle.
        ik_iterations: The number of IK iterations.
        lambda_val: The damping factor for the differential IK solver.
    """
    # extract the used quantities
    robot: Articulation = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    device = robot.device
    ee_jacobian_index = asset_cfg.body_ids[0] - 1  # type: ignore

    # Get poses: _w denotes world frame
    root_pos_w = robot.data.root_pos_w[env_ids]
    root_quat_w = robot.data.root_quat_w[env_ids]
    object_pos_w = object.data.body_link_pos_w[env_ids, object_cfg.body_ids, :].squeeze(1)
    # object_quat_w = object.data.body_link_quat_w[env_ids, object_cfg.body_ids, :].squeeze(1)
    root_pos_w = robot.data.root_pos_w[env_ids]
    root_quat_w = robot.data.root_quat_w[env_ids]

    # Compute target pose
    shape = (len(env_ids), 3)
    pos_unoise_lower = -torch.tensor(unoise_ee_pos, device=device)
    pos_unoise_upper = torch.tensor(unoise_ee_pos, device=device)
    pos_unoise = math_utils.sample_uniform(pos_unoise_lower, pos_unoise_upper, shape, device=device)
    ori_unoise_lower = -torch.tensor(unoise_ee_axisangle, device=device)
    ori_unoise_upper = torch.tensor(unoise_ee_axisangle, device=device)
    ori_unoise = math_utils.sample_uniform(ori_unoise_lower, ori_unoise_upper, shape, device=device)
    unoise = torch.cat([pos_unoise, ori_unoise], dim=-1)

    ref_ee_pos_w = object_pos_w + torch.tensor(ref_ee_object_pos_offset, device=device)
    ref_ee_quat_w = torch.tensor(ref_ee_quat, device=device).view(1, -1).expand(*ref_ee_pos_w.shape[:1], -1)
    target_ee_pos_w, target_ee_quat_w = apply_delta_pose(ref_ee_pos_w, ref_ee_quat_w, unoise)
    target_ee_pos_b, target_ee_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, target_ee_pos_w, target_ee_quat_w
    )
    # target_ee_pose_b = torch.cat([target_ee_pos_b, target_ee_quat_b], dim=-1)

    # Solve IK iteratively
    for i in range(ik_iterations):

        # Reset to default (hopefully close to target pose) joint positions
        if i == 0:
            target_joint_pos = robot.data.default_joint_pos.clone()[env_ids][:, asset_cfg.joint_ids]

        else:
            # Robot states (only consider joints that contribute to end-effector movements)
            joint_pos = robot.data.joint_pos[env_ids][:, asset_cfg.joint_ids]
            ee_pos_w = robot.data.body_link_pos_w[env_ids, asset_cfg.body_ids, :].squeeze(1)
            ee_quat_w = robot.data.body_link_quat_w[env_ids, asset_cfg.body_ids, :].squeeze(1)
            ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

            # Compute error
            # TODO: Use this to check convergence and early termination
            # pos_error, rot_error = compute_pose_error(
            #     ee_pos_b, ee_quat_b, target_ee_pos_b, target_ee_quat_b)
            # pos_error_norm = torch.norm(pos_error, dim=-1)
            # rot_error_norm = torch.norm(rot_error, dim=-1)

            # Compute Jacobian in body  frame
            jacobian_w = robot.root_physx_view.get_jacobians()[env_ids][:, ee_jacobian_index, :, asset_cfg.joint_ids]
            base_rot_matrix = matrix_from_quat(quat_inv(root_quat_w))
            jacobian = jacobian_w.clone()
            jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
            jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)

            # Solve differential IK step with dls
            position_error, axis_angle_error = compute_pose_error(
                ee_pos_b, ee_quat_b, target_ee_pos_b, target_ee_quat_b, rot_error_type="axis_angle"
            )
            pose_error = torch.cat((position_error, axis_angle_error), dim=1)
            lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=device)
            delta_joint_pos = (
                jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ pose_error.unsqueeze(-1)
            )
            delta_joint_pos = delta_joint_pos.squeeze(-1)
            target_joint_pos = joint_pos + delta_joint_pos
            joint_pos_limits = robot.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
            target_joint_pos = target_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

        # Set into the physics simulation
        robot.write_joint_state_to_sim(
            target_joint_pos,
            torch.zeros_like(target_joint_pos, device=device),
            joint_ids=asset_cfg.joint_ids,
            env_ids=env_ids,  # type: ignore
        )

    # Check final error
    # pos_error, rot_error = compute_pose_error(ee_pos_b, ee_quat_b, target_ee_pos_b, target_ee_quat_b)
    # pos_error_norm = torch.norm(pos_error, dim=-1)
    # rot_error_norm = torch.norm(rot_error, dim=-1)
    # print(f"Final pos error: {pos_error_norm.mean():.6f} , Final rot error: {rot_error_norm.mean():.6f}")



=== randomize_osc_joint_effort_deadzones ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.events.randomize_osc_joint_effort_deadzones ===
def randomize_osc_joint_effort_deadzones(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    deadzone_range: tuple[float, float],
):
    """
    Randomizes the joint effort deadzones of the operational space controller action in the action manager.

    Args:
        env: The ManagerBasedRLEnv instance, required to access the action manager.
        env_ids: The environment IDs.
        term_name: The name of the term in the action manager.
        deadzone_range: The range for the deadzones.
    """

    osc: OperationalSpaceControllerActionFilteredDeadzone = env.action_manager.get_term(term_name)  # type: ignore
    deadzone = math_utils.sample_uniform(*deadzone_range, (env.num_envs, osc._num_DoF, 2), env.device)  # type: ignore
    osc.modify_joint_effort_deadzone(deadzone)



=== randomize_osc_pd_gains ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.events.randomize_osc_pd_gains ===
def randomize_osc_pd_gains(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    p_range: tuple[float, float] | None = None,
    d_range: tuple[float, float] | None = None,
):
    """
    Randomizes the PD gains of the operational space controller action in the action manager.

    Args:
        env: The ManagerBasedRLEnv instance, required to access the action manager.
        env_ids: The environment IDs.
        term_name: The name of the term in the action manager.
        p_range: The range for the proportional gains. If None, the proportional gains are kept unchanged.
        d_range: The range for the derivative gains. If None, the derivative gains are kept unchanged.
    """
    p_values = math_utils.sample_uniform(*p_range, (env.num_envs, 6), env.device) if p_range is not None else None
    d_values = math_utils.sample_uniform(*d_range, (env.num_envs, 6), env.device) if d_range is not None else None

    osc: OperationalSpaceControllerActionFiltered = env.action_manager.get_term(term_name)  # type: ignore
    osc.modify_pd_gains(p_values, d_values)



=== reset_root_state_uniform ===
# === isaaclab.envs.mdp.events.reset_root_state_uniform ===
def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)




##### === CURRICULUM FUNCTIONS === #####

=== modify_deadzone_reset_progressively ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.curriculums.modify_deadzone_reset_progressively ===
def modify_deadzone_reset_progressively(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    deadzone_range_start: Sequence[float],
    deadzone_range_end: Sequence[float],
    num_steps_start: int = 0,
    num_steps_end: int = 1000,
    num_steps_per_env: int = 24,
):
    """
    Curriculum that modifies the joint effort deadzone range given a progress (based on the current step counter).

    It linearly interpolates between deadzone_range_start and deadzone_range_end.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the event term.
        deadzone_range_start: The initial range for deadzone randomization.
        deadzone_range_end: The final range for deadzone randomization.
        num_steps_start: The step count at which interpolation starts.
        num_steps_end: The step count at which interpolation ends.
        num_steps_per_env: The number of steps per environment. Logging purposes.
    """

    # Only modify ranges if we're within the specified curriculum window
    if num_steps_start <= env.common_step_counter <= num_steps_end:

        term_cfg = env.event_manager.get_term_cfg(term_name)  # type: ignore

        # Compute progress in [0, 1]
        progress = (env.common_step_counter - num_steps_start) / float(abs(num_steps_end - num_steps_start))

        deadzone_range = [
            start + progress * (end - start) for start, end in zip(deadzone_range_start, deadzone_range_end)
        ]

        # Update and set the new pose range
        term_cfg = env.event_manager.get_term_cfg(term_name)  # type: ignore
        term_cfg.params["deadzone_range"] = deadzone_range  # type: ignore
        env.event_manager.set_term_cfg(term_name, term_cfg)

        if env.common_step_counter % num_steps_per_env == 0:
            print(f"[Curriculum]: {term_name} -> progress: {progress}\n\t\tdeadzone_range: {deadzone_range}")



=== modify_ee_pose_reset_progressively_with_performance ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.curriculums.modify_ee_pose_reset_progressively_with_performance ===
def modify_ee_pose_reset_progressively_with_performance(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    term_name: str,
    ref_ee_object_pos_offset_start: Sequence[float],
    ref_ee_object_pos_offset_end: Sequence[float],
    unoise_ee_pos_start: Sequence[float],
    unoise_ee_pos_end: Sequence[float],
    unoise_ee_axisangle_start: Sequence[float],
    unoise_ee_axisangle_end: Sequence[float],
    performance_reward_term_name: str,
    performance_low: int,
    performance_high: int,
    num_steps_enable: int = 0,
    num_steps_disable: int | float = float("inf"),
    num_steps_per_env: int = 24,
):
    """
    Curriculum that modifies the range for ee pose randomization given progressively based on a performance value.

    It linearly interpolates between pose_range_start and pose_range_end.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the event term.
        ref_ee_object_pos_offset_start: The initial range for pose randomization.
        ref_ee_object_pos_offset_end: The final range for pose randomization.
        unoise_ee_pos_start: The initial range for pose randomization noise.
        unoise_ee_pos_end: The final range for pose randomization noise.
        unoise_ee_axisangle_start: The initial range for orientation randomization noise as axis-angle.
        unoise_ee_axisangle_end: The final range for orientation randomization noise as axis-angle.
        performance_reward_term_name: The name of the reward term to track the performance.
        performance_low: The lower performance level to start the curriculum.
        performance_high: The upper performance level to end the curriculum.
        num_steps_enable: The step count at which curriculum is enabled.
        num_steps_disable: The step count at which curriculum is disabled.
        num_steps_per_env: The number of steps per environment. Logging purposes.
    """

    if env.common_step_counter >= num_steps_enable:

        if "log" in env.extras and ("Episode_Reward/" + performance_reward_term_name) in env.extras["log"]:
            performance = env.extras["log"][("Episode_Reward/" + performance_reward_term_name)].item()
        else:
            performance = 0

        performance = min(max(performance, performance_low), performance_high)
        progress = (performance - performance_low) / float(abs(performance_high - performance_low))

        if env.common_step_counter >= num_steps_disable:
            progress = 1.0

        ref_ee_object_pos_offset = [
            start + progress * (end - start)
            for start, end in zip(ref_ee_object_pos_offset_start, ref_ee_object_pos_offset_end)
        ]

        unoise_ee_pos = [start + progress * (end - start) for start, end in zip(unoise_ee_pos_start, unoise_ee_pos_end)]

        unoise_ee_axisangle = [
            start + progress * (end - start) for start, end in zip(unoise_ee_axisangle_start, unoise_ee_axisangle_end)
        ]

        # Update and set the new pose range
        term_cfg = env.event_manager.get_term_cfg(term_name)  # type: ignore
        term_cfg.params["ref_ee_object_pos_offset"] = ref_ee_object_pos_offset  # type: ignore
        term_cfg.params["unoise_ee_pos"] = unoise_ee_pos  # type: ignore
        term_cfg.params["unoise_ee_axisangle"] = unoise_ee_axisangle
        env.event_manager.set_term_cfg(term_name, term_cfg)

        if env.common_step_counter % num_steps_per_env == 0 and progress != 1.0:
            print(
                f"[Curriculum]: {term_name} -> progress: {progress}"
                f"\n\t\tref_ee_object_pos_offset: {ref_ee_object_pos_offset}"
                f"\n\t\tunoise_ee_pos: {unoise_ee_pos}\n\t\tunoise_ee_axisangle: {unoise_ee_axisangle}\n"
            )



=== modify_reward_object_offset_progressively ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.curriculums.modify_reward_object_offset_progressively ===
def modify_reward_object_offset_progressively(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name_list: Sequence[str],
    object_offset_w_start: Sequence[float],
    object_offset_w_end: Sequence[float],
    num_steps_start: int,
    num_steps_end: int,
    num_steps_per_env: int = 24,
):
    """Curriculum that modifies the object offset for all reward terms on the list, based on the progress.

    It linearly interpolates between object_offset_start and object_offset_end.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name_list: The list of term names.
        object_offset_w_start: The initial object offset, in world frame.
        object_offset_w_ end: The final object offset, in world frame.
        num_steps_start: The step count at which interpolation starts.
        num_steps_end: The step count at which interpolation ends.
        num_steps_per_env: The number of steps per environment. Logging purposes.
    """

    if num_steps_start <= env.common_step_counter <= num_steps_end:

        progress = (env.common_step_counter - num_steps_start) / float(abs(num_steps_end - num_steps_start))

        object_offset_w = [
            start + progress * (end - start) for start, end in zip(object_offset_w_start, object_offset_w_end)
        ]

        for term_name in term_name_list:
            term_cfg = env.reward_manager.get_term_cfg(term_name)  # type: ignore
            term_cfg.params["object_offset_w"] = object_offset_w
            env.reward_manager.set_term_cfg(term_name, term_cfg)

        if env.common_step_counter % num_steps_per_env == 0:
            print(f"[Curriculum]: {term_name_list} object_offset: {object_offset_w}")




##### === TERMINATIONS FUNCTIONS === #####

=== time_out ===
# === isaaclab.envs.mdp.terminations.time_out ===
def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length




##### === OBSERVATIONS FUNCTIONS === #####

=== ee_delta_rotation_in_robot_root_frame ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.observations.ee_delta_rotation_in_robot_root_frame ===
def ee_delta_rotation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The rotation of the end-effector in the robot's root frame.

    The rotation is represented as a delta axis-angle wrt unit quad.

    Args:
        env: The environment.
        robot_cfg: The robot configuration.
        ee_frame_cfg: The end-effector frame configuration.

    Returns:
        The rotation of the end-effector in the robot's root frame, represented as a delta axis-angle.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]

    unit_pos = torch.zeros_like(ee_pos_w)
    unit_rot = torch.zeros_like(ee_quat_w)
    unit_rot[:, 1] = 1.0

    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        ee_pos_w,
        ee_quat_w,
    )
    ee_quat_b = normalize(ee_quat_b)
    # ee_quat_b = quat_unique(ee_quat_b)

    _, ee_rot_delta = compute_pose_error(unit_pos, unit_rot, ee_pos_b, ee_quat_b)

    return ee_rot_delta



=== ee_position_in_robot_root_frame ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.observations.ee_position_in_robot_root_frame ===
def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The position of the end-effector in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: The robot configuration.
        ee_frame_cfg: The end-effector frame configuration.

    Returns:
        The position of the end-effector in the robot's root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_pos_b, _ = subtract_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_pos_w)
    return ee_pos_b



=== ee_velocity_in_robot_root_frame ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.observations.ee_velocity_in_robot_root_frame ===
def ee_velocity_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The velocity of the end-effector in the robot's root frame.

    Args:
        env: The environment.
        asset_cfg: The robot configuration.

    Returns:
        The velocity of the end-effector in the robot's root frame.
    """
    robot: RigidObject = env.scene[asset_cfg.name]

    # asset_cfg.body_ids[0] should correspond to the end-effector body
    ee_vel_w = robot.data.body_vel_w[:, asset_cfg.body_ids[0], :]  # type: ignore
    relative_vel_w = ee_vel_w - robot.data.root_vel_w

    # Convert ee velocities from world to root frame
    ee_vel_linear_b = quat_rotate_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])
    ee_vel_angular_b = quat_rotate_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])

    ee_vel_b = torch.cat([ee_vel_linear_b, ee_vel_angular_b], dim=-1)
    return ee_vel_b



=== interaction_forces_prim_filteredprims ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.observations.interaction_forces_prim_filteredprims ===
def interaction_forces_prim_filteredprims(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Interaction forces between all the filtered objects and the toolhead, projected to the contact sensor."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    contact_forces = contact_sensor.data.force_matrix_w[:, sensor_cfg.body_ids, :, :]  # type: ignore
    contact_forces = contact_forces.squeeze(1).flatten(1)
    return contact_forces



=== interaction_forces_prim_net ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.observations.interaction_forces_prim_net ===
def interaction_forces_prim_net(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """The net interaction force at the toolhead projected to the contact sensor."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # type: ignore
    # contact_forces = torch.mean(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :], dim=1)
    contact_forces.squeeze_(1)
    return contact_forces



=== last_action ===
# === isaaclab.envs.mdp.observations.last_action ===
def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions



=== last_processed_action ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.observations.last_processed_action ===
def last_processed_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input processed action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).processed_actions



=== object_delta_yaw_in_robot_root_frame ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.observations.object_delta_yaw_in_robot_root_frame ===
def object_delta_yaw_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The yaw angle of the object in the robot's root frame.

    The yaw angle is represented as a delta axis-angle component wrt unit quad.

    Args:
        env: The environment.
        robot_cfg: The robot configuration.
        object_cfg: The object configuration.

    Returns:
        The yaw angle of the object in the robot's root frame, represented as a delta axis-angle component.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_rot_w = object.data.root_quat_w[:, :]

    unit_pos = torch.zeros_like(object_pos_w)
    unit_rot = torch.zeros_like(object_rot_w)
    unit_rot[:, 0] = 1.0

    object_pos_b, object_rot_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w, object_rot_w
    )
    object_rot_b = normalize(object_rot_b)

    _, object_rot_delta = compute_pose_error(unit_pos, unit_rot, object_pos_b, object_rot_b)
    delta_yaw = object_rot_delta[..., 2].unsqueeze(-1)

    return delta_yaw



=== object_position_in_robot_root_frame ===
# === isaaclab_tasks.sbtc_tasks.manager_based.mdp.observations.object_position_in_robot_root_frame ===
def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: The robot configuration.
        object_cfg: The object configuration.

    Returns:
        The position of the object in the robot's root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_rot_w = object.data.root_quat_w[:, :]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w, object_rot_w
    )
    return object_pos_b
'''
SUCCESS_METRIC_CODE = '''
# task success metric for unscrew task
import torch
def compute_success_metric(self, env_ids):
    object = self.scene["object"]
    ee_frame = self.scene["ee_frame"]

    # 🔹 Retrieve current reward term parameters
    screw_engaged_cfg = self.reward_manager.get_term_cfg("screw_engaged")
    offset = screw_engaged_cfg.params["offset"]
    object_offset_w = torch.tensor(
        screw_engaged_cfg.params["object_offset_w"], device=object.data.root_pos_w.device
    )

    # 🔹 Positions
    object_pos_w = object.data.root_pos_w + object_offset_w             # (num_envs, 3)
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]                   # (num_envs, 3)

    # 🔹 Distance between EE and screw position
    dist = torch.linalg.norm(ee_pos_w[env_ids] - object_pos_w[env_ids], dim=1)

    # 🔹 Binary success: ee is within offset distance to screw.
    success = dist < offset
    return {"success_metric": success.float().mean(), 
            "dist": dist.mean(), 
            'object_offset': object_offset_w.mean()}
'''

PPO_CODE = '''
from typing import Any, Mapping, Optional, Tuple, Union

import copy
import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR


# fmt: off
# [start-config-dict-torch]
PPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class PPO(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Proximal Policy Optimization (PPO)

        https://arxiv.org/abs/1707.06347

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.policy.parameters(), self.value.parameters()), lr=self._learning_rate
                )
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            # tensors sampled during training
            self._tensors_names = ["states", "actions", "log_prob", "values", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")
            self._current_log_prob = log_prob

        return actions, log_prob, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
                values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob,
                    values=values,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

'''