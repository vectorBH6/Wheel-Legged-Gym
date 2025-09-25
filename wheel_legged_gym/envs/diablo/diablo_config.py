# ==============================================================================
# Author: Zhengkr
# Created on: 2024-03-11
# Description: 双轮足机器人刑天训练配置
# Copyright: Optional, add copyright information if needed.
# Revision History:
#   YYYY-MM-DD: Made modifications, updated XX feature.
# ==============================================================================

from wheel_legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class DiabloCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 1024

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.15]  # x,y,z [m]
        # todo设置关节初始位置 以及 PD参数
        # 参考传统控制
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # 'hip_left':  0.,
            "hip2_left": 0.0,
            "knee_left": 0.0,
            "ankle_left": 0.0,
            # 'hip_right': 0.,
            "hip2_right": 0.0,
            "knee_right": 0.0,
            "ankle_right": 0.0,
        }

    class control(LeggedRobotCfg.control):
        pos_action_scale = 0.5
        vel_action_scale = 10.0
        # PD Drive parameters:
        stiffness = {"hip": 80.0, "knee": 80.0, "ankle": 0}  # [N*m/rad]
        damping = {"hip": 5.0, "knee": 5.0, "ankle": 2.5}  # [N*m*s/rad]
        
    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/diablo/urdf/diablo.urdf"
        name = "diablo"
        foot_name = "wheel"
        penalize_contacts_on = ["base", "leg"]
        terminate_after_contacts_on = ["base", "leg"]  # 防止跪刹
        flip_visual_attachments = False
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_restitution = False
        restitution_range = [0.0, 1.0]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]
        randomize_inertia = False
        randomize_inertia_range = [0.8, 1.2]
        randomize_base_com = False
        rand_com_vec = [0.05, 0.05, 0.05]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0  # default: 1.
        randomize_Kp = False
        randomize_Kp_range = [0.9, 1.1]
        randomize_Kd = False
        randomize_Kd_range = [0.9, 1.1]
        randomize_motor_torque = False
        randomize_motor_torque_range = [0.9, 1.1]
        randomize_default_dof_pos = False
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = False
        delay_ms_range = [0, 10]

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 2.0
        num_commands = 7  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading, jump，adjust leg(2)
        resampling_time = 20  # time before command are changed[s] default:2
        heading_command = True  # if true: compute ang vel command from heading error
        threshold = 0.5  # 控制技能学习采样频率

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-2.0, 2.0]  # 更高的运动效率
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-5.0, 5.0]
            heading = [-3.14, 3.14]
            # knee_angle = [-3.0, -1.0]  # 膝关节弯曲角度
            knee_angle = [-2.3, -1.5]  # 膝关节弯曲角度
    
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            torque = 0.05
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

class DiabloCfgPPO(LeggedRobotCfgPPO):

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0

    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 1.0e-3  # 5.e-4
        gamma = 0.99

    class runner(LeggedRobotCfgPPO.runner):
        max_iterations = 5000 

        experiment_name = "diablo"

        resume = False

