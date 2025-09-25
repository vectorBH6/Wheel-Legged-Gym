# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class SFRobotCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = 8192 

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"           # 平地
        curriculum = False            # 关闭课程学习
        static_friction = 0.8         # 适当的摩擦力
        dynamic_friction = 0.8
        restitution = 0.2           # 弹性碰撞 

    class commands:
        curriculum = True
        basic_max_curriculum = 2.5
        advanced_max_curriculum = 1.5
        curriculum_threshold = 0.7
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.7, 0.7]  # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]  # min max [rad/s]
            height = [0.12, 0.14]  # 调整目标高度 0.12 0.15
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.14]  # x,y,z [m]
        default_joint_angles = {  # target angles when action = 0.0
            "LT_joint": -0.50,    # 左腿大腿关节
            "LC_joint": 0.25,    # 左腿小腿关节  
            "LW_joint": 0.0,    # 左轮关节
            "RT_joint": 0.50,    # 右腿大腿关节
            "RC_joint": -0.25,    # 右腿小腿关节
            "RW_joint": 0.0,    # 右轮关节
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        pos_action_scale = 0.5
        vel_action_scale = 10.0
        # PD Drive parameters:
        stiffness = {"T": 4.0, "C": 4.0, "W": 0}  # [N*m/rad] T:大腿, C:小腿, W:轮子
        damping = {"T": 0.5, "C": 0.5, "W": 0.03}  # [N*m*s/rad]

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/sf/urdf/sf.urdf"
        name = "SFRobot"
        foot_name = "W_link"  # 足端名称：LW_link, RW_link (轮子作为足端)
        offset = 0.00
        l1 = 0.00
        l2 = 0.00
        penalize_contacts_on = ["LT", "LC", "RT", "RC", "base"]  
        terminate_after_contacts_on = ["base", "LC", "RC"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            torque = 0.05

        clip_observations = 100.0
        clip_actions = 100.0

    class domain_rand(LeggedRobotCfg.domain_rand):

        randomize_base_com = False
        randomize_Kp = False
        randomize_Kd = False
        
        # 惯性矩阵随机化
        randomize_inertia = True
        randomize_inertia_range = [0.9, 1.1]

        # 动作延迟随机化
        randomize_action_delay = False
        delay_ms_range = [0, 10]

        # 电机力矩随机化
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.9, 1.1]

        # 摩擦力随机化
        randomize_friction = True
        friction_range = [0.5, 0.8]

        # 地面弹性系数随机化
        randomize_restitution = True
        restitution_range = [0.0, 0.6]

        # 质量随机化
        randomize_base_mass = True
        added_mass_range = [-0.00, 0.08]

        # 推力干扰
        push_robots = True
        push_interval_s = 5           # 每5秒推一次，适合小机器人
        max_push_vel_xy = 0.1        # 小机器人用更小的推力
        
        # 关节位置随机化
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            tracking_lin_vel = 1.5
            tracking_lin_vel_enhance = 1
            tracking_ang_vel = 1.5
            tracking_ang_vel_enhance = 0.0

            base_height = 4.0
            nominal_state = -0.1
            lin_vel_z = -2.0
            ang_vel_xy = -0.1   #
            orientation = -15.0
            leg_together = -0.8 

            dof_vel = -5e-5
            dof_acc = -2.5e-7
            torques = -0.0001
            action_rate = -0.1   #
            action_smooth = -0.1   #

            collision = -1.0
            dof_pos_limits = -1.0
        
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_single_reward = 1
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = (
            0.97  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        max_contact_force = 30.0  # forces above this value are penalized

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.3
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.1
            height_measurements = 0.1



class SFRobotCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 1e-5  # 学习率
        gamma = 0.995

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.3  # 标准差

    class runner(LeggedRobotCfgPPO.runner):
        max_iterations = 5000     
        experiment_name = "sf_robot"
        resume = True
        load_run = "Sep22_14-09-50_"  # 指定要加载的训练文件夹
        checkpoint = 500  # 指定要加载的模型iteration数
