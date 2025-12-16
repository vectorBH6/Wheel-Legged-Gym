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


class WheelLeggedCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = 4096

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"           # 平地
        curriculum = False            # 关闭课程学习
        static_friction = 0.9         # 适当的摩擦力
        dynamic_friction = 0.9
        restitution = 0.2           # 弹性碰撞 

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.25]  # x,y,z [m]
        default_joint_angles = {  # target angles when action = 0.0
            "lf0_Joint": 0.5,
            "lf1_Joint": 0.35,
            "l_wheel_Joint": 0.0,
            "rf0_Joint": -0.5,
            "rf1_Joint": -0.35,
            "r_wheel_Joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        pos_action_scale = 0.5
        vel_action_scale = 10.0
        # PD Drive parameters:
        stiffness = {"f0": 40.0, "f1": 40.0, "wheel": 0}  # [N*m/rad]
        damping = {"f0": 1.0, "f1": 1.0, "wheel": 0.5}  # [N*m*s/rad]

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/wl/urdf/wl.urdf"
        name = "WheelLegged"
        offset = 0.054
        l1 = 0.15
        l2 = 0.25
        penalize_contacts_on = ["lf", "rf", "base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class domain_rand:
        randomize_friction = False
        friction_range = [0.1, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-2.0, 3.0]
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]
        randomize_base_com = True
        rand_com_vec = [0.05, 0.05, 0.05]
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 2.0
        randomize_Kp = True
        randomize_Kp_range = [0.9, 1.1]
        randomize_Kd = True
        randomize_Kd_range = [0.9, 1.1]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.9, 1.1]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = True
        delay_ms_range = [0, 10]

    class rewards:
        class scales:
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1
            tracking_ang_vel = 1.0

            base_height = 1.0
            nominal_state = -0.1
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -10.0

            dof_vel = -5e-5
            dof_acc = -2.5e-7
            torques = -0.0001
            action_rate = -0.01
            action_smooth = -0.01

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
        base_height_target = 0.18
        max_contact_force = 100.0  # forces above this value are penalized

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

class WheelLeggedCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 1.0e-3  # 学习率
        gamma = 0.99
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        # logging
        experiment_name = "wheel_legged"

    class runner(LeggedRobotCfgPPO.runner):
        max_iterations = 2000     
        experiment_name = "wheel_legged"
        resume = False
        load_run = "Sep13_17-42-10_"  # 指定要加载的训练文件夹
        checkpoint = 900  # 指定要加载的模型iteration数
