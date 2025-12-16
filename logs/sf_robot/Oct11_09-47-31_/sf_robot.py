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

import torch
from wheel_legged_gym.envs.base.legged_robot import LeggedRobot
from .sf_robot_config import SFRobotCfg


class SFRobot(LeggedRobot):
    def __init__(self, cfg: SFRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # SF机器人关节索引定义
        # 关节顺序：LT_joint, LC_joint, LW_joint, RT_joint, RC_joint, RW_joint
        # 索引：     0,       1,       2,       3,       4,       5
        self.leg_joint_indices = [0, 1, 3, 4]  # 腿部关节（舵机）
        self.wheel_joint_indices = [2, 5]       # 轮子关节

        # 初始化上一步力矩存储（用于变化率限制）
        self.last_torques = torch.zeros_like(self.torques)
        
        # 初始化上一步关节速度存储（用于轮子D控制）
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        # 初始化舵机关节积分误差（仅用于舵机PID控制）
        self.leg_integral_error = torch.zeros(self.num_envs, len(self.leg_joint_indices), 
                                            dtype=torch.float, device=self.device)

    def compute_proprioception_observations(self):
        """
        计算本体感受观测，针对SF轮足机器人优化：
        - 舵机关节：保留位置信息，速度信息设为0
        - 轮子关节：保留速度信息，位置信息设为0
        """
        # 创建修改后的关节位置和速度观测
        dof_pos_obs = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dof_vel_obs = self.dof_vel * self.obs_scales.dof_vel
        
        # 对于轮子关节，将位置观测设为0
        dof_pos_obs_modified = dof_pos_obs.clone()
        dof_pos_obs_modified = self.actions * self.cfg.control.pos_action_scale * 0.5
        dof_pos_obs_modified[:, self.wheel_joint_indices] = 0.0
        #dof_pos_obs_modified[:, self.leg_joint_indices] = 0.0
        
        # 对于舵机关节，将速度观测设为0
        dof_vel_obs_modified = dof_vel_obs.clone()
        dof_vel_obs_modified[:, self.leg_joint_indices] = 0.0
        
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                dof_pos_obs_modified,  # 修改后的位置观测（轮子位置为0）
                dof_vel_obs_modified,  # 修改后的速度观测（舵机速度为0）
                self.actions,
            ),
            dim=-1,
        )
        return obs_buf

    def _compute_torques(self, actions):
        """
        计算力矩，针对SF轮足机器人优化：
        - 舵机关节：位置控制
        - 轮子关节：速度控制
        """
        # pd controller
        pos_ref = actions * self.cfg.control.pos_action_scale
        pos_ref[:, self.wheel_joint_indices] *= 0  # 轮子关节位置参考设为0
        
        vel_ref = actions * self.cfg.control.vel_action_scale
        vel_ref[:, self.leg_joint_indices] *= 0    # 舵机关节速度参考设为0
        
        torques = self.p_gains * (
            pos_ref + self.default_dof_pos - self.dof_pos
        ) + self.d_gains * (vel_ref - self.dof_vel)
        
        # 为舵机关节添加积分项（PID控制）
        if hasattr(self.cfg.control, 'leg_integral_gain'):
            dt = self.dt * self.cfg.control.decimation
            leg_pos_error = (pos_ref + self.default_dof_pos - self.dof_pos)[:, self.leg_joint_indices]
            
            # 更新积分误差
            #self.leg_integral_error += leg_pos_error * dt
            self.leg_integral_error += leg_pos_error * 0.2
            
            # 积分饱和限制
            windup_limit = getattr(self.cfg.control, 'integral_windup_limit', 1.0)
            self.leg_integral_error = torch.clip(self.leg_integral_error, -windup_limit, windup_limit)
            
            # 添加积分项到舵机关节力矩
            integral_torques = self.cfg.control.leg_integral_gain * self.leg_integral_error
            torques[:, self.leg_joint_indices] += integral_torques

        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def _reward_leg_together(self):
        """
        两腿并拢，防止太空步：
        - 计算左右腿关节位置差异
        - 差异越小，奖励越高
        """
        # 获取左右腿关节位置
        left_leg_pos = self.dof_pos[:, [0, 1]]  # LT_joint, LC_joint
        right_leg_pos = self.dof_pos[:, [3, 4]]  # RT_joint, RC_joint
        
        # 计算左右腿对应关节的位置差异
        leg_diff = torch.abs(left_leg_pos + right_leg_pos)
        
        # 计算平均差异，差异越小奖励越高
        mean_diff = torch.mean(leg_diff, dim=1) 

        # 指数函数，差异为0时奖励最大
        reward = torch.exp(-mean_diff / 0.01)  # 0.1是调节参数，控制奖励衰减速度

        if self.reward_scales["leg_together"] < 0:
            return mean_diff
        else:
            return reward

    def _reward_base_height(self):
        # Penalize base height away from target
        # print(self.commands[0, 2], self.base_height[0])
        if self.reward_scales["base_height"] < 0:
            return torch.abs(self.base_height - self.commands[:, 2])
        else:
            base_height_error = torch.square(self.base_height - self.commands[:, 2])
            return torch.exp(-base_height_error / 0.0001)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 1] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / 0.05)

