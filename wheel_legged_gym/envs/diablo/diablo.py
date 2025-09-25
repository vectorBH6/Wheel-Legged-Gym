# ==============================================================================
# Author: Zhengkr
# Created on: 2024-03-11
# Description: 双轮足机器人刑天任务设计
# Copyright: Optional, add copyright information if needed.
# Revision History:
#   YYYY-MM-DD: Made modifications, updated XX feature.
# ==============================================================================


from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from wheel_legged_gym.utils.math import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from wheel_legged_gym.envs import LeggedRobot
from wheel_legged_gym.utils.terrain import Terrain
from .diablo_config import DiabloCfg


class Diablo(LeggedRobot):
    def __init__(self, cfg: DiabloCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # SF机器人关节索引定义
        # 关节顺序：LT_joint, LC_joint, LW_joint, RT_joint, RC_joint, RW_joint
        # 索引：     0,       1,       2,       3,       4,       5
        self.leg_joint_indices = [0, 1, 3, 4]  # 腿部关节（舵机）
        self.wheel_joint_indices = [2, 5]       # 轮子关节


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
        dof_pos_obs_modified[:, self.wheel_joint_indices] = 0.0
        
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
        
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )
