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

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym.torch_utils import *
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from pynput import keyboard
import time

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

# 键盘控制状态
key_status = {
    'w': False,
    'a': False,
    's': False,
    'd': False,
    'z': False,
    'c': False,
}
def update_commands(commands):
    commands[0] = 0.0
    commands[1] = 0.0
    commands[2] = 0.13  # 默认高度
    if(key_status['w']):commands[0] = 0.6
    if(key_status['s']):commands[0] = -0.6
    if(key_status['a']):commands[1] = 2.0
    if(key_status['d']):commands[1] = -2.0
    if(key_status['z']):commands[2] = 0.14
    if(key_status['c']):commands[2] = 0.12
# 键盘事件处理
def on_press(key):
    try:
        if key.char in key_status:
            key_status[key.char] = True
    except AttributeError:
        pass
def on_release(key):
    try:
        if key.char in key_status:
            key_status[key.char] = False
    except AttributeError:
        pass
# 启动键盘监听
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 30
    env_cfg.env.fail_to_terminal_time_s = 10
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.friction_range = [0.2, 0.7]
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 7
    env_cfg.domain_rand.max_push_vel_xy = 1.0
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = True
    env_cfg.domain_rand.delay_ms_range = [0, 7]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_history = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            WHEEL_LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 0  # which joint is used for logging
    stop_state_log = 3000  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    latent = None

    CoM_offset_compensate = False
    vel_err_intergral = torch.zeros(env.num_envs, device=env.device)
    vel_cmd = torch.zeros(env.num_envs, device=env.device)
    
    cmd_ranges = env_cfg.commands.ranges
    command_update_interval = int(5.0 / env.dt)  # 每5秒更新命令（步数）

    for i in range(5000 * int(env.max_episode_length)):
        if ppo_runner.alg.actor_critic.is_sequence:
            actions, latent = policy(obs, obs_history)
        else:
            actions = policy(obs.detach())

        # 打印观测数据
        print("=== 观测数据 ===")
        print(f"角速度 (ang_vel): [{obs[robot_index, 0].item():.4f}, {obs[robot_index, 1].item():.4f}, {obs[robot_index, 2].item():.4f}]")
        print(f"重力投影 (gravity): [{obs[robot_index, 3].item():.4f}, {obs[robot_index, 4].item():.4f}, {obs[robot_index, 5].item():.4f}]")
        print(f"命令 (commands): lin_vel={obs[robot_index, 6].item():.4f}, ang_vel={obs[robot_index, 7].item():.4f}, height={obs[robot_index, 8].item():.4f}")
        print(f"关节位置 (dof_pos): L0={obs[robot_index, 9].item():.4f}, L1={obs[robot_index, 10].item():.4f}, LW={obs[robot_index, 11].item():.4f}, R0={obs[robot_index, 12].item():.4f}, R1={obs[robot_index, 13].item():.4f}, RW={obs[robot_index, 14].item():.4f}")
        print(f"关节速度 (dof_vel): L0={obs[robot_index, 15].item():.4f}, L1={obs[robot_index, 16].item():.4f}, LW={obs[robot_index, 17].item():.4f}, R0={obs[robot_index, 18].item():.4f}, R1={obs[robot_index, 19].item():.4f}, RW={obs[robot_index, 20].item():.4f}")
        print(f"上一次动作 (actions): L0={obs[robot_index, 21].item():.4f}, L1={obs[robot_index, 22].item():.4f}, LW={obs[robot_index, 23].item():.4f}, R0={obs[robot_index, 24].item():.4f}, R1={obs[robot_index, 25].item():.4f}, RW={obs[robot_index, 26].item():.4f}")
        print()

        '''
        if i % command_update_interval == 0:
            current_lin_vel_cmd = np.random.uniform(cmd_ranges.lin_vel_x[0], cmd_ranges.lin_vel_x[1])
            current_height_cmd = np.random.uniform(cmd_ranges.height[0], cmd_ranges.height[1])
            current_yaw_cmd = np.random.uniform(cmd_ranges.ang_vel_yaw[0], cmd_ranges.ang_vel_yaw[1])
        
        env.commands[:, 0] = current_lin_vel_cmd 
        env.commands[:, 1] = current_yaw_cmd     
        env.commands[:, 2] = current_height_cmd   
        '''
        # 键盘控制
        keyboard_commands = [0.0, 0.0, 0.0]
        update_commands(keyboard_commands)
        env.commands[:, 0] = keyboard_commands[0]
        env.commands[:, 1] = keyboard_commands[1]
        env.commands[:, 2] = keyboard_commands[2]
        
        if CoM_offset_compensate:
            if i > 200 and i < 600:
                vel_cmd[:] = 2.5 * np.clip((i - 200) * 0.05, 0, 1)
            else:
                vel_cmd[:] = 0
            vel_err_intergral += (
                (vel_cmd - env.base_lin_vel[:, 0])
                * env.dt
                * ((vel_cmd - env.base_lin_vel[:, 0]).abs() < 0.5)
            )
            vel_err_intergral = torch.clip(vel_err_intergral, -0.5, 0.5)
            env.commands[:, 0] = vel_cmd + vel_err_intergral
        '''
        actions[robot_index, 0] = - 1.0
        actions[robot_index, 1] = 1.0
        actions[robot_index, 2] = 0.0
        actions[robot_index, 3] = 1.0
        actions[robot_index, 4] = - 1.0
        actions[robot_index, 5] = 0.0
        '''
        # 五连杆并联结构参数
        la_length = 0.06  # 短杆长度 (m)
        ra_length = 0.06  # 短杆长度 (m)  
        lb_length = 0.10  # 长杆长度 (m)
        rb_length = 0.10  # 长杆长度 (m)
        interval_distance = 0.04  # 间距 (m)
        
        # 关节角度计算
        L_theta1 = 6.46 + 0.6 * 180 / 3.14 - actions[robot_index, 0].item()*env.cfg.control.pos_action_scale* 180 / 3.14
        L_leg_angle = 52.96 + 0.25 * 180 / 3.14 + actions[robot_index, 1].item()*env.cfg.control.pos_action_scale* 180 / 3.14
        R_theta1 = 6.46 + 0.6 * 180 / 3.14 + actions[robot_index, 3].item()*env.cfg.control.pos_action_scale* 180 / 3.14
        R_leg_angle = 52.96 + 0.25 * 180 / 3.14 - actions[robot_index, 4].item()*env.cfg.control.pos_action_scale* 180 / 3.14
        
        # 轮子力矩计算
        torque_wL = env.cfg.control.damping["W"]*(actions[robot_index, 2].item()*env.cfg.control.vel_action_scale )
        torque_wR = -env.cfg.control.damping["W"]*(actions[robot_index, 5].item()*env.cfg.control.vel_action_scale )

        # 五连杆运动学解算
        l1 = interval_distance + la_length * np.cos(L_theta1 * np.pi / 180.0) - lb_length * np.cos((L_leg_angle - L_theta1) * np.pi / 180.0)
        l2 = la_length * np.sin(L_theta1 * np.pi / 180.0) + lb_length * np.sin((L_leg_angle - L_theta1) * np.pi / 180.0)
        r1 = interval_distance + ra_length * np.cos(R_theta1 * np.pi / 180.0) - rb_length * np.cos((R_leg_angle - R_theta1) * np.pi / 180.0)
        r2 = ra_length * np.sin(R_theta1 * np.pi / 180.0) + rb_length * np.sin((R_leg_angle - R_theta1) * np.pi / 180.0)
        # 计算左腿l1和短杆的夹角 (四边形简化方法)
        L_third_edge = np.sqrt(l1 * l1 + l2 * l2)  # 第三边长度
        L_angle_deg = np.arccos(l1 / L_third_edge) * 180.0 / np.pi  # 度
        # 用余弦定理计算L_third_edge和la_length的夹角
        L_cos_theta = (L_third_edge * L_third_edge + la_length * la_length - lb_length * lb_length) / (2.0 * L_third_edge * la_length)
        L_theta2 = np.arccos(L_cos_theta) * 180.0 / np.pi + L_angle_deg
        # 计算右腿r1和短杆的夹角 (四边形简化方法)  
        R_third_edge = np.sqrt(r1 * r1 + r2 * r2)  # 第三边长度
        R_angle_deg = np.arccos(r1 / R_third_edge) * 180.0 / np.pi  # 度
        # 用余弦定理计算R_third_edge和ra_length的夹角
        R_cos_theta = (R_third_edge * R_third_edge + ra_length * ra_length - rb_length * rb_length) / (2.0 * R_third_edge * ra_length)
        R_theta2 = np.arccos(R_cos_theta) * 180.0 / np.pi + R_angle_deg
        
        # 打印五连杆解算数据
        print(f"=== 五连杆解算数据 ===")
        print(f"左腿: theta1={L_theta1:.2f}°, theta2={L_theta2:.2f}°")
        print(f"右腿: theta1={R_theta1:.2f}°, theta2={R_theta2:.2f}°")
        print(f"轮子ac*scale: L={-actions[robot_index, 2].item()*env.cfg.control.vel_action_scale:.4f}rad/s, R={actions[robot_index, 5].item()*env.cfg.control.vel_action_scale:.4f}rad/s")
        # print(f"轮子dof_vel: L={env.dof_vel[robot_index, 2].item():.4f}rad/s, R={-env.dof_vel[robot_index, 5].item():.4f}rad/s")
        print(f"轮子力矩: L={torque_wL:.4f}Nm, R={torque_wR:.4f}Nm", "\n")

        # 打印舵机力矩
        #print(f"=== 舵机力矩 ===")
        #print(f"舵机力矩: L0={env.torques[robot_index, 0].item():.4f}Nm, R0={env.torques[robot_index, 3].item():.4f}Nm", "\n")

        obs, _, rews, dones, infos, obs_history = env.step(actions)
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    WHEEL_LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_position[robot_index, :].to(device="cpu")
            )
            camera_position = target_position + camera_offset
            env.set_camera(camera_position, target_position)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item()
                    * env.cfg.control.pos_action_scale
                    + env.default_dof_pos[robot_index, joint_index].item(),
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_yaw": env.commands[robot_index, 1].item(),
                    "command_height": env.commands[robot_index, 2].item(),
                    "base_height": env.base_height[robot_index].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
            if CoM_offset_compensate:
                logger.log_states({"command_x": vel_cmd[robot_index].item()})
            else:
                logger.log_states({"command_x": env.commands[robot_index, 0].item()})
            if latent is not None:
                logger.log_states(
                    {
                        "est_lin_vel_x": latent[robot_index, 0].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_y": latent[robot_index, 1].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_z": latent[robot_index, 2].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                    }
                )
                if latent.shape[1] > 3 and env_cfg.noise.add_noise:
                    logger.log_states(
                        {
                            "base_vel_yaw_obs": obs[robot_index, 2].item()
                            / env.cfg.normalization.obs_scales.ang_vel,
                            "dof_pos_obs": obs[robot_index, 9 + joint_index].item()
                            / env.cfg.normalization.obs_scales.dof_pos
                            + env.default_dof_pos[robot_index, joint_index].item(),
                            "dof_vel_obs": obs[robot_index, 15 + joint_index].item()
                            / env.cfg.normalization.obs_scales.dof_vel,
                        }
                    )
                    logger.log_states(
                        {
                            "base_vel_yaw_est": latent[robot_index, 3 + 2].item()
                            / env.cfg.normalization.obs_scales.ang_vel,
                            "dof_pos_est": latent[
                                robot_index, 3 + 9 + joint_index
                            ].item()
                            / env.cfg.normalization.obs_scales.dof_pos
                            + env.default_dof_pos[robot_index, joint_index].item(),
                            "dof_vel_est": latent[
                                robot_index, 3 + 15 + joint_index
                            ].item()
                            / env.cfg.normalization.obs_scales.dof_vel,
                        }
                    )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
