#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <mutex>
#include <thread>
#include <iostream>
#include "stdio.h"

struct _msg_request
{
    float trigger;
    float command[4];
    float eu_ang[3];
    float omega[3];
    float q[6];        // SF机器人6个关节
    float dq[6];
    float init_pos[6];
};

struct _msg_response
{
    float q_exp[6];    // SF机器人6个关节
    float dq_exp[6];
};

class RL_WL {
public:
    std::string model_path;
    void init_policy();
    void load_policy();
    torch::Tensor model_infer(torch::Tensor policy_input);
    void handleMessage(_msg_request request);
    
    // 自定义读取和发送函数接口
    bool readRobotState(_msg_request& request);
    bool sendControlCommand(_msg_response& response);

    //gamepad
    float smooth = 0.03;  // 平滑系数
    float dead_zone = 0.01; // 死区阈值，小于的输入会被忽略

    float cmd_x = 0.;
    float cmd_rate = 0.;
    float cmd_height = 0.;

    std::vector<float> action;   // 实际发送给机器人的动作
    std::vector<float> action_temp;  // 神经网络滤波后输出的动作
    std::vector<float> prev_action;  

    float vel_action_scale = 10;
    float pos_action_scale = 0.5;

    torch::Tensor action_buf;
    torch::Tensor obs_buf;
    torch::Tensor last_action;

    // default values
    int action_refresh=0;
    int history_length = 5;  // 与SF机器人训练配置保持一致
    // SF机器人关节配置：
    // 关节索引: 0=LT_joint(左大腿), 1=LC_joint(左小腿), 2=LW_joint(左轮), 
    //          3=RT_joint(右大腿), 4=RC_joint(右小腿), 5=RW_joint(右轮)
    // 初始位置：LT_joint, LC_joint, LW_joint, RT_joint, RC_joint, RW_joint
    float init_pos[6] = {-0.50, 0.25, 0.0, 0.50, -0.25, 0.0};
    float eu_ang_scale= 1;
    float omega_scale= 0.25;
    float pos_scale = 1.0;
    float vel_scale = 0.05;
    float lin_vel = 2.0;
    float ang_vel = 0.25;
    float height_measurements = 5.0;   
    torch::jit::script::Module model;
    torch::DeviceType device;
private:

   
};

