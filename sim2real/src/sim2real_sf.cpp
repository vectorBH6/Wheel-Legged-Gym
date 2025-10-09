#include "sim2real_sf.h"
#include <iostream>
#include <valarray>
#include <pthread.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
using namespace std;
using namespace torch::indexing; // 确保使用了正确的命名空间
RL_WL sf_rl;
//RL
struct _msg_request msg_request;
struct _msg_response msg_response;

float limit(float input,float min,float max){
    if(input>max)
        return max;
    if(input<min)
        return min;
    return input;
} 

void RL_WL::handleMessage(_msg_request request)//获取机器人反馈
{              
    // #单次观测
    // obs[0, 0] = omega[0] *cfg.normalization.obs_scales.ang_vel
    // obs[0, 1] = omega[1] *cfg.normalization.obs_scales.ang_vel
    // obs[0, 2] = omega[2] *cfg.normalization.obs_scales.ang_vel
    // obs[0, 3] = eu_ang[0] *cfg.normalization.obs_scales.quat
    // obs[0, 4] = eu_ang[1] *cfg.normalization.obs_scales.quat
    // obs[0, 5] = eu_ang[2] *cfg.normalization.obs_scales.quat
    // obs[0, 6] = cmd.vx * cfg.normalization.obs_scales.lin_vel
    // obs[0, 7] = cmd.vy * cfg.normalization.obs_scales.lin_vel
    // obs[0, 8] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
    // obs[0, 9:15] = (q-default_dof_pos) * cfg.normalization.obs_scales.dof_pos #SF机器人6个关节
    // obs[0, 15:21] = dq * cfg.normalization.obs_scales.dof_vel  
    // obs[0, 21:27] = last_actions#上次控制指令
    #if 0
        cout<<"cmd:";
        cout<<request.command[0]<<" ";
        cout<<request.command[1]<<" ";
        cout<<request.command[2]<<" ";
        cout<<request.command[3]<<" ";
        cout<<endl;
        cout<<"att:";
        cout<<request.eu_ang[0]<<" ";
        cout<<request.eu_ang[1]<<" ";
        cout<<request.eu_ang[2]<<" ";
        cout<<endl;
        cout<<"rate:";
        cout<<request.omega[0]<<" ";
        cout<<request.omega[1]<<" ";
        cout<<request.omega[2]<<" ";
        cout<<endl;
        cout<<"q:";
        for(int i=0;i<6;i++)
            cout<<request.q[i]<<" ";
        cout<<endl;
        cout<<"dq:";
        for(int i=0;i<6;i++)
            cout<<request.dq[i]<<" ";
        cout<<endl;    
        cout<<"trigger:"<<request.trigger<<" ";
        
    #endif
    // 将 data 转为 tensor 类型，输入到模型
    if(request.trigger==1){
        request.trigger=0;
        std::vector<float> obs;
        for(int i=0;i<6;i++)
            init_pos[i]=request.init_pos[i];
            
        //---------------Push data into obsbuf--------------------
        obs.push_back(request.omega[0]*omega_scale);
        obs.push_back(request.omega[1]*omega_scale);
        obs.push_back(request.omega[2]*omega_scale);

        obs.push_back(request.eu_ang[0]*eu_ang_scale);
        obs.push_back(request.eu_ang[1]*eu_ang_scale);
        obs.push_back(request.eu_ang[2]*eu_ang_scale);

        // cmd
        float max = 1.0;
        float min = -1.0;

        cmd_x = cmd_x * (1 - smooth) + (std::fabs(request.command[0]) < dead_zone ? 0.0 : request.command[0]) * smooth;
        cmd_rate = cmd_rate * (1 - smooth) + (std::fabs(request.command[1]) < dead_zone ? 0.0 : request.command[2]) * smooth;
        cmd_height = std::fabs(request.command[2]) < dead_zone ? 0.0 : request.command[2];

        obs.push_back(cmd_x*lin_vel);//控制指令x (前进后退)
        obs.push_back(cmd_rate*ang_vel);//控制指令yaw rate (转向)  
        obs.push_back(cmd_height*height_measurements);//控制指令height (高度控制)

        // pos q joint - 轮子位置设为0，舵机位置保留
        for (int i = 0; i < 6; ++i)
        {
            if (i == 2 || i == 5) {  // 轮子关节 (LW_joint, RW_joint)
                obs.push_back(0.0);  // 轮子位置设为0
            } else {  // 舵机关节 (LT_joint, LC_joint, RT_joint, RC_joint)
                float pos = (request.q[i] - init_pos[i]) * pos_scale;
                obs.push_back(pos);
            }
        }
        // vel q joint - 舵机速度设为0，轮子速度保留
        for (int i = 0; i < 6; ++i)
        {
            if (i == 0 || i == 1 || i == 3 || i == 4) {  // 舵机关节 (LT_joint, LC_joint, RT_joint, RC_joint)
                obs.push_back(0.0);  // 舵机速度设为0
            } else {  // 轮子关节 (LW_joint, RW_joint)
                float vel = request.dq[i] * vel_scale;
                obs.push_back(vel);
            }
        }
        // last action
        for (int i = 0; i < 6; ++i)
        {
            obs.push_back(action_temp[i]);
        }
        // std::cout<<("----------------obs---------------")<<std::endl;
        // cout<<obs<<endl;
        // std::cout<<("--------------------------------")<<std::endl;

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor obs_tensor = torch::from_blob(obs.data(),{1,27},options).to(device);
        //-------------b---------------------------------------------------
        auto obs_buf_batch = obs_buf.unsqueeze(0);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(obs_tensor.to(torch::kHalf));
        inputs.push_back(obs_buf_batch.to(torch::kHalf));
        
        //---------------------------网络推理----------------------------- Execute the model and turn its output into a tensor
        //cout<<"obs_tensor1:"<<endl<<obs_tensor<<endl;
        //std::cout<<("*****************")<<std::endl;
        //cout<<"obs_buf_batch:"<<endl<<obs_buf_batch<<endl;
        torch::Tensor action_tensor = model.forward(inputs).toTensor();
        action_buf = torch::cat({action_buf.index({ Slice(1,None),Slice()}),action_tensor},0);
        //cout<<"[action out]:"<<endl<<action_tensor<<endl;
        bool has_nan = false;
        for (float val : obs) {
            //cout << val << " ";
            if (std::isnan(val)) {
                has_nan = true;
            }
        }
        if (has_nan) {
            cout << "NaN detected in obs. Press any key to continue..." << endl;
            getchar(); // 等待键盘输入
        }

        //-----------------------------网络输出滤波--------------------------------
        torch::Tensor action_blend_tensor = 0.8*action_tensor + 0.2*last_action;
        last_action = action_tensor.clone();
    
        this->obs_buf = torch::cat({this->obs_buf.index({Slice(1, None), Slice()}), obs_tensor}, 0); // 历史观测移位
        // //obs_buf = torch::cat({obs_buf.index({Slice(1,None),Slice()}),obs_tensor},0);//历史观测移位
        // //----------------------------------------------------------------
        torch::Tensor action_raw = action_blend_tensor.squeeze(0);
        // move to cpu
        action_raw = action_raw.to(torch::kFloat32);
        action_raw = action_raw.to(torch::kCPU);
        // // assess the result
        auto action_getter = action_raw.accessor <float,1>();//bug
        for (int j = 0; j < 6; j++)
        {
            action_temp[j] = limit(action_getter[j], -100, 100);   //对应config文件的clip_actions，裁剪动作范围
            if (j == 2 || j == 5) {  // 轮子关节 (LW_joint, RW_joint)
                action[j] = action_temp[j] * vel_action_scale;
            } else {  // 舵机关节 (LT_joint, LC_joint, RT_joint, RC_joint)  
                action[j] = action_temp[j] * pos_action_scale;
            } 
        }   //此时action为期望舵机角度和电机速度
        
        action_refresh=1;
    }
}

void RL_WL::load_policy()
{   
    std::cout << model_path << std::endl;
    // load model from check point
    std::cout << "cuda::is_available():" << torch::cuda::is_available() << std::endl;
    device= torch::kCPU;
    if (torch::cuda::is_available()&&1){
        device = torch::kCUDA;
        printf("device = torch::kCUDA\n");
    }
    std::cout<<"device:"<<device<<endl;
    model = torch::jit::load(model_path);
    std::cout << "load model is successed!" << std::endl;
    model.to(device);
    std::cout << "LibTorch Version: " << TORCH_VERSION_MAJOR << "." 
              << TORCH_VERSION_MINOR << "." 
              << TORCH_VERSION_PATCH << std::endl;
    model.to(torch::kHalf);
    std::cout << "load model to device!" << std::endl;
    model.eval();
}
 
void RL_WL::init_policy(){
 // load policy
    std::cout << "RL model thread start"<<endl;
    cout <<"cuda_is_available:"<< torch::cuda::is_available() << endl;
    cout <<"cudnn_is_available:"<< torch::cuda::cudnn_is_available() << endl;
    
    model_path = "../../logs/sf_robot/exported/policies/policy_1.pt";//载入SF机器人模型
    load_policy();

 // initialize record
    action_buf = torch::zeros({history_length,6},device);
    obs_buf = torch::zeros({history_length,27}, device);//历史观测
    last_action = torch::zeros({1,6},device);

    action_buf.to(torch::kHalf);
    obs_buf.to(torch::kHalf);
    last_action.to(torch::kHalf);

    for (int j = 0; j < 6; j++)
    {
        action_temp.push_back(0.0);
	    action.push_back(init_pos[j]);
        prev_action.push_back(init_pos[j]);
    }
    //hot start
    for (int i = 0; i < history_length; i++)//为历史观测初始化
    {
        // 将 data 转为 tensor 类型，输入到模型
        std::vector<float> obs;
        //---------------Push data into obsbuf--------------------
        obs.push_back(0);//request->omega[0]*omega_scale);
        obs.push_back(0);//request->omega[1]*omega_scale);
        obs.push_back(0);//request->omega[2]*omega_scale);

        obs.push_back(0);//request->eu_ang[0]*eu_ang_scale);
        obs.push_back(0);//request->eu_ang[1]*eu_ang_scale);
        obs.push_back(0);//request->eu_ang[2]*eu_ang_scale);

        // cmd
        obs.push_back(0);//控制指令x
        obs.push_back(0);//控制指令y
        obs.push_back(0);//控制指令yaw rate

        // pos q joint
        for (int i = 0; i < 6; ++i)
        {
            float pos = 0;
            obs.push_back(pos);
            action[i]=init_pos[i];
        }
        // vel q joint
        for (int i = 0; i < 6; ++i)
        {
            float vel = 0;
            obs.push_back(vel);
        }
        // last action
        for (int i = 0; i < 6; ++i)
        {
            obs.push_back(0);
        }
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor obs_tensor = torch::from_blob(obs.data(),{1,27},options).to(device);
    }

    for (int i = 0; i < 200; i++)
    {
        //sf_rl.model_infer();
    }
}

// TODO: 您需要实现这个函数 - 读取机器人状态
bool RL_WL::readRobotState(_msg_request& request) {
    // TODO: 在这里实现读取您的机器人状态
    // 例如：
    // - 从IMU读取姿态角 request.eu_ang[3] 和角速度 request.omega[3]
    // - 从加速度计读取 request.acc[3]
    // - 从电机读取关节角度 request.q[6] 和角速度 request.dq[6]
    // - 从遥控器读取指令 request.command[4]
    // - 设置触发标志 request.trigger = 1
    
    // 示例框架（请根据您的硬件接口修改）：
    /*
    // 读取IMU数据
    readIMU(request.eu_ang, request.omega, request.acc);
    
    // 读取关节状态（SF机器人6个关节）
    readJointStates(request.q, request.dq, request.tau);
    
    // 读取遥控指令
    readRemoteControl(request.command);
    
    // 设置触发
    request.trigger = 1;
    
    // 设置初始位置（第一次运行时）
    static bool first_run = true;
    if (first_run) {
        for(int i = 0; i < 6; i++) {
            request.init_pos[i] = init_pos[i];
        }
        first_run = false;
    }
    */
    
    return true; // 返回成功
}

// TODO: 您需要实现这个函数 - 发送控制指令
bool RL_WL::sendControlCommand(_msg_response& response) {
    // TODO: 在这里实现发送控制指令到您的硬件
    // 例如：通过串口发送给单片机
    
    // 示例框架（请根据您的硬件接口修改）：
    /*
    // 通过串口发送关节角度指令
    sendSerialCommand(response.q_exp, 6);
    
    // 或者发送速度指令
    // sendSerialCommand(response.dq_exp, 6);
    
    // 或者发送扭矩指令  
    // sendSerialCommand(response.tau_exp, 6);
    */
    
    return true; // 返回成功
}

int main(int argc, char** argv) {
    sf_rl.init_policy();
    for(int i=0;i<6;i++)
        msg_response.q_exp[i]=sf_rl.action[i];
    printf("Sim2real RL-SF started\n");
    int cnt_p=0;
    
    while (1)
    {
        //send action 打印
        if(sf_rl.action_refresh){
            sf_rl.action_refresh=0;
            for(int i=0;i<6;i++)
                msg_response.q_exp[i]=sf_rl.action[i];
            std::cout.precision(2);
            #if 1
                cout<<endl;
                cout<<"SF robot action send:";
                for(int i=0;i<6;i++)
                cout<<msg_response.q_exp[i]<<" ";
                cout<<endl;
            #endif
            cnt_p++;
        }
        
        // 发送控制指令数据
        sf_rl.sendControlCommand(msg_response);
        
        // 读取机器人状态数据
        if(sf_rl.readRobotState(msg_request)) {
            sf_rl.handleMessage(msg_request);
        }
        
        usleep(2000);  //2ms
    }
    return 0;
}

