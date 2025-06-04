#include <ATen/ops/ones_like.h>
#include <c10/core/ScalarType.h>
#include <cassert>
#include <cstdio>
#include <memory>
#include <torch/optim/sgd.h>
#include <torch/torch.h>
#include <unordered_map>
#include "rl/algorithms/a2c.h"
#include "rl/model/mlp_ac_cuda.h"

// 假设 cuda_backward 函数的声明在某个头文件中
extern void cuda_backward(
    float *input, float *shared_output, float *actor_hidden, float *critic_hidden,
    float *actor_fc2_output, float *critic_fc2_output, float *actor_output, float *critic_output,
    float *grad_actor_output, float *grad_critic_output,
    int batch_size, int input_dim, int hidden_dim, int output_dim,
    float *actor_fc2_w, float *critic_fc2_w, float *actor_head_w, float *critic_head_w,
    float *grad_shared_w, float *grad_shared_b,
    float *grad_actor_fc1_w, float *grad_actor_fc1_b,
    float *grad_actor_fc2_w, float *grad_actor_fc2_b,
    float *grad_actor_head_w, float *grad_actor_head_b,
    float *grad_critic_fc1_w, float *grad_critic_fc1_b,
    float *grad_critic_fc2_w, float *grad_critic_fc2_b,
    float *grad_critic_head_w, float *grad_critic_head_b
);

namespace rl
{

std::unordered_map<std::string, float> A2C::update(Traj &traj)
{
    traj.finalize();

    if (!traj.get_observations().defined()) { throw std::runtime_error("Observations tensor is undefined"); }
    if (!traj.get_actions().defined()) { throw std::runtime_error("Actions tensor is undefined"); }
    if (!traj.get_rewards().defined()) { throw std::runtime_error("Rewards tensor is undefined"); }
    if (!traj.get_returns().defined()) { throw std::runtime_error("Returns tensor is undefined"); }

    auto reward = traj.get_rewards();
    auto reward_dim = reward.size(-1);
    assert(reward_dim == 1);
    float total_episode_reward = reward.sum().item().toFloat();
    float mean_episode_reward = total_episode_reward / reward.size(0);

    int num_steps = traj.current_step();

    auto obs = traj.get_observations();
    auto action = traj.get_actions();
    auto v_logp_e_p = actor_and_critic->evaluate_actions(
        obs, action
    );

    for (size_t i = 0; i < v_logp_e_p.size(); ++i) {
        if (!v_logp_e_p[i].defined()) {
            throw std::runtime_error("Output tensor " + std::to_string(i) + " from evaluate_actions is undefined");
        }
    }

    auto distEntropy = v_logp_e_p[2].view({1, num_steps, 1});
    auto values = v_logp_e_p[0].view({1, num_steps, 1});
    auto actLogProbs = v_logp_e_p[1].view({1, num_steps, 1});

    auto advantages = traj.get_returns().view({1, num_steps, 1}) - values;

    auto value_loss = advantages.pow(2).mean();
    auto action_loss = -(advantages.detach() * actLogProbs).mean();
    auto distEntropy_loss = -distEntropy.mean();

    if (!value_loss.defined() || !action_loss.defined() || !distEntropy_loss.defined())
        { throw std::runtime_error("One of the loss components is undefined"); }

    auto loss = (
        value_loss       * value_loss_coef +
        action_loss      * actor_loss_coef +
        distEntropy_loss * entropy_coef
    );

    // Step optimizer
    optimizer->zero_grad();

    // 获取中间输出
    auto intermediate_outputs = actor_and_critic->get_intermediate_outputs(obs);
    auto input = intermediate_outputs[0].cpu().contiguous().data_ptr<float>();
    auto shared_output = intermediate_outputs[1].cpu().contiguous().data_ptr<float>();
    auto actor_hidden = intermediate_outputs[2].cpu().contiguous().data_ptr<float>();
    auto critic_hidden = intermediate_outputs[3].cpu().contiguous().data_ptr<float>();
    auto actor_output = intermediate_outputs[4].cpu().contiguous().data_ptr<float>();
    auto critic_output = intermediate_outputs[5].cpu().contiguous().data_ptr<float>();

    // 计算梯度
    auto grad_actor_output = actLogProbs.cpu().contiguous().data_ptr<float>();
    auto grad_critic_output = advantages.cpu().contiguous().data_ptr<float>();

    auto batch_size = obs.size(0);
    auto input_dim = obs.size(1);
    auto hidden_dim = dynamic_cast<MlpAC *>(actor_and_critic.get())->get_hidden_dim();
    auto output_dim = actor_and_critic->get_n_actions();

    // 获取权重
    auto actor_fc2_w = actor_and_critic->get_actor_fc2_w().cpu().contiguous().data_ptr<float>();
    auto critic_fc2_w = actor_and_critic->get_critic_fc2_w().cpu().contiguous().data_ptr<float>();
    auto actor_head_w = actor_and_critic->get_actor_head_w().cpu().contiguous().data_ptr<float>();
    auto critic_head_w = actor_and_critic->get_critic_head_w().cpu().contiguous().data_ptr<float>();

    // 分配梯度内存
    float *grad_shared_w = new float[hidden_dim * input_dim];
    float *grad_shared_b = new float[hidden_dim];
    float *grad_actor_fc1_w = new float[hidden_dim * hidden_dim];
    float *grad_actor_fc1_b = new float[hidden_dim];
    float *grad_actor_fc2_w = new float[output_dim * hidden_dim];
    float *grad_actor_fc2_b = new float[output_dim];
    float *grad_actor_head_w = new float[output_dim * output_dim];
    float *grad_actor_head_b = new float[output_dim];
    float *grad_critic_fc1_w = new float[hidden_dim * hidden_dim];
    float *grad_critic_fc1_b = new float[hidden_dim];
    float *grad_critic_fc2_w = new float[output_dim * hidden_dim];
    float *grad_critic_fc2_b = new float[output_dim];
    float *grad_critic_head_w = new float[output_dim * output_dim];
    float *grad_critic_head_b = new float[output_dim];

    // 调用CUDA反向传播函数
    cuda_backward(
        input, shared_output, actor_hidden, critic_hidden, actor_output, critic_output,
        grad_actor_output, grad_critic_output,
        batch_size, input_dim, hidden_dim, output_dim,
        actor_fc2_w, critic_fc2_w, actor_head_w, critic_head_w,
        grad_shared_w, grad_shared_b,
        grad_actor_fc1_w, grad_actor_fc1_b,
        grad_actor_fc2_w, grad_actor_fc2_b,
        grad_actor_head_w, grad_actor_head_b,
        grad_critic_fc1_w, grad_critic_fc1_b,
        grad_critic_fc2_w, grad_critic_fc2_b,
        grad_critic_head_w, grad_critic_head_b
    );

    // 更新参数
    actor_and_critic->update_parameters(
        grad_shared_w, grad_shared_b,
        grad_actor_fc1_w, grad_actor_fc1_b,
        grad_actor_fc2_w, grad_actor_fc2_b,
        grad_actor_head_w, grad_actor_head_b,
        grad_critic_fc1_w, grad_critic_fc1_b,
        grad_critic_fc2_w, grad_critic_fc2_b,
        grad_critic_head_w, grad_critic_head_b
    );

    // 释放梯度内存
    delete[] grad_shared_w;
    delete[] grad_shared_b;
    delete[] grad_actor_fc1_w;
    delete[] grad_actor_fc1_b;
    delete[] grad_actor_fc2_w;
    delete[] grad_actor_fc2_b;
    delete[] grad_actor_head_w;
    delete[] grad_actor_head_b;
    delete[] grad_critic_fc1_w;
    delete[] grad_critic_fc1_b;
    delete[] grad_critic_fc2_w;
    delete[] grad_critic_fc2_b;
    delete[] grad_critic_head_w;
    delete[] grad_critic_head_b;

    return {
        {"Value loss", value_loss.item().toFloat()},
        {"Action loss", action_loss.item().toFloat()},
        {"distEntropy loss", distEntropy_loss.item().toFloat()},
        {"Loss", loss.item().toFloat()},
        {"toatal_episode_reward", total_episode_reward},
        {"mean_episode_reward", mean_episode_reward},
        {"num_steps", num_steps}
    };
}

}