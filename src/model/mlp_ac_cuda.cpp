#include <c10/core/ScalarType.h>
#include <cassert>
#include <torch/torch.h>

#include "rl/model/mlp_ac_cuda.h"
#include "rl/model/output_layers.h"

using namespace torch::nn;

namespace rl {

// Xavier初始化函数
void xavier_init(Linear linear) {
  float gain = 1.0f; // 对于tanh激活函数
  float fan_in = linear->weight.size(1);
  float fan_out = linear->weight.size(0);
  float std = gain * std::sqrt(2.0f / (fan_in + fan_out));
  torch::nn::init::normal_(linear->weight, 0.0, std);
  torch::nn::init::zeros_(linear->bias);
}

class MlpAC : public AC_Base {
private:
  unsigned int hidden_dim;
  Sequential shared;
  Sequential actor;
  Sequential critic;
  std::shared_ptr<CategoricalOutput> actor_head;
  Linear critic_head;

  // 缓存中间结果
  torch::Tensor cached_obs;
  torch::Tensor cached_x;
  torch::Tensor cached_hidden_actor;
  torch::Tensor cached_hidden_critic;
  torch::Tensor cached_actor_output;
  torch::Tensor cached_value;

public:
  MlpAC(unsigned int obs_dim, unsigned int n_actions,
             unsigned int hidden_dim)
      : AC_Base(obs_dim, n_actions), hidden_dim(hidden_dim), shared(nullptr),
        actor(nullptr), critic(nullptr), actor_head(nullptr),
        critic_head(nullptr) {
    // 创建网络层
    shared = Sequential(Linear(obs_dim, hidden_dim), Functional(torch::tanh));
    actor =
        Sequential(Linear(hidden_dim, hidden_dim),                          // 0
                   Functional(torch::tanh), Linear(hidden_dim, hidden_dim), // 2
                   Functional(torch::tanh));
    critic = Sequential(Linear(hidden_dim, hidden_dim), Functional(torch::tanh),
                        Linear(hidden_dim, hidden_dim), Functional(torch::tanh));
    critic_head = Linear(hidden_dim, 1);

    actor_head = std::make_shared<CategoricalOutput>(hidden_dim, n_actions);

    // 应用Xavier初始化
    // xavier_init(std::dynamic_pointer_cast<LinearImpl>(shared[0]));
    // xavier_init(actor[0]->as<Linear>());
    // xavier_init(actor[2]->as<Linear>());
    // xavier_init(critic[0]->as<Linear>());
    // xavier_init(critic[2]->as<Linear>());
    // xavier_init(critic_head);

    // 注册模块
    register_module("shared", shared);
    register_module("actor", actor);
    register_module("critic", critic);
    register_module("critic_head", critic_head);
    register_module("actor_head", actor_head);

    train();
  }

  std::vector<torch::Tensor> act(torch::Tensor obs) {
    if (!obs.is_floating_point()) {
      obs = obs.to(torch::kFloat);
    }
    return forward(obs);
  }

  std::vector<torch::Tensor> forward(torch::Tensor obs) {
    if (obs.sizes().size() != 2) {
      throw std::runtime_error("Input tensor must have 2 dimensions");
    }

    auto input_obs_dim = obs.size(1);
    auto batch_size = obs.size(0);
    assert(input_obs_dim == obs_dim);

    // 获取权重和偏置
    // printf("Get weights and biases...\n");
    auto shared_fc_w = shared[0]
                           ->as<Linear>()
                           ->weight.data()
                           .cpu()
                           .contiguous()
                           .view({hidden_dim, input_obs_dim})
                           .data_ptr<float>();
    auto shared_fc_b =
        shared[0]->as<Linear>()->bias.data().cpu().contiguous().data_ptr<float>();
    auto actor_fc1_w = actor[0]
                           ->as<Linear>()
                           ->weight.data()
                           .cpu()
                           .contiguous()
                           .view({hidden_dim, hidden_dim})
                           .data_ptr<float>();
    auto actor_fc1_b =
        actor[0]->as<Linear>()->bias.data().cpu().contiguous().data_ptr<float>();
    auto actor_fc2_w = actor[2]
                           ->as<Linear>()
                           ->weight.data()
                           .cpu()
                           .contiguous()
                           .view({hidden_dim, hidden_dim})
                           .data_ptr<float>();
    auto actor_fc2_b =
        actor[2]->as<Linear>()->bias.data().cpu().contiguous().data_ptr<float>();
    auto critic_fc1_w = critic[0]
                            ->as<Linear>()
                            ->weight.data()
                            .cpu()
                            .contiguous()
                            .view({hidden_dim, hidden_dim})
                            .data_ptr<float>();
    auto critic_fc1_b =
        critic[0]->as<Linear>()->bias.data().cpu().contiguous().data_ptr<float>();
    auto critic_fc2_w = critic[2]
                            ->as<Linear>()
                            ->weight.data()
                            .cpu()
                            .contiguous()
                            .view({hidden_dim, hidden_dim})
                            .data_ptr<float>();
    auto critic_fc2_b =
        critic[2]->as<Linear>()->bias.data().cpu().contiguous().data_ptr<float>();
    auto actor_head_w = actor_head->as<CategoricalOutput>()
                            ->get_linear()
                            ->weight.data()
                            .cpu()
                            .contiguous()
                            .view({n_actions, hidden_dim})
                            .data_ptr<float>();
    auto actor_head_b = actor_head->as<CategoricalOutput>()
                            ->get_linear()
                            ->bias.data()
                            .cpu()
                            .contiguous()
                            .data_ptr<float>();
    auto critic_head_w = critic_head->weight.data()
                             .cpu()
                             .contiguous()
                             .view({1, hidden_dim})
                             .data_ptr<float>();
    auto critic_head_b =
        critic_head->bias.data().cpu().contiguous().data_ptr<float>();

    auto input = obs.cpu().contiguous().data_ptr<float>();

    // 分配输出内存
    float *actor_output_data = new float[batch_size * n_actions];
    float *critic_output_data = new float[batch_size * 1];

    // 调用CUDA前向传播函数
    // printf("Calling CUDA forward function...\n");
    cuda_forward(input, batch_size, input_obs_dim, hidden_dim, n_actions,
                 shared_fc_w, shared_fc_b, actor_fc1_w, actor_fc1_b, actor_fc2_w,
                 actor_fc2_b, actor_head_w, actor_head_b, critic_fc1_w,
                 critic_fc1_b, critic_fc2_w, critic_fc2_b, critic_head_w,
                 critic_head_b, actor_output_data, critic_output_data);

    // 将结果转换回PyTorch张量
    // printf("Converting results to PyTorch tensors...\n");
    auto actor_output = torch::from_blob(actor_output_data,
                                         {batch_size, n_actions}, torch::kFloat32)
                            .to(obs.device());
    auto critic_output =
        torch::from_blob(critic_output_data, {batch_size, 1}, torch::kFloat32)
            .to(obs.device());

    // 处理Actor输出（策略分布）
    // printf("Processing actor output...\n");
    auto dist = actor_head->forward(actor_output);
    auto action = dist->sample({});
    auto actLogProbs = dist->log_prob(action);

    delete[] actor_output_data;
    delete[] critic_output_data;

    // 更新缓存
    cached_obs = obs;
    auto x = shared->forward(obs);
    cached_x = x;
    cached_hidden_actor = actor->forward(x);
    cached_hidden_critic = critic->forward(x);
    cached_actor_output = dist->get_probs();
    cached_value = critic_output;

    // printf("Returning results...\n");
    return {action.to(torch::kFloat), critic_output, actLogProbs};
  }

  std::vector<torch::Tensor> evaluate_actions(torch::Tensor obs,
                                                   torch::Tensor actions) {
    if (obs.sizes().size() != 2) {
      throw std::runtime_error("Input tensor must have 2 dimensions");
    }

    auto input_obs_dim = obs.size(1);
    auto batch_size = obs.size(0);
    assert(input_obs_dim == obs_dim);

    // 获取权重和偏置
    auto shared_fc_w = shared[0]
                           ->as<Linear>()
                           ->weight.data()
                           .cpu()
                           .contiguous()
                           .view({hidden_dim, input_obs_dim})
                           .data_ptr<float>();
    auto shared_fc_b =
        shared[0]->as<Linear>()->bias.data().cpu().contiguous().data_ptr<float>();
    auto actor_fc1_w = actor[0]
                           ->as<Linear>()
                           ->weight.data()
                           .cpu()
                           .contiguous()
                           .view({hidden_dim, hidden_dim})
                           .data_ptr<float>();
    auto actor_fc1_b =
        actor[0]->as<Linear>()->bias.data().cpu().contiguous().data_ptr<float>();
    auto actor_fc2_w = actor[2]
                           ->as<Linear>()
                           ->weight.data()
                           .cpu()
                           .contiguous()
                           .view({hidden_dim, hidden_dim})
                           .data_ptr<float>();
    auto actor_fc2_b =
        actor[2]->as<Linear>()->bias.data().cpu().contiguous().data_ptr<float>();
    auto critic_fc1_w = critic[0]
                            ->as<Linear>()
                            ->weight.data()
                            .cpu()
                            .contiguous()
                            .view({hidden_dim, hidden_dim})
                            .data_ptr<float>();
    auto critic_fc1_b =
        critic[0]->as<Linear>()->bias.data().cpu().contiguous().data_ptr<float>();
    auto critic_fc2_w = critic[2]
                            ->as<Linear>()
                            ->weight.data()
                            .cpu()
                            .contiguous()
                            .view({hidden_dim, hidden_dim})
                            .data_ptr<float>();
    auto critic_fc2_b =
        critic[2]->as<Linear>()->bias.data().cpu().contiguous().data_ptr<float>();
    auto actor_head_w = actor_head->as<CategoricalOutput>()
                            ->get_linear()
                            ->weight.data()
                            .cpu()
                            .contiguous()
                            .view({n_actions, hidden_dim})
                            .data_ptr<float>();
    auto actor_head_b = actor_head->as<CategoricalOutput>()
                            ->get_linear()
                            ->bias.data()
                            .cpu()
                            .contiguous()
                            .data_ptr<float>();
    auto critic_head_w = critic_head->weight.data()
                             .cpu()
                             .contiguous()
                             .view({1, hidden_dim})
                             .data_ptr<float>();
    auto critic_head_b =
        critic_head->bias.data().cpu().contiguous().data_ptr<float>();

    auto input = obs.cpu().contiguous().data_ptr<float>();

    // 分配输出内存
    float *actor_output_data = new float[batch_size * n_actions];
    float *critic_output_data = new float[batch_size * 1];

    // 调用CUDA前向传播函数
    cuda_forward(input, batch_size, input_obs_dim, hidden_dim, n_actions,
                 shared_fc_w, shared_fc_b, actor_fc1_w, actor_fc1_b, actor_fc2_w,
                 actor_fc2_b, actor_head_w, actor_head_b, critic_fc1_w,
                 critic_fc1_b, critic_fc2_w, critic_fc2_b, critic_head_w,
                 critic_head_b, actor_output_data, critic_output_data);

    // 将结果转换回PyTorch张量
    auto actor_output = torch::from_blob(actor_output_data,
                                         {batch_size, n_actions}, torch::kFloat32)
                            .to(obs.device());
    auto critic_output =
        torch::from_blob(critic_output_data, {batch_size, 1}, torch::kFloat32)
            .to(obs.device());

    // 处理Actor输出（策略分布）
    auto dist = actor_head->forward(actor_output);
    // auto dist = std::make_shared<Categorical>(actor_output);
    auto actLogProbs = dist->log_prob(actions.squeeze(-1))
                           .view({actions.size(0), -1})
                           .sum(-1)
                           .unsqueeze(-1);

    auto distEntropy = dist->entropy();
    assert(distEntropy.sizes().size() == 1);
    distEntropy = distEntropy.unsqueeze(1);
    auto probs = dist->get_probs();

    delete[] actor_output_data;
    delete[] critic_output_data;

    // 更新缓存
    cached_obs = obs;
    auto x = shared->forward(obs);
    cached_x = x;
    cached_hidden_actor = actor->forward(x);
    cached_hidden_critic = critic->forward(x);
    cached_actor_output = probs;
    cached_value = critic_output;

    return {critic_output, actLogProbs, distEntropy, probs};
  }

  std::vector<torch::Tensor> get_intermediate_outputs(torch::Tensor obs) {
    if (obs.sizes().size() != 2) {
      throw std::runtime_error("Input tensor must have 2 dimensions");
    }

    auto input_obs_dim = obs.size(1);
    // auto batch_size = obs.size(0);
    assert(input_obs_dim == obs_dim);

    if (obs.equal(cached_obs)) {
      return {cached_obs, cached_x, cached_hidden_actor, cached_hidden_critic, cached_actor_output, cached_value};
    } else {
      // 如果输入不同，重新计算并更新缓存
      forward(obs);
      return {cached_obs, cached_x, cached_hidden_actor, cached_hidden_critic, cached_actor_output, cached_value};
    }
  }
};

} // namespace rl