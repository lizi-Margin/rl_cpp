#include <c10/core/ScalarType.h>
#include <cassert>
#include <cuda_runtime.h>
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

MlpAC::~MlpAC() {
  // 释放CUDA分配的内存
  cudaFree(cuda_shared_output);
  cudaFree(cuda_actor_hidden);
  cudaFree(cuda_critic_hidden);
  cudaFree(cuda_actor_fc2_output);
  cudaFree(cuda_critic_fc2_output);
  cudaFree(cuda_actor_output);
  cudaFree(cuda_critic_output);

  // 释放梯度缓冲区
  cudaFree(grad_shared_w);
  cudaFree(grad_shared_b);
  cudaFree(grad_actor_fc1_w);
  cudaFree(grad_actor_fc1_b);
  cudaFree(grad_actor_fc2_w);
  cudaFree(grad_actor_fc2_b);
  cudaFree(grad_actor_head_w);
  cudaFree(grad_actor_head_b);
  cudaFree(grad_critic_fc1_w);
  cudaFree(grad_critic_fc1_b);
  cudaFree(grad_critic_fc2_w);
  cudaFree(grad_critic_fc2_b);
  cudaFree(grad_critic_head_w);
  cudaFree(grad_critic_head_b);

  // 清理CUDA环境
  cuda_cleanup();
}

unsigned int MlpAC::get_hidden_dim() const { return hidden_dim; }

torch::Tensor MlpAC::get_actor_fc2_w() const {
  return actor->ptr(2)->as<Linear>()->weight.data();
}

torch::Tensor MlpAC::get_critic_fc2_w() const {
  return critic->ptr(2)->as<Linear>()->weight.data();
}

torch::Tensor MlpAC::get_actor_head_w() const {
  return actor_head->get_linear()->weight.data();
}

torch::Tensor MlpAC::get_critic_head_w() const {
  return critic_head->weight.data();
}

std::vector<torch::Tensor> MlpAC::get_cached_outputs() const {
  return {cached_shared_output,    cached_actor_hidden,
          cached_critic_hidden,    cached_actor_output,
          cached_critic_output,    cached_actor_fc2_output,
          cached_critic_fc2_output};
}

MlpAC::MlpAC(unsigned int obs_dim, unsigned int n_actions,
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
  // xavier_init(shared[0]->as<Linear>());
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

  // 初始化CUDA环境
  cuda_initialize(max_batch_size, obs_dim, hidden_dim, n_actions);

  train();
}

std::vector<torch::Tensor> MlpAC::act(torch::Tensor obs) {
  if (!obs.is_floating_point()) {
    obs = obs.to(torch::kFloat);
  }
  return forward(obs);
}

std::vector<torch::Tensor> MlpAC::forward(torch::Tensor obs) {
  if (obs.sizes().size() != 2) {
    throw std::runtime_error("Input tensor must have 2 dimensions");
  }

  auto input_obs_dim = obs.size(1);
  auto batch_size = obs.size(0);
  assert(input_obs_dim == obs_dim);
  assert(batch_size <= max_batch_size);

  // 确保输入在GPU上
  if (obs.device().type() != torch::kCUDA) {
    obs = obs.to(torch::kCUDA);
  }

  // 手动前向传播
  manual_forward(obs);

  // 处理Actor输出（策略分布）
  auto dist = actor_head->forward(cached_actor_output);
  auto action = dist->sample({});
  auto actLogProbs = dist->log_prob(action);

  return {action.to(torch::kFloat), cached_critic_output, actLogProbs};
}

std::vector<torch::Tensor> MlpAC::evaluate_actions(torch::Tensor obs,
                                                   torch::Tensor actions) {
  if (obs.sizes().size() != 2) {
    throw std::runtime_error("Input tensor must have 2 dimensions");
  }

  auto input_obs_dim = obs.size(1);
  auto batch_size = obs.size(0);
  assert(input_obs_dim == obs_dim);
  assert(batch_size <= max_batch_size);

  // 确保输入在GPU上
  if (obs.device().type() != torch::kCUDA) {
    obs = obs.to(torch::kCUDA);
  }
  if (actions.device().type() != torch::kCUDA) {
    actions = actions.to(torch::kCUDA);
  }

  // 手动前向传播
  manual_forward(obs);

  auto dist = actor_head->forward(cached_actor_output);
  auto actLogProbs = dist->log_prob(actions.squeeze(-1))
                         .view({actions.size(0), -1})
                         .sum(-1)
                         .unsqueeze(-1);

  auto distEntropy = dist->entropy();
  assert(distEntropy.sizes().size() == 1);
  distEntropy = distEntropy.unsqueeze(1);
  auto probs = dist->get_probs();

  return {cached_critic_output, actLogProbs, distEntropy, probs};
}

void MlpAC::manual_forward(torch::Tensor obs) {
  auto input_obs_dim = obs.size(1);
  auto batch_size = obs.size(0);

  // 获取权重和偏置的GPU指针
  auto shared_fc_w =
      shared[0]->as<Linear>()->weight.data().contiguous().data_ptr<float>();
  auto shared_fc_b =
      shared[0]->as<Linear>()->bias.data().contiguous().data_ptr<float>();
  auto actor_fc1_w =
      actor[0]->as<Linear>()->weight.data().contiguous().data_ptr<float>();
  auto actor_fc1_b =
      actor[0]->as<Linear>()->bias.data().contiguous().data_ptr<float>();
  auto actor_fc2_w =
      actor[2]->as<Linear>()->weight.data().contiguous().data_ptr<float>();
  auto actor_fc2_b =
      actor[2]->as<Linear>()->bias.data().contiguous().data_ptr<float>();
  auto critic_fc1_w =
      critic[0]->as<Linear>()->weight.data().contiguous().data_ptr<float>();
  auto critic_fc1_b =
      critic[0]->as<Linear>()->bias.data().contiguous().data_ptr<float>();
  auto critic_fc2_w =
      critic[2]->as<Linear>()->weight.data().contiguous().data_ptr<float>();
  auto critic_fc2_b =
      critic[2]->as<Linear>()->bias.data().contiguous().data_ptr<float>();
  auto actor_head_w =
      actor_head->get_linear()->weight.data().contiguous().data_ptr<float>();
  auto actor_head_b =
      actor_head->get_linear()->bias.data().contiguous().data_ptr<float>();
  auto critic_head_w =
      critic_head->weight.data().contiguous().data_ptr<float>();
  auto critic_head_b = critic_head->bias.data().contiguous().data_ptr<float>();

  // 获取输入的GPU指针
  auto input = obs.contiguous().data_ptr<float>();

  // 调用CUDA前向传播函数（所有计算在GPU上完成）
  cuda_forward(input, batch_size, input_obs_dim, hidden_dim, n_actions,
               shared_fc_w, shared_fc_b, actor_fc1_w, actor_fc1_b, actor_fc2_w,
               actor_fc2_b, actor_head_w, actor_head_b, critic_fc1_w,
               critic_fc1_b, critic_fc2_w, critic_fc2_b, critic_head_w,
               critic_head_b, cuda_actor_output, cuda_critic_output,
               cuda_shared_output, cuda_actor_hidden, cuda_critic_hidden,
               cuda_actor_fc2_output, cuda_critic_fc2_output);

  // 更新缓存的中间结果
  cached_shared_output =
      torch::from_blob(cuda_shared_output, {batch_size, hidden_dim},
                       torch::TensorOptions().device(torch::kCUDA));
  cached_actor_hidden =
      torch::from_blob(cuda_actor_hidden, {batch_size, hidden_dim},
                       torch::TensorOptions().device(torch::kCUDA));
  cached_critic_hidden =
      torch::from_blob(cuda_critic_hidden, {batch_size, hidden_dim},
                       torch::TensorOptions().device(torch::kCUDA));
  cached_actor_output =
      torch::from_blob(cuda_actor_output, {batch_size, n_actions},
                       torch::TensorOptions().device(torch::kCUDA));
  cached_critic_output =
      torch::from_blob(cuda_critic_output, {batch_size, 1},
                       torch::TensorOptions().device(torch::kCUDA));
  cached_actor_fc2_output =
      torch::from_blob(cuda_actor_fc2_output, {batch_size, hidden_dim},
                       torch::TensorOptions().device(torch::kCUDA));
  cached_critic_fc2_output =
      torch::from_blob(cuda_critic_fc2_output, {batch_size, hidden_dim},
                       torch::TensorOptions().device(torch::kCUDA));
}

void MlpAC::manual_backward(torch::Tensor grad_actor_output,
                            torch::Tensor grad_critic_output) {
  auto batch_size = cached_shared_output.size(0);
  auto input_dim = cached_shared_output.size(1);

  auto actor_fc2_w = get_actor_fc2_w().contiguous().data_ptr<float>();
  auto critic_fc2_w = get_critic_fc2_w().contiguous().data_ptr<float>();
  auto actor_head_w = get_actor_head_w().contiguous().data_ptr<float>();
  auto critic_head_w = get_critic_head_w().contiguous().data_ptr<float>();

  // 调用CUDA反向传播函数（所有计算在GPU上完成）
  cuda_backward(
      cached_shared_output.contiguous().data_ptr<float>(), // input
      cached_shared_output.contiguous().data_ptr<float>(), // shared_output
      cached_actor_hidden.contiguous().data_ptr<float>(),  // actor_hidden
      cached_critic_hidden.contiguous().data_ptr<float>(), // critic_hidden
      cached_actor_fc2_output.contiguous()
          .data_ptr<float>(), // actor_fc2_output
      cached_critic_fc2_output.contiguous()
          .data_ptr<float>(),                              // critic_fc2_output
      cached_actor_output.contiguous().data_ptr<float>(),  // actor_output
      cached_critic_output.contiguous().data_ptr<float>(), // critic_output
      grad_actor_output.contiguous().data_ptr<float>(),
      grad_critic_output.contiguous().data_ptr<float>(), batch_size, input_dim,
      hidden_dim, n_actions, actor_fc2_w, critic_fc2_w, actor_head_w,
      critic_head_w, grad_shared_w, grad_shared_b, grad_actor_fc1_w,
      grad_actor_fc1_b, grad_actor_fc2_w, grad_actor_fc2_b, grad_actor_head_w,
      grad_actor_head_b, grad_critic_fc1_w, grad_critic_fc1_b,
      grad_critic_fc2_w, grad_critic_fc2_b, grad_critic_head_w,
      grad_critic_head_b);
}

void MlpAC::update_parameters(float learning_rate) {
  // 共享层
  shared[0]->as<nn::Linear>()->weight.data() -=
      learning_rate * (*grad_shared_w);
  shared[0]->as<nn::Linear>()->bias.data() -= learning_rate * (*grad_shared_b);

  // 策略网络
  actor[0]->as<nn::Linear>()->weight.data() -=
      learning_rate * (*grad_actor_fc1_w);
  actor[0]->as<nn::Linear>()->bias.data() -=
      learning_rate * (*grad_actor_fc1_b);
  actor[2]->as<nn::Linear>()->weight.data() -=
      learning_rate * (*grad_actor_fc2_w);
  actor[2]->as<nn::Linear>()->bias.data() -=
      learning_rate * (*grad_actor_fc2_b);
  actor_head->get_linear()->weight.data() -=
      learning_rate * (*grad_actor_head_w);
  actor_head->get_linear()->bias.data() -= learning_rate * (*grad_actor_head_b);

  // 价值网络
  critic[0]->as<nn::Linear>()->weight.data() -=
      learning_rate * (*grad_critic_fc1_w);
  critic[0]->as<nn::Linear>()->bias.data() -=
      learning_rate * (*grad_critic_fc1_b);
  critic[2]->as<nn::Linear>()->weight.data() -=
      learning_rate * (*grad_critic_fc2_w);
  critic[2]->as<nn::Linear>()->bias.data() -=
      learning_rate * (*grad_critic_fc2_b);
  critic_head->weight.data() -= learning_rate * (*grad_critic_head_w);
  critic_head->bias.data() -= learning_rate * (*grad_critic_head_b);
}

void MlpAC::update_parameters_with_cuda_gradients(float learning_rate) {
  // 手动反向传播
  auto intermediate_outputs = get_cached_outputs();
  auto grad_actor_output = intermediate_outputs[4].contiguous();
  auto grad_critic_output = intermediate_outputs[5].contiguous();
  manual_backward(grad_actor_output, grad_critic_output);

  // 拷贝梯度回主机
  copy_gradients_to_host();

  // 更新参数
  update_parameters(learning_rate);
}

std::vector<float *> MlpAC::get_cuda_gradients() const {
  return {grad_shared_w,      grad_shared_b,     grad_actor_fc1_w,
          grad_actor_fc1_b,   grad_actor_fc2_w,  grad_actor_fc2_b,
          grad_actor_head_w,  grad_actor_head_b, grad_critic_fc1_w,
          grad_critic_fc1_b,  grad_critic_fc2_w, grad_critic_fc2_b,
          grad_critic_head_w, grad_critic_head_b};
}
} // namespace rl