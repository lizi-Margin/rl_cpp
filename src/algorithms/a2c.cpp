#include "rl/algorithms/a2c.h"
#include "rl/model/mlp_ac_cuda.h"
#include <ATen/ops/ones_like.h>
#include <c10/core/ScalarType.h>
#include <cassert>
#include <cstdio>
#include <memory>
#include <torch/optim/sgd.h>
#include <torch/torch.h>
#include <unordered_map>

namespace rl {

std::unordered_map<std::string, float> A2C::update(Traj &traj) {
  printf("traj finalize...\n");
  traj.finalize();

  if (!traj.get_observations().defined()) {
    throw std::runtime_error("Observations tensor is undefined");
  }
  if (!traj.get_actions().defined()) {
    throw std::runtime_error("Actions tensor is undefined");
  }
  if (!traj.get_rewards().defined()) {
    throw std::runtime_error("Rewards tensor is undefined");
  }
  if (!traj.get_returns().defined()) {
    throw std::runtime_error("Returns tensor is undefined");
  }

  printf("Calculating loss...\n");

  auto reward = traj.get_rewards();
  auto reward_dim = reward.size(-1);
  assert(reward_dim == 1);
  float total_episode_reward = reward.sum().item().toFloat();
  float mean_episode_reward = total_episode_reward / reward.size(0);

  int num_steps = traj.current_step();

  auto obs = traj.get_observations();
  auto action = traj.get_actions();

  std::cout << "obs: " << obs << std::endl;
  std::cout << "action: " << action << std::endl;

  auto v_logp_e_p = actor_and_critic->evaluate_actions(obs, action);

  for (size_t i = 0; i < v_logp_e_p.size(); ++i) {
    if (!v_logp_e_p[i].defined()) {
      throw std::runtime_error("Output tensor " + std::to_string(i) +
                               " from evaluate_actions is undefined");
    }
  }

  printf("Calculating loss...\n");

  auto distEntropy = v_logp_e_p[2].view({1, num_steps, 1});
  auto values = v_logp_e_p[0].view({1, num_steps, 1});
  auto actLogProbs = v_logp_e_p[1].view({1, num_steps, 1});

  auto advantages = traj.get_returns().view({1, num_steps, 1}) - values;

  auto value_loss = advantages.pow(2).mean();
  auto action_loss = -(advantages.detach() * actLogProbs).mean();
  auto distEntropy_loss = -distEntropy.mean();

  if (!value_loss.defined() || !action_loss.defined() ||
      !distEntropy_loss.defined()) {
    throw std::runtime_error("One of the loss components is undefined");
  }

  auto loss = (value_loss * value_loss_coef + action_loss * actor_loss_coef +
               distEntropy_loss * entropy_coef);

  //   Step optimizer
  //   optimizer->zero_grad();

  //   计算梯度
  printf("Calculating gradients...\n");
  auto grad_actor_output = actLogProbs.cpu().contiguous().data_ptr<float>();
  auto grad_critic_output = advantages.cpu().contiguous().data_ptr<float>();

  auto batch_size = obs.size(0);
  auto input_dim = obs.size(1);

  actor_and_critic = std::dynamic_pointer_cast<MlpAC>(actor_and_critic);
  auto hidden_dim = actor_and_critic->get_hidden_dim();
  auto output_dim = actor_and_critic->get_n_actions();

  // get gradients
  auto mlp_ac = std::dynamic_pointer_cast<MlpAC>(actor_and_critic);
  auto shared_fc_w =
      mlp_ac->get_shared_fc_w().cpu().contiguous().data_ptr<float>();
  auto shared_fc_b =
      mlp_ac->get_shared_fc_b().cpu().contiguous().data_ptr<float>();
  auto actor_fc1_w =
      mlp_ac->get_actor_fc1_w().cpu().contiguous().data_ptr<float>();
  auto actor_fc1_b =
      mlp_ac->get_actor_fc1_b().cpu().contiguous().data_ptr<float>();
  auto actor_fc2_w =
      mlp_ac->get_actor_fc2_w().cpu().contiguous().data_ptr<float>();
  auto actor_fc2_b =
      mlp_ac->get_actor_fc2_b().cpu().contiguous().data_ptr<float>();
  auto actor_head_w =
      mlp_ac->get_actor_head_w().cpu().contiguous().data_ptr<float>();
  auto actor_head_b =
      mlp_ac->get_actor_head_b().cpu().contiguous().data_ptr<float>();
  auto critic_fc1_w =
      mlp_ac->get_critic_fc1_w().cpu().contiguous().data_ptr<float>();
  auto critic_fc1_b =
      mlp_ac->get_critic_fc1_b().cpu().contiguous().data_ptr<float>();
  auto critic_fc2_w =
      mlp_ac->get_critic_fc2_w().cpu().contiguous().data_ptr<float>();
  auto critic_fc2_b =
      mlp_ac->get_critic_fc2_b().cpu().contiguous().data_ptr<float>();
  auto critic_head_w =
      mlp_ac->get_critic_head_w().cpu().contiguous().data_ptr<float>();
  auto critic_head_b =
      mlp_ac->get_critic_head_b().cpu().contiguous().data_ptr<float>();

  // 分配梯度内存，使用智能指针
  auto grad_shared_w = std::make_unique<float[]>(hidden_dim * input_dim);
  auto grad_shared_b = std::make_unique<float[]>(hidden_dim);
  auto grad_actor_fc1_w = std::make_unique<float[]>(hidden_dim * hidden_dim);
  auto grad_actor_fc1_b = std::make_unique<float[]>(hidden_dim);
  auto grad_actor_fc2_w = std::make_unique<float[]>(output_dim * hidden_dim);
  auto grad_actor_fc2_b = std::make_unique<float[]>(output_dim);
  auto grad_actor_head_w = std::make_unique<float[]>(output_dim * output_dim);
  auto grad_actor_head_b = std::make_unique<float[]>(output_dim);
  auto grad_critic_fc1_w = std::make_unique<float[]>(hidden_dim * hidden_dim);
  auto grad_critic_fc1_b = std::make_unique<float[]>(hidden_dim);
  auto grad_critic_fc2_w = std::make_unique<float[]>(output_dim * hidden_dim);
  auto grad_critic_fc2_b = std::make_unique<float[]>(output_dim);
  auto grad_critic_head_w = std::make_unique<float[]>(output_dim * output_dim);
  auto grad_critic_head_b = std::make_unique<float[]>(output_dim);

  // 调用CUDA反向传播函数
  printf("Calling CUDA backward function...\n");
  cuda_backward(
      obs.cpu().contiguous().data_ptr<float>(),
      actor_and_critic->get_intermediates(), grad_actor_output,
      grad_critic_output, batch_size, input_dim, hidden_dim, output_dim,
      actor_fc2_w, critic_fc2_w, actor_head_w, critic_head_w,
      grad_shared_w.get(), grad_shared_b.get(), grad_actor_fc1_w.get(),
      grad_actor_fc1_b.get(), grad_actor_fc2_w.get(), grad_actor_fc2_b.get(),
      grad_actor_head_w.get(), grad_actor_head_b.get(), grad_critic_fc1_w.get(),
      grad_critic_fc1_b.get(), grad_critic_fc2_w.get(), grad_critic_fc2_b.get(),
      grad_critic_head_w.get(), grad_critic_head_b.get());

  // 更新参数
  printf("Updating parameters...\n");
  auto update_param = [&](float *param, const float *grad, int size) {
    for (int i = 0; i < size; ++i) {
      param[i] -= original_learning_rate * grad[i];
    }
  };

  update_param(shared_fc_w, grad_shared_w.get(), hidden_dim * input_dim);
  update_param(shared_fc_b, grad_shared_b.get(), hidden_dim);
  update_param(actor_fc1_w, grad_actor_fc1_w.get(), hidden_dim * hidden_dim);
  update_param(actor_fc1_b, grad_actor_fc1_b.get(), hidden_dim);
  update_param(actor_fc2_w, grad_actor_fc2_w.get(), output_dim * hidden_dim);
  update_param(actor_fc2_b, grad_actor_fc2_b.get(), output_dim);
  update_param(actor_head_w, grad_actor_head_w.get(), output_dim * output_dim);
  update_param(actor_head_b, grad_actor_head_b.get(), output_dim);
  update_param(critic_fc1_w, grad_critic_fc1_w.get(), hidden_dim * hidden_dim);
  update_param(critic_fc1_b, grad_critic_fc1_b.get(), hidden_dim);
  update_param(critic_fc2_w, grad_critic_fc2_w.get(), output_dim * hidden_dim);
  update_param(critic_fc2_b, grad_critic_fc2_b.get(), output_dim);
  update_param(critic_head_w, grad_critic_head_w.get(),
               output_dim * output_dim);
  update_param(critic_head_b, grad_critic_head_b.get(), output_dim);

  return {{"Value loss", value_loss.item().toFloat()},
          {"Action loss", action_loss.item().toFloat()},
          {"distEntropy loss", distEntropy_loss.item().toFloat()},
          {"Loss", loss.item().toFloat()},
          {"toatal_episode_reward", total_episode_reward},
          {"mean_episode_reward", mean_episode_reward},
          {"num_steps", num_steps}};
}
} // namespace rl