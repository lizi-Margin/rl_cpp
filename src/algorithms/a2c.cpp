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

  auto reward = traj.get_rewards();
  auto reward_dim = reward.size(-1);
  assert(reward_dim == 1);
  float total_episode_reward = reward.sum().item().toFloat();
  float mean_episode_reward = total_episode_reward / reward.size(0);

  int num_steps = traj.current_step();

  auto obs = traj.get_observations();
  auto action = traj.get_actions();

  // 确保数据在GPU上
  if (obs.device().type() != torch::kCUDA) {
    obs = obs.to(torch::kCUDA);
  }
  if (action.device().type() != torch::kCUDA) {
    action = action.to(torch::kCUDA);
  }

  // 执行前向传播，缓存中间结果
  auto v_logp_e_p = actor_and_critic->evaluate_actions(obs, action);

  for (size_t i = 0; i < v_logp_e_p.size(); ++i) {
    if (!v_logp_e_p[i].defined()) {
      throw std::runtime_error("Output tensor " + std::to_string(i) +
                               " from evaluate_actions is undefined");
    }
  }

  auto distEntropy = v_logp_e_p[2].view({1, num_steps, 1});
  auto values = v_logp_e_p[0].view({1, num_steps, 1});
  auto actLogProbs = v_logp_e_p[1].view({1, num_steps, 1});

  auto advantages =
      traj.get_returns().view({1, num_steps, 1}).to(torch::kCUDA) - values;

  auto value_loss = advantages.pow(2).mean();
  auto action_loss = -(advantages.detach() * actLogProbs).mean();
  auto distEntropy_loss = -distEntropy.mean();

  if (!value_loss.defined() || !action_loss.defined() ||
      !distEntropy_loss.defined()) {
    throw std::runtime_error("One of the loss components is undefined");
  }

  auto loss = (value_loss * value_loss_coef + action_loss * actor_loss_coef +
               distEntropy_loss * entropy_coef);

  auto mlp_ac = dynamic_cast<rl::MlpAC *>(actor_and_critic.get());

  // 使用CUDA梯度直接更新参数（避免数据传输回CPU）
  mlp_ac->update_parameters_with_cuda_gradients(learning_rate);

  return {{"Value loss", value_loss.item().toFloat()},
          {"Action loss", action_loss.item().toFloat()},
          {"Entropy loss", distEntropy_loss.item().toFloat()},
          {"Total Loss", loss.item().toFloat()},
          {"Total Episode Reward", total_episode_reward},
          {"Mean Episode Reward", mean_episode_reward},
          {"Num Steps", static_cast<float>(num_steps)}};
}
} // namespace rl