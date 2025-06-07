#pragma once

#include <memory>
#include <torch/torch.h>
#include <unordered_map>

#include "rl/model/ac.h"
#include "rl/model/mlp_ac_cuda.h"
#include "rl/traj.h"

namespace rl {

class A2C {
private:
  std::shared_ptr<::rl::MlpAC> actor_and_critic;
  float actor_loss_coef, value_loss_coef, entropy_coef, max_grad_norm,
      original_learning_rate;
  std::unique_ptr<torch::optim::SGD> optimizer;
  std::unordered_map<std::string, torch::Tensor> parameter_cache;

  float *grad_shared_w, *grad_shared_b;
  float *grad_actor_fc1_w, *grad_actor_fc1_b, *grad_actor_fc2_w,
      *grad_actor_fc2_b, *grad_actor_head_w, *grad_actor_head_b;
  float *grad_critic_fc1_w, *grad_critic_fc1_b, *grad_critic_fc2_w,
      *grad_critic_fc2_b, *grad_critic_head_w, *grad_critic_head_b;

public:
  inline A2C(std::shared_ptr<::rl::MlpAC> actor_and_critic,
             float actor_loss_coef, float value_loss_coef, float entropy_coef,
             float learning_rate, float epsilon = 1e-8, float alpha = 0.99,
             float max_grad_norm = 0.5)
      : actor_and_critic(actor_and_critic), actor_loss_coef(actor_loss_coef),
        value_loss_coef(value_loss_coef), entropy_coef(entropy_coef),
        max_grad_norm(max_grad_norm), original_learning_rate(learning_rate),
        optimizer(std::make_unique<torch::optim::SGD>(
            actor_and_critic->parameters(),
            torch::optim::SGDOptions(learning_rate))) {}

  std::unordered_map<std::string, float> update(Traj &traj);
  void update_parameters(float learning_rate = -1.0f);
};
} // namespace rl