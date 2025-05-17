#pragma once

#include <memory>
#include <unordered_map>
#include <torch/torch.h>

#include "rl/traj.h"
#include "rl/model/ac.h"

namespace rl
{

class A2C
{
  private:
    std::shared_ptr<::rl::AC_Base> actor_and_critic;
    float actor_loss_coef, value_loss_coef, entropy_coef, max_grad_norm, original_learning_rate;
    std::unique_ptr<torch::optim::SGD> optimizer;

  public:
    inline A2C(
      std::shared_ptr<::rl::AC_Base> actor_and_critic,
      float actor_loss_coef,
      float value_loss_coef,
      float entropy_coef,
      float learning_rate,
      float epsilon = 1e-8,
      float alpha = 0.99,
      float max_grad_norm = 0.5
    )
      : actor_and_critic(actor_and_critic),
        actor_loss_coef(actor_loss_coef),
        value_loss_coef(value_loss_coef),
        entropy_coef(entropy_coef),
        max_grad_norm(max_grad_norm),
        original_learning_rate(learning_rate),
        optimizer(std::make_unique<torch::optim::SGD>(
          actor_and_critic->parameters(),
          torch::optim::SGDOptions(learning_rate))
        ) {}

    std::unordered_map<std::string, float> update(Traj &traj);
};
}