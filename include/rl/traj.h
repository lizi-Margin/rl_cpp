#pragma once

#include <ATen/core/TensorBody.h>
#include <c10/core/DeviceType.h>
#include <cassert>
#include <cstdio>
#include <torch/torch.h>
#include <vector>
#include "rubbish_can.h"

namespace rl {

class Traj {
private:
  std::vector<torch::Tensor> observations_;
  std::vector<torch::Tensor> actions_;
  std::vector<torch::Tensor> rewards_;
  std::vector<torch::Tensor> action_log_probs_;
  std::vector<torch::Tensor> values_;
  std::vector<torch::Tensor> dones_;

  std::vector<torch::Tensor> returns_;

  float gamma_ = 0.99;

  size_t current_step_ = 0;
  size_t max_steps_;
  torch::Device device_;

  bool frozen_ = false;

public:
  Traj(size_t max_steps) : max_steps_(max_steps), device_(torch::kCPU) {}

  void remember(torch::Tensor observation, torch::Tensor action,
                torch::Tensor reward,
                torch::Tensor action_log_prob = torch::Tensor(),
                torch::Tensor value = torch::Tensor(),
                torch::Tensor done = torch::Tensor()) {

    if (frozen_) {
      throw std::runtime_error("Cannot remember to a frozen trajectory");
    }

    if (current_step_ >= max_steps_) {
      throw std::runtime_error("Trajectory buffer is full");
    }

    observations_.push_back(observation.to(device_));
    actions_.push_back(action.to(device_));
    rewards_.push_back(reward.to(device_));

    if (action_log_prob.defined()) {
      action_log_probs_.push_back(action_log_prob.to(device_));
    }
    if (value.defined()) {
      values_.push_back(value.to(device_));
    }
    if (done.defined()) {
      dones_.push_back(done.to(device_));
    }

    current_step_++;
  }

  void clear() {
    auto logger = get_logger("Traj::clear");
    logger->debug("called");
    observations_.clear();
    actions_.clear();
    rewards_.clear();
    action_log_probs_.clear();
    values_.clear();
    dones_.clear();
    current_step_ = 0;
    frozen_ = false;
    logger->debug("returned");
  }

  size_t current_step() const { return current_step_; }

  size_t max_steps() const { return max_steps_; }

  void freeze() { frozen_ = true; }

  void finalize() {
    if (frozen_) {
      throw std::runtime_error("Trajectory frozen. Cannot finalize.");
    }
    cut_tail();
    compute_returns();
    freeze();
  }

  torch::Tensor get_observations() const {
    if (observations_.empty())
      return torch::Tensor();
    return torch::stack(observations_);
  }

  torch::Tensor get_actions() const {
    if (actions_.empty())
      return torch::Tensor();
    return torch::stack(actions_);
  }

  torch::Tensor get_rewards() const {
    if (rewards_.empty())
      return torch::Tensor();
    return torch::stack(rewards_);
  }

  torch::Tensor get_action_log_probs() const {
    if (action_log_probs_.empty())
      return torch::Tensor();
    return torch::stack(action_log_probs_);
  }

  torch::Tensor get_values() const {
    if (values_.empty())
      return torch::Tensor();
    return torch::stack(values_);
  }

  torch::Tensor get_dones() const {
    if (dones_.empty())
      return torch::Tensor();
    return torch::stack(dones_);
  }

  torch::Tensor get_returns() const {
    if (returns_.empty())
      return torch::Tensor();
    return torch::stack(returns_);
  }

private:
  void cut_tail() {
    auto traj_length = current_step_;

    if (observations_.size() > traj_length) {
      observations_.resize(traj_length);
    }
    if (actions_.size() > traj_length) {
      actions_.resize(traj_length);
    }
    if (rewards_.size() > traj_length) {
      rewards_.resize(traj_length);
    }
    if (action_log_probs_.size() > traj_length) {
      action_log_probs_.resize(traj_length);
    }
    if (values_.size() > traj_length) {
      values_.resize(traj_length);
    }
    if (dones_.size() > traj_length) {
      dones_.resize(traj_length);
    }
  }

  void compute_returns() {
    if (current_step_ == 0) {
      throw std::runtime_error("Trajectory is empty. Cannot compute returns.");
    }

    auto traj_len = rewards_.size();
    assert(traj_len == current_step_);
    assert(traj_len == rewards_.size());

    std::vector<torch::Tensor> returns(traj_len);

    torch::Tensor running_return{torch::zeros_like(rewards_[0])};
    for (int i = traj_len - 1; i >= 0; --i) {
      running_return = rewards_[i] + gamma_ * running_return;
      returns[i] = running_return;
    }

    returns_ = std::move(returns);
  }
};

} // namespace rl