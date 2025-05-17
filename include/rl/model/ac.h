#pragma once

#include <torch/torch.h>
#include <torch/nn/module.h>

namespace rl {

class AC_Base : public torch::nn::Module
{
  public:
    AC_Base(unsigned int obs_dim, unsigned int n_actions) : obs_dim(obs_dim), n_actions(n_actions) {};
  protected:
    unsigned int obs_dim;
    unsigned int n_actions;
    unsigned int reward_dim = 1;
  public: 
    virtual std::vector<torch::Tensor> forward(torch::Tensor obs) = 0;
    virtual std::vector<torch::Tensor> act(torch::Tensor obs) = 0;
    virtual std::vector<torch::Tensor> evaluate_actions(torch::Tensor obs, torch::Tensor actions) = 0;
};

}