#pragma once

#include <torch/nn/module.h>
#include <vector>
#include <torch/torch.h>
#include "rl/model/output_layers.h"
#include "rl/model/ac.h"


using namespace torch;

namespace rl
{

class MlpAC : public AC_Base
{
  private:
    unsigned int hidden_dim;
    
    nn::Sequential shared;
    nn::Sequential actor;
    nn::Sequential critic;
    std::shared_ptr<OutputLayer> actor_head;
    nn::Linear critic_head;

  public:
    MlpAC(unsigned int obs_dim, unsigned int action_dim, unsigned int hidden_dim = 64);
    ~MlpAC() = default;

    std::vector<torch::Tensor> forward(torch::Tensor obs) override;
    std::vector<torch::Tensor> act(torch::Tensor obs) override;
    std::vector<torch::Tensor> evaluate_actions(torch::Tensor obs, torch::Tensor actions) override;
};

}