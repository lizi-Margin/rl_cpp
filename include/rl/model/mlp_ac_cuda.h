#pragma once

#include "rl/model/ac.h"
#include "rl/model/output_layers.h"
#include <torch/nn/module.h>
#include <torch/torch.h>
#include <vector>

extern "C" void
cuda_forward(const float *input, int batch_size, int input_dim, int hidden_dim,
             int output_dim, const float *shared_w, const float *shared_b,
             const float *actor_fc1_w, const float *actor_fc1_b,
             const float *actor_fc2_w, const float *actor_fc2_b,
             const float *actor_head_w, const float *actor_head_b,
             const float *critic_fc1_w, const float *critic_fc1_b,
             const float *critic_fc2_w, const float *critic_fc2_b,
             const float *critic_head_w, const float *critic_head_b,
             float *actor_output, float *critic_output);

extern "C" void cuda_backward(
    const float *input, const float *shared_output, const float *actor_hidden,
    const float *critic_hidden, const float *actor_fc2_output,
    const float *critic_fc2_output, const float *actor_output,
    const float *critic_output, const float *grad_actor_output,
    const float *grad_critic_output, int batch_size, int input_dim,
    int hidden_dim, int output_dim, const float *actor_fc2_w,
    const float *critic_fc2_w, const float *actor_head_w,
    const float *critic_head_w, float *grad_shared_w, float *grad_shared_b,
    float *grad_actor_fc1_w, float *grad_actor_fc1_b, float *grad_actor_fc2_w,
    float *grad_actor_fc2_b, float *grad_actor_head_w, float *grad_actor_head_b,
    float *grad_critic_fc1_w, float *grad_critic_fc1_b,
    float *grad_critic_fc2_w, float *grad_critic_fc2_b,
    float *grad_critic_head_w, float *grad_critic_head_b);

using namespace torch;

namespace rl {

class MlpAC : public AC_Base {
private:
  unsigned int hidden_dim;

  nn::Sequential shared;
  nn::Sequential actor;
  nn::Sequential critic;
  std::shared_ptr<OutputLayer> actor_head;
  nn::Linear critic_head;

public:
  MlpAC(unsigned int obs_dim, unsigned int action_dim,
        unsigned int hidden_dim = 64);
  ~MlpAC() = default;

  std::vector<torch::Tensor> forward(torch::Tensor obs) override;
  std::vector<torch::Tensor> act(torch::Tensor obs) override;
  std::vector<torch::Tensor> evaluate_actions(torch::Tensor obs,
                                              torch::Tensor actions) override;
  std::vector<torch::Tensor> get_intermediate_outputs(torch::Tensor obs) override;
};
} // namespace rl