#pragma once

#include "rl/model/ac.h"
#include "rl/model/output_layers.h"
#include <torch/nn/module.h>
#include <torch/torch.h>
#include <vector>

struct IntermediateData {
  float *d_shared_output;
  float *d_actor_hidden;
  float *d_critic_hidden;
  float *d_actor_fc2_output;
  float *d_critic_fc2_output;
  float *d_actor_output;
  float *d_critic_output;
};

extern "C" {
void cuda_forward(const float *input, int batch_size, int input_dim,
                  int hidden_dim, int output_dim, const float *shared_w,
                  const float *shared_b, const float *actor_fc1_w,
                  const float *actor_fc1_b, const float *actor_fc2_w,
                  const float *actor_fc2_b, const float *actor_head_w,
                  const float *actor_head_b, const float *critic_fc1_w,
                  const float *critic_fc1_b, const float *critic_fc2_w,
                  const float *critic_fc2_b, const float *critic_head_w,
                  const float *critic_head_b, float *actor_output,
                  float *critic_output, IntermediateData *intermediates);

void cuda_backward(const float *input, IntermediateData *fwd_data,
                   const float *grad_actor_output,
                   const float *grad_critic_output, int batch_size,
                   int input_dim, int hidden_dim, int output_dim,
                   const float *actor_fc2_w, const float *critic_fc2_w,
                   const float *actor_head_w, const float *critic_head_w,
                   float *grad_shared_w, float *grad_shared_b,
                   float *grad_actor_fc1_w, float *grad_actor_fc1_b,
                   float *grad_actor_fc2_w, float *grad_actor_fc2_b,
                   float *grad_actor_head_w, float *grad_actor_head_b,
                   float *grad_critic_fc1_w, float *grad_critic_fc1_b,
                   float *grad_critic_fc2_w, float *grad_critic_fc2_b,
                   float *grad_critic_head_w, float *grad_critic_head_b);

void cuda_alloc_mem(IntermediateData *data, int batch_size, int input_dim,
                    int hidden_dim, int output_dim);
void cuda_free_mem(IntermediateData *data);
}

using namespace torch;

namespace rl {

class MlpAC : public AC_Base {
private:
  unsigned int obs_dim;
  unsigned int hidden_dim;
  unsigned int action_dim;

  nn::Sequential shared;
  nn::Sequential actor;
  nn::Sequential critic;
  std::shared_ptr<OutputLayer> actor_head;
  nn::Linear critic_head;

  IntermediateData intermediates;

public:
  MlpAC(unsigned int obs_dim, unsigned int action_dim,
        unsigned int hidden_dim = 64);
  ~MlpAC() = default;

  std::vector<torch::Tensor> forward(torch::Tensor obs) override;
  std::vector<torch::Tensor> act(torch::Tensor obs) override;
  std::vector<torch::Tensor> evaluate_actions(torch::Tensor obs,
                                              torch::Tensor actions) override;

  void update();
  auto get_hidden_dim() const -> unsigned int { return hidden_dim; }
  auto get_n_actions() const -> unsigned int { return action_dim; }
  auto get_shared_fc_w() const -> torch::Tensor {
    return shared->ptr(0)->as<nn::Linear>()->weight;
  }
  auto get_shared_fc_b() const -> torch::Tensor {
    return shared->ptr(0)->as<nn::Linear>()->bias;
  }
  auto get_actor_fc1_w() const -> torch::Tensor {
    return actor->ptr(0)->as<nn::Linear>()->weight;
  }
  auto get_actor_fc1_b() const -> torch::Tensor {
    return actor->ptr(0)->as<nn::Linear>()->bias;
  }
  auto get_actor_head_w() const -> torch::Tensor {
    return actor_head->as<CategoricalOutput>()->get_linear()->weight;
  }
  auto get_actor_head_b() const -> torch::Tensor {
    return actor_head->as<CategoricalOutput>()->get_linear()->bias;
  }
  auto get_actor_fc2_w() const -> torch::Tensor {
    return actor->ptr(2)->as<nn::Linear>()->weight;
  }
  auto get_actor_fc2_b() const -> torch::Tensor {
    return actor->ptr(2)->as<nn::Linear>()->bias;
  }
  auto get_critic_fc1_w() const -> torch::Tensor {
    return critic->ptr(0)->as<nn::Linear>()->weight;
  }
  auto get_critic_fc1_b() const -> torch::Tensor {
    return critic->ptr(0)->as<nn::Linear>()->bias;
  }
  auto get_critic_fc2_w() const -> torch::Tensor {
    return critic->ptr(2)->as<nn::Linear>()->weight;
  }
  auto get_critic_fc2_b() const -> torch::Tensor {
    return critic->ptr(2)->as<nn::Linear>()->bias;
  }
  auto get_critic_head_w() const -> torch::Tensor {
    return critic_head->weight;
  }
  auto get_critic_head_b() const -> torch::Tensor { return critic_head->bias; }

  auto get_intermediates() { return &intermediates; }
};
} // namespace rl