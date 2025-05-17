#pragma once
#include <ATen/core/TensorBody.h>
#include <torch/torch.h>
#include <unordered_map>

namespace env{

typedef std::unordered_map<std::string, torch::Tensor> tensor_dict_t;

class EnvBase {
protected:
  unsigned int obs_dim;
  unsigned int n_actions;
  unsigned int reward_dim = 1;
  unsigned int done_dim = 1;
public:
  inline unsigned int get_obs_dim() { return obs_dim; }
  inline unsigned int get_n_actions() { return n_actions; }
  inline unsigned int get_reward_dim() { return reward_dim; }
  inline unsigned int get_done_dim() { return done_dim; }

  virtual tensor_dict_t step(tensor_dict_t action) = 0;
  virtual tensor_dict_t reset() = 0;
  virtual tensor_dict_t reset_render() = 0;
};

}