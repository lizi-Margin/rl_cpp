#include "rl/model/ac.h"
#include "rl/model/output_layers.h"
#include <torch/nn/module.h>
#include <torch/torch.h>
#include <vector>

// CUDA函数声明
extern "C" void cuda_initialize(int max_batch_size, int input_dim, int hidden_dim, int output_dim);
extern "C" void cuda_cleanup();
extern "C" void cuda_forward(const float *input, int batch_size, int input_dim, int hidden_dim,
             int output_dim, const float *shared_w, const float *shared_b,
             const float *actor_fc1_w, const float *actor_fc1_b,
             const float *actor_fc2_w, const float *actor_fc2_b,
             const float *actor_head_w, const float *actor_head_b,
             const float *critic_fc1_w, const float *critic_fc1_b,
             const float *critic_fc2_w, const float *critic_fc2_b,
             const float *critic_head_w, const float *critic_head_b,
             float *actor_output, float *critic_output,
             float *shared_output, float *actor_hidden, float *critic_hidden,
             float *actor_fc2_output, float *critic_fc2_output);

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
  unsigned int max_batch_size;

  nn::Sequential shared;
  nn::Sequential actor;
  nn::Sequential critic;
  std::shared_ptr<OutputLayer> actor_head;
  nn::Linear critic_head;

  // 缓存的中间结果（保持在GPU上）
  torch::Tensor cached_shared_output;
  torch::Tensor cached_actor_hidden;
  torch::Tensor cached_critic_hidden;
  torch::Tensor cached_actor_output;
  torch::Tensor cached_critic_output;
  torch::Tensor cached_actor_fc2_output;
  torch::Tensor cached_critic_fc2_output;

  // CUDA内存缓冲区
  float* cuda_shared_output;
  float* cuda_actor_hidden;
  float* cuda_critic_hidden;
  float* cuda_actor_fc2_output;
  float* cuda_critic_fc2_output;
  float* cuda_actor_output;
  float* cuda_critic_output;

public:
  MlpAC(unsigned int obs_dim, unsigned int action_dim,
        unsigned int hidden_dim = 64, unsigned int max_batch_size = 1024);
  ~MlpAC();

  std::vector<torch::Tensor> forward(torch::Tensor obs) override;
  std::vector<torch::Tensor> act(torch::Tensor obs) override;
  std::vector<torch::Tensor> evaluate_actions(torch::Tensor obs,
                                              torch::Tensor actions) override;
  unsigned int get_hidden_dim() const;
  torch::Tensor get_actor_fc2_w() const;
  torch::Tensor get_critic_fc2_w() const;
  torch::Tensor get_actor_head_w() const;
  torch::Tensor get_critic_head_w() const;
  
  // 获取GPU上的缓存输出
  std::vector<torch::Tensor> get_cached_outputs() const;
  
  // 使用CUDA梯度更新参数
  void update_parameters_with_cuda_gradients(float learning_rate);
  
  // 直接获取CUDA梯度指针（用于优化器）
  std::vector<float*> get_cuda_gradients() const;
};
} // namespace rl