#include <algorithm>
#include <vector>

#include <torch/torch.h>

#include "rl/generators/recurrent_generator.h"
#include "rl/generators/generator.h"
 

namespace rl
{
torch::Tensor flatten_helper(int timesteps, int processes, torch::Tensor tensor)
{
    auto tensor_shape = tensor.sizes().vec();
    tensor_shape.erase(tensor_shape.begin());
    tensor_shape[0] = timesteps * processes;
    return tensor.view(tensor_shape);
}

RecurrentGenerator::RecurrentGenerator(int num_processes,
                                       int num_mini_batch,
                                       torch::Tensor observations,
                                       torch::Tensor hidden_states,
                                       torch::Tensor actions,
                                       torch::Tensor value_predictions,
                                       torch::Tensor returns,
                                       torch::Tensor masks,
                                       torch::Tensor action_log_probs,
                                       torch::Tensor advantages)
    : observations(observations),
      hidden_states(hidden_states),
      actions(actions),
      value_predictions(value_predictions),
      returns(returns),
      masks(masks),
      action_log_probs(action_log_probs),
      advantages(advantages),
      indices(torch::randperm(num_processes, torch::TensorOptions(torch::kLong))),
      index(0),
      num_envs_per_batch(num_processes / num_mini_batch) {}

bool RecurrentGenerator::done() const
{
    return index >= indices.size(0);
}

MiniBatch RecurrentGenerator::next()
{
    if (index >= indices.size(0))
    {
        throw std::runtime_error("No minibatches left in generator.");
    }

    MiniBatch mini_batch;

    // Fill minibatch with tensors of shape (timestep, process, *whatever)
    // Except hidden states, that is just (process, *whatever)
    int64_t env_index = indices[index].item().toLong();
    mini_batch.observations = observations
                                  .narrow(0, 0, observations.size(0) - 1)
                                  .narrow(1, env_index, num_envs_per_batch);
    mini_batch.hidden_states = hidden_states[0]
                                   .narrow(0, env_index, num_envs_per_batch)
                                   .view({num_envs_per_batch, -1});
    mini_batch.actions = actions.narrow(1, env_index, num_envs_per_batch);
    mini_batch.value_predictions = value_predictions
                                       .narrow(0, 0, value_predictions.size(0) - 1)
                                       .narrow(1, env_index, num_envs_per_batch);
    mini_batch.returns = returns.narrow(0, 0, returns.size(0) - 1)
                             .narrow(1, env_index, num_envs_per_batch);
    mini_batch.masks = masks.narrow(0, 0, masks.size(0) - 1)
                           .narrow(1, env_index, num_envs_per_batch);
    mini_batch.action_log_probs = action_log_probs.narrow(1, env_index,
                                                          num_envs_per_batch);
    mini_batch.advantages = advantages.narrow(1, env_index, num_envs_per_batch);

    // Flatten tensors to (timestep * process, *whatever)
    int num_timesteps = mini_batch.observations.size(0);
    int num_processes = num_envs_per_batch;
    mini_batch.observations = flatten_helper(num_timesteps, num_processes,
                                             mini_batch.observations);
    mini_batch.actions = flatten_helper(num_timesteps, num_processes,
                                        mini_batch.actions);
    mini_batch.value_predictions = flatten_helper(num_timesteps, num_processes,
                                                  mini_batch.value_predictions);
    mini_batch.returns = flatten_helper(num_timesteps, num_processes,
                                        mini_batch.returns);
    mini_batch.masks = flatten_helper(num_timesteps, num_processes,
                                      mini_batch.masks);
    mini_batch.action_log_probs = flatten_helper(num_timesteps, num_processes,
                                                 mini_batch.action_log_probs);
    mini_batch.advantages = flatten_helper(num_timesteps, num_processes,
                                           mini_batch.advantages);

    index++;

    return mini_batch;
}


}