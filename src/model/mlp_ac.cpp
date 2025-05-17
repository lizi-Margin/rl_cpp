#include <c10/core/ScalarType.h>
#include <cassert>
#include <torch/torch.h>

#include "rl/model/mlp_ac.h"
#include "rl/model/output_layers.h"

 

rl::MlpAC::MlpAC(unsigned int obs_dim, unsigned int n_actions, unsigned int hidden_dim)
  : 
    AC_Base(obs_dim, n_actions),
    hidden_dim(hidden_dim),
    shared(nullptr),
    actor(nullptr),
    critic(nullptr),
    actor_head(nullptr),
    critic_head(nullptr)
{

  shared= nn::Sequential(nn::Linear(obs_dim, hidden_dim),
                          nn::Functional(torch::tanh));
  actor = nn::Sequential(nn::Linear(hidden_dim, hidden_dim),
                          nn::Functional(torch::tanh),
                          nn::Linear(hidden_dim, hidden_dim),
                          nn::Functional(torch::tanh));
  critic = nn::Sequential(nn::Linear(hidden_dim, hidden_dim),
                          nn::Functional(torch::tanh),
                          nn::Linear(hidden_dim, hidden_dim),
                          nn::Functional(torch::tanh));
  critic_head = nn::Linear(hidden_dim, 1);

  actor_head = std::make_shared<CategoricalOutput>(
    hidden_dim,
    n_actions
  );

  register_module("shared", shared);
  register_module("actor", actor);
  register_module("critic", critic);
  register_module("critic_head", critic_head);
  register_module("actor_head", actor_head);

  train();
}


std::vector<torch::Tensor> rl::MlpAC::act(torch::Tensor obs)
{
  if (!obs.is_floating_point()) {
    obs = obs.to(torch::kFloat);
  }
  return forward(obs);
}

std::vector<torch::Tensor> rl::MlpAC::forward(torch::Tensor obs)
{
  if (obs.sizes().size() != 2) {
    throw std::runtime_error("Input tensor must have 2 dimensions");
  }

  auto input_obs_dim = obs.size(1);
  auto batch_size = obs.size(0);
  assert(input_obs_dim == obs_dim);
  assert(batch_size == 1);


  auto x = shared->forward(obs);

  auto hidden_critic = critic->forward(x);
  auto hidden_actor = actor->forward(x);

  auto value = critic_head->forward(hidden_critic);
  auto dist = actor_head->forward(hidden_actor);

  auto action = dist->sample({});
  auto actLogProbs = dist->log_prob(action);

  return {action.to(torch::kFloat), value, actLogProbs};
}


std::vector<torch::Tensor> rl::MlpAC::evaluate_actions(
  torch::Tensor obs,
  torch::Tensor actions
)
{
  auto x = shared->forward(obs);

  auto hidden_critic = critic->forward(x);
  auto hidden_actor = actor->forward(x);

  auto value = critic_head->forward(hidden_critic);
  auto dist = actor_head->forward(hidden_actor);

  auto actLogProbs = dist->log_prob(actions.squeeze(-1))
                              .view({actions.size(0), -1})
                              .sum(-1)
                              .unsqueeze(-1);

  auto distEntropy = dist->entropy();
  assert(distEntropy.sizes().size() == 1);
  distEntropy = distEntropy.unsqueeze(1);
  auto probs = dist->get_probs();

  return {value, actLogProbs, distEntropy, probs};
}
