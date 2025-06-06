#pragma once

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

#include "rl/distributions/distribution.h"

namespace rl
{
class Categorical : public Distribution
{
  private:
    torch::Tensor probs, logits, param;
    int num_events;

  public:
    Categorical(const torch::Tensor *probs, const torch::Tensor *logits);

    torch::Tensor entropy();
    torch::Tensor log_prob(torch::Tensor value);
    torch::Tensor sample(c10::ArrayRef<int64_t> sample_shape = {});

    inline torch::Tensor get_logits() { return logits; }
    inline torch::Tensor get_probs() { return probs; }
};
}