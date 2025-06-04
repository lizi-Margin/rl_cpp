#include <memory>

#include <torch/torch.h>

#include "rl/model/output_layers.h"
#include "rl/distributions/distribution.h"
#include "rl/distributions/categorical.h"
 

using namespace torch;

namespace rl
{

CategoricalOutput::CategoricalOutput(unsigned int num_inputs,
                                     unsigned int num_outputs)
    : linear(num_inputs, num_outputs)
{
    register_module("linear", linear);
}

std::unique_ptr<Distribution> CategoricalOutput::forward(torch::Tensor x)
{
    // x = linear(x);
    return std::make_unique<Categorical>(nullptr, &x);
}

}