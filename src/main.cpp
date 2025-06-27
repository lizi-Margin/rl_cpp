#include "rl/traj.h"
#include "rl/model/mlp_ac_cuda.h"
#include "rl/algorithms/a2c.h"
#include "draw/csv_drawer.h"
#include "rubbish_can.h"
#include <c10/core/ScalarType.h>
#include <cassert>
#include <chrono>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

draw::csv_drawer drawer("./train_info.csv");

void simple_game()
{
    auto logger = get_logger("simple_game()");

    // The game is: If the action matches the max dim of input, give a reward of 1, otherwise -1
    logger->info("Simple game starting...");

    unsigned int N_parallel = 1;
    unsigned int N_episode = 5000;
    unsigned int episode_steps = 20;

    unsigned int obs_dim = 2;
    unsigned int n_actions = 2;
    unsigned int hidden_dim = 128;

    auto ac = std::make_shared<::rl::MlpAC>(obs_dim, n_actions, hidden_dim);
    ::rl::Traj traj(episode_steps);
    ::rl::A2C trainer(static_cast<std::shared_ptr<::rl::MlpAC>>(ac), 1., 1., 0.1, 1e-3);
    
    auto obs = torch::randint(0, 10, {N_parallel, obs_dim}).to(torch::kFloat);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (unsigned int e = 0; e < N_episode; e += 1)
    {
        logger->info("Episode {} begins", e);
        for (unsigned int step = 0; step < episode_steps; step += 1)
        {
            logger->debug("Episode {}, Step {}", e, step);
            std::vector<torch::Tensor> a_v_logp;
            {
                torch::NoGradGuard no_grad;
                a_v_logp = ac->act(obs);
            }
            auto action = a_v_logp[0];
            auto value = a_v_logp[1];
            auto actLogProbs = a_v_logp[2];
            auto reward = (action == obs.argmax(1).unsqueeze(1)).to(torch::kFloat) * 2 - 1;
            auto done = e == episode_steps - 1 ? \
                torch::ones({N_parallel, 1}) : torch::zeros({N_parallel, 1});
            
            // std::cout << "Obs: " << obs << std::endl;
            // std::cout << "Action: " << action << std::endl;
            // std::cout << "Value: " << value << std::endl;
            // std::cout << "actLogProbs: " << actLogProbs << std::endl;
            // std::cout << "Reward: " << reward << std::endl;
            // std::cout << "Done: " << done << std::endl;

            traj.remember(
                obs[0],
                action[0],
                reward[0],
                actLogProbs[0],
                value[0],
                done[0] 
            );
            
            obs = torch::randint(0, 10, {N_parallel, obs_dim}).to(torch::kFloat);
        }
        logger->info("Episode {} ends", e);

        // torch::Tensor next_value;
        // {
        //     torch::NoGradGuard no_grad;
        //     next_value = ac->get_values(traj.get_observations()[-1]).detach();
        // }

        logger->info("Episode {} begins to update:\n", e);
        auto train_info = trainer.update(traj);
        logger->info("Episode {} finished, Train info:\n", e);
        print_unordered_map(train_info);
        drawer.draw(train_info);
        traj.clear();
        logger->info("Episode %d done.\n", e);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count(); 

    logger->info("Training completed in {} seconds.\n", duration);
}


int main() {
    spdlog::set_pattern("%^[%l]%$ [%s:%# %!] %v");
    spdlog::set_level(spdlog::level::trace);
    at::set_num_threads(12);
    at::set_num_interop_threads(12);
    simple_game();
    return 0;
}