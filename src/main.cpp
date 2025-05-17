#include "rl/traj.h"
#include "rl/model/mlp_ac.h"
#include "rl/algorithms/a2c.h"
#include "draw/csv_drawer.h"
#include "rubbish_can.h"
#include <c10/core/ScalarType.h>
#include <cstdio>

draw::csv_drawer drawer("./train_info.csv");

void simple_game()
{
    // The game is: If the action matches the max dim of input, give a reward of 1, otherwise -1
    std::printf("Simple game starting...\n");

    unsigned int N_parallel = 1;
    unsigned int N_episode = 5000;
    unsigned int episode_steps = 20;

    unsigned int obs_dim = 2;
    unsigned int n_actions = 2;
    unsigned int hidden_dim = 128;

    auto ac = std::make_shared<::rl::MlpAC>(obs_dim, n_actions, hidden_dim);
    ::rl::Traj traj(episode_steps);
    ::rl::A2C trainer(static_cast<std::shared_ptr<::rl::AC_Base>>(ac), 1., 1., 0.1, 1e-3);
    
    auto obs = torch::randint(0, 10, {N_parallel, obs_dim}).to(torch::kFloat);

    for (unsigned int e = 0; e < N_episode; e += 1)
    {
        for (unsigned int step = 0; step < episode_steps; step += 1)
        {
            std::printf("\rEpisode %d, Step %d", e, step);
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
            // std::cout << "Value: " << value.sizes() << std::endl;
            // std::cout << "actLogProbs: " << actLogProbs.sizes() << std::endl;
            // std::cout << "Reward: " << reward << std::endl;
            // std::cout << "Done: " << done.sizes() << std::endl;

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
        std::printf("\r\n");

        // torch::Tensor next_value;
        // {
        //     torch::NoGradGuard no_grad;
        //     next_value = ac->get_values(traj.get_observations()[-1]).detach();
        // }

        std::printf("Episode %d begins to update:\n", e);
        auto train_info = trainer.update(traj);
        std::printf("Episode %d finished, Train info:\n", e);
        print_unordered_map(train_info);
        drawer.draw(train_info);
        traj.clear();
    }
}


int main() {
    at::set_num_threads(12);
    at::set_num_interop_threads(12);
    simple_game();
    return 0;
}