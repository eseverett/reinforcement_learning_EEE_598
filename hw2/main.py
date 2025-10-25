import torch
import matplotlib.pyplot as plt
import numpy as np
import random

from lib import GridWorld, ReplayBuffer, DQNModel, EvaluateResults, Agent

def main() -> None:
    print('===== Reinforcement Learning: DQN on GridWorld =====')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    seed = 4
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print('Using seed:', seed)

    print('===== Initializing GridWorld Environment =====')
    env = GridWorld.GridWorld(penalty=-1.0, start_state=(0, 0))

    print('===== Initializing Replay Buffer =====')
    replay_buffer_capacity = 10000
    replay_buffer_minibatch_size = 64
    print(f'Replay Buffer Capacity: {replay_buffer_capacity}, Minibatch Size: {replay_buffer_minibatch_size}')
    replay_buffer = ReplayBuffer.ReplayBuffer(
        capacity=replay_buffer_capacity,
        sample_size=replay_buffer_minibatch_size,
        device=device
    )

    print('==== Initializing Agent and Model =====')
    
    rl_cfg = {
        'learning_rate': 1e-4,
        'discount_factor': 0.95,
        'exploration_rate': 0.9,
    }
    
    

    model = DQNModel.DQN_Model(input_dim=2, output_dim=4, hidden_layers=[64, 64])

    print(f'Learning Rate: {rl_cfg["learning_rate"]}, Discount Factor: {rl_cfg["discount_factor"]}, Exploration Rate: {rl_cfg["exploration_rate"]}')

    training_cfg = {
        'num_episodes': 2000,
        'max_steps_per_episode': 50,
        'exploration_decay_rate': 0.995,
        'min_exploration_rate': 0.01,
        'loss_function': torch.nn.MSELoss(),
        'optimizer': torch.optim.Adam,
        'model': model,
        'buffer': replay_buffer,
        'environment': env,
        'device': device,
        'target_update_steps': 100,
        'warmup_steps': 100,
        'gradient_steps_per_env_step': 1,
    }

    agent = Agent.DQN_Agent(rl_cfg, training_cfg)

    print('===== Starting Training =====')
    for episode in range(agent.num_episodes):
        agent.optimize_model()
        ep_idx = episode + 1
        ep_return = agent.get_running_return()[-1]
        ep_loss = agent.get_running_loss()[-1]
        
        if ep_idx % 50 == 0 or ep_idx == 1:
            print(f'Episode {ep_idx}/{agent.num_episodes} | Return: {ep_return:.3f} | Loss: {ep_loss:.3f} | Exploration Rate: {agent.exploration_rate:.3f}')

    print('===== Starting Evaluation =====')
    evaluator = EvaluateResults.EvaluateResults()
    evaluator.plot_loss_per_episode(agent.get_running_loss(), title=f'Training_seed_{seed}')
    plt.savefig(f'loss_plot_seed_{seed}.png')
    plt.close()
    
    evaluator.plot_return_per_episode(agent.get_running_return(), title=f'Training_seed_{seed}')
    plt.savefig(f'return_plot_seed_{seed}.png')
    plt.close()

    traces, ep_returns, ep_losses = evaluator.evaluate_policy_greedy(model, env, device, training_cfg['loss_function'], rl_cfg['discount_factor'], episodes=250, max_steps=50)
    
    evaluator.plot_loss_per_episode(ep_losses, title=f'Evaluation_seed_{seed}')
    plt.savefig(f'eval_loss_plot_seed_{seed}.png')
    plt.close()
    
    evaluator.plot_return_per_episode(ep_returns, title=f'Evaluation_seed_{seed}')
    plt.savefig(f'eval_return_plot_seed_{seed}.png')
    plt.close()


    annotated = evaluator.render_trace_grid(env, traces[0])
    evaluator.print_trace_grid(annotated)
    print(" ")
    evaluator.print_policy_arrows(model, env, device)

    print('==== Finished =====')


if __name__ == "__main__":
    main()
