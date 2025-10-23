import torch
import matplotlib.pyplot as plt
from typing import Tuple

from lib import GridWorld, ReplayBuffer, DQNModel, EvaluateResults, Agent

def run(rl_cfg: dict, training_cfg: dict) -> Tuple[Agent.DQN_Agent, list, list]:

    agent = Agent.DQN_Agent(rl_cfg, training_cfg)

    for _ in range(agent.num_episodes):
        agent.optimize_model()

    evaluator = EvaluateResults.EvaluateResults()

    traces, ep_returns, ep_losses = evaluator.evaluate_policy_greedy(training_cfg['model'], training_cfg['environment'], training_cfg['device'], training_cfg['loss_function'], rl_cfg['discount_factor'], episodes=250, max_steps=50)

    annotated = evaluator.render_trace_grid(training_cfg['environment'], traces[0])
    evaluator.print_trace_grid(annotated)
    print(" ")
    evaluator.print_policy_arrows(training_cfg['model'], training_cfg['environment'], training_cfg['device'])
    print(" ")
    
    return (agent, ep_returns, ep_losses)


def multi_plots(evaluator, agents: list, ep_returns_list: list, ep_losses_list: list, colors: list, label_name: str, label_quant: list) -> None:
    for i in range(len(agents)):
        evaluator.plot_loss_per_episode(agents[i].get_running_loss(), title='Training', color=colors[i], label=f'{label_name}={label_quant[i]}')
    # plt.show()
    plt.savefig(f'{label_name}_Loss_Training.png')
    plt.close()
        
    for i in range(len(agents)):
        evaluator.plot_return_per_episode(agents[i].get_running_return(), title='Training', color=colors[i], label=f'{label_name}={label_quant[i]}')
    # plt.show()
    plt.savefig(f'{label_name}_Return_Training.png')
    plt.close()
        
    for i in range(len(ep_losses_list)):
        evaluator.plot_loss_per_episode(ep_losses_list[i], title='Evaluation', color=colors[i], label=f'{label_name}={label_quant[i]}')
    # plt.show()
    plt.savefig(f'{label_name}_Loss_Evaluation.png')
    plt.close()
    
    for i in range(len(ep_returns_list)):
        evaluator.plot_return_per_episode(ep_returns_list[i], title='Evaluation', color=colors[i], label=f'{label_name}={label_quant[i]}')
    # plt.show()
    plt.savefig(f'{label_name}_Return_Evaluation.png')
    plt.close()
        
    agents = []
    ep_returns_list = []
    ep_losses_list = []

def main() -> None:
    
    print('===== Reinforcement Learning: DQN on GridWorld =====')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorld.GridWorld(penalty=-1.0, start_state=(0, 0))
    env.reset()
    
    replay_buffer_capacity = 10000
    replay_buffer_minibatch_size = 64

    replay_buffer = ReplayBuffer.ReplayBuffer(
        capacity=replay_buffer_capacity,
        sample_size=replay_buffer_minibatch_size,
        device=device
    )
    
    evaluator = EvaluateResults.EvaluateResults()
    
    agents = []
    ep_returns_list = []
    ep_losses_list = []
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    print('--- Vary Learning Rate ---')
    
    learning_rate_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    for i in range(len(learning_rate_list)):
        
        rl_cfg = {
            'learning_rate': learning_rate_list[i],
            'discount_factor': 0.95,
            'exploration_rate': 0.9,
        }
        
        training_cfg = {
            'num_episodes': 2000,
            'max_steps_per_episode': 50,
            'exploration_decay_rate': 0.995,
            'min_exploration_rate': 0.01,
            'loss_function': torch.nn.MSELoss(),
            'optimizer': torch.optim.Adam,
            'model': DQNModel.DQN_Model(input_dim=2, output_dim=4, hidden_layers=[64, 64]),
            'buffer': replay_buffer,
            'environment': env,
            'device': device,
            'target_update_steps': 100,
            'warmup_steps': 100,
            'gradient_steps_per_env_step': 1,
        }
        
        agent, ep_returns, ep_losses = run(rl_cfg, training_cfg)
        
        agents.append(agent)
        ep_returns_list.append(ep_returns)
        ep_losses_list.append(ep_losses)
        
    multi_plots(evaluator, agents, ep_returns_list, ep_losses_list, colors, 'Learning Rate', learning_rate_list)
    agents.clear(); ep_returns_list.clear(); ep_losses_list.clear();
    
    
    
    print('--- Vary Discount Factor ---')
    
    env.reset()
    replay_buffer.clear()
    
    discount_factor_list = [0.1, 0.3, 0.5, 0.7, 0.9]

    for i in range(len(discount_factor_list)):
        
        rl_cfg = {
            'learning_rate': 1e-4,
            'discount_factor': discount_factor_list[i],
            'exploration_rate': 0.9,
        }
        
        training_cfg = {
            'num_episodes': 2000,
            'max_steps_per_episode': 50,
            'exploration_decay_rate': 0.995,
            'min_exploration_rate': 0.01,
            'loss_function': torch.nn.MSELoss(),
            'optimizer': torch.optim.Adam,
            'model': DQNModel.DQN_Model(input_dim=2, output_dim=4, hidden_layers=[64, 64]),
            'buffer': replay_buffer,
            'environment': env,
            'device': device,
            'target_update_steps': 100,
            'warmup_steps': 100,
            'gradient_steps_per_env_step': 1,
        }
        
        agent, ep_returns, ep_losses = run(rl_cfg, training_cfg)
        
        agents.append(agent)
        ep_returns_list.append(ep_returns)
        ep_losses_list.append(ep_losses)
        
    multi_plots(evaluator, agents, ep_returns_list, ep_losses_list, colors, 'Discount Factor', discount_factor_list)
    agents.clear(); ep_returns_list.clear(); ep_losses_list.clear();



    print('--- Vary Model ---')
    
    env.reset()
    replay_buffer.clear()
    
    Model_list = [[128], [64, 64], [32, 32, 32], [16, 16, 16, 16], [8, 8, 8, 8, 8]]

    for i in range(len(Model_list)):
        
        rl_cfg = {
            'learning_rate': 1e-4,
            'discount_factor': 0.95,
            'exploration_rate': 0.9,
        }
        
        training_cfg = {
            'num_episodes': 2000,
            'max_steps_per_episode': 50,
            'exploration_decay_rate': 0.995,
            'min_exploration_rate': 0.01,
            'loss_function': torch.nn.MSELoss(),
            'optimizer': torch.optim.Adam,
            'model': DQNModel.DQN_Model(input_dim=2, output_dim=4, hidden_layers=Model_list[i]),
            'buffer': replay_buffer,
            'environment': env,
            'device': device,
            'target_update_steps': 100,
            'warmup_steps': 100,
            'gradient_steps_per_env_step': 1,
        }
        
        agent, ep_returns, ep_losses = run(rl_cfg, training_cfg)
        
        agents.append(agent)
        ep_returns_list.append(ep_returns)
        ep_losses_list.append(ep_losses)
        
    multi_plots(evaluator, agents, ep_returns_list, ep_losses_list, colors, 'Model', Model_list)
    agents.clear(); ep_returns_list.clear(); ep_losses_list.clear();
    
    
    
    print('--- Vary Batch Size ---')
    
    env.reset()
    replay_buffer.clear()
    
    exploration_rate_list = [0.1, 0.3, 0.5, 0.7, 0.9]

    for i in range(len(exploration_rate_list)):
        
        rl_cfg = {
            'learning_rate': 1e-4,
            'discount_factor': 0.95,
            'exploration_rate': 0.9,
        }
        
        training_cfg = {
            'num_episodes': 2000,
            'max_steps_per_episode': 50,
            'exploration_decay_rate': 0.995,
            'min_exploration_rate': 0.01,
            'loss_function': torch.nn.MSELoss(),
            'optimizer': torch.optim.Adam,
            'model': DQNModel.DQN_Model(input_dim=2, output_dim=4, hidden_layers=[64, 64]),
            'buffer': replay_buffer,
            'environment': env,
            'device': device,
            'target_update_steps': 100,
            'warmup_steps': 100,
            'gradient_steps_per_env_step': 1,
        }
        
        agent, ep_returns, ep_losses = run(rl_cfg, training_cfg)
        
        agents.append(agent)
        ep_returns_list.append(ep_returns)
        ep_losses_list.append(ep_losses)
        
    multi_plots(evaluator, agents, ep_returns_list, ep_losses_list, colors, 'Exploration Rate', exploration_rate_list)
    agents.clear(); ep_returns_list.clear(); ep_losses_list.clear();
    
    
    
    print('--- Vary Exploration Rate ---')
    
    env.reset()
    replay_buffer.clear()
    
    exploration_rate_list = [0.1, 0.3, 0.5, 0.7, 0.9]

    for i in range(len(exploration_rate_list)):
        
        rl_cfg = {
            'learning_rate': 1e-4,
            'discount_factor': 0.95,
            'exploration_rate': exploration_rate_list[i],
        }
        
        training_cfg = {
            'num_episodes': 2000,
            'max_steps_per_episode': 50,
            'exploration_decay_rate': 0.995,
            'min_exploration_rate': 0.01,
            'loss_function': torch.nn.MSELoss(),
            'optimizer': torch.optim.Adam,
            'model': DQNModel.DQN_Model(input_dim=2, output_dim=4, hidden_layers=[64, 64]),
            'buffer': replay_buffer,
            'environment': env,
            'device': device,
            'target_update_steps': 100,
            'warmup_steps': 100,
            'gradient_steps_per_env_step': 1,
        }
        
        agent, ep_returns, ep_losses = run(rl_cfg, training_cfg)
        
        agents.append(agent)
        ep_returns_list.append(ep_returns)
        ep_losses_list.append(ep_losses)
        
    multi_plots(evaluator, agents, ep_returns_list, ep_losses_list, colors, 'Exploration Rate', exploration_rate_list)
    agents.clear(); ep_returns_list.clear(); ep_losses_list.clear();
    
    
    
    print('--- Vary Batch Size ---')
    
    env.reset()
    replay_buffer.clear()
    
    batch_size_list = [1, 16, 32, 64, 128]

    for i in range(len(batch_size_list)):
        
        replay_buffer = ReplayBuffer.ReplayBuffer(
            capacity=replay_buffer_capacity,
            sample_size=batch_size_list[i],
            device=device
        )
        
        rl_cfg = {
            'learning_rate': 1e-4,
            'discount_factor': 0.95,
            'exploration_rate': 0.9,
        }
        
        training_cfg = {
            'num_episodes': 2000,
            'max_steps_per_episode': 50,
            'exploration_decay_rate': 0.995,
            'min_exploration_rate': 0.01,
            'loss_function': torch.nn.MSELoss(),
            'optimizer': torch.optim.Adam,
            'model': DQNModel.DQN_Model(input_dim=2, output_dim=4, hidden_layers=[64, 64]),
            'buffer': replay_buffer,
            'environment': env,
            'device': device,
            'target_update_steps': 100,
            'warmup_steps': 100,
            'gradient_steps_per_env_step': 1,
        }
        
        agent, ep_returns, ep_losses = run(rl_cfg, training_cfg)
        
        agents.append(agent)
        ep_returns_list.append(ep_returns)
        ep_losses_list.append(ep_losses)
        
    multi_plots(evaluator, agents, ep_returns_list, ep_losses_list, colors, 'Batch Size', batch_size_list)
    agents.clear(); ep_returns_list.clear(); ep_losses_list.clear();
    

    print('==== Finished =====')


if __name__ == "__main__":
    main()
