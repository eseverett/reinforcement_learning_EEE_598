from lib import data, evaluate, grid, model, train, generator
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss

def main() -> None:
    
    epsilon = 0.5
    learning_rate = 1e-4
    epochs = 10
    batch_size = 128
    
    env = grid.GridWorld()
    data_gen = generator.DataGenerator(env, epsilon=epsilon)

    S, A, R, S_next, Done = data_gen.generate_data(steps=50000, seed=42)

    
    dataset = data.CustomDataset(S, A, R, S_next, Done)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dqn_model = model.DQN(input_dim=2, output_dim=4)
    optimizer = Adam(dqn_model.parameters(), lr=learning_rate)
    loss_fn = MSELoss()
    gamma = 0.95
    
    training_loop = train.TrainingLoop(dqn_model, dataloader, optimizer, loss_fn, device, gamma)
    
    running_loss = []
    for epoch in range(epochs):
        loss = training_loop.train_epoch()
        running_loss.append(loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        
    evaluator = evaluate.Evaluator(dqn_model, device, running_loss)
    evaluator.plot_running_loss()
    
    metrics1 = evaluator.evaluate_policy(env, n_episodes=30)
    print(metrics1)
    metrics2 = evaluator.run_one_episode_with_trace(env)
    print(metrics2)


if __name__ == "__main__":
    main()