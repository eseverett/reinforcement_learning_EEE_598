import torch
from typing import Tuple

class ReplayBuffer:
    
    def __init__(self, capacity: int, sample_size: int, device: torch.device) -> None:
        
        """
        Initializes the replay buffer with a given capacity.
        Args:
            capacity (int): Maximum number of experiences to store in the buffer.
            sample_size (int): Number of experiences to sample during training.
            device (torch.device): The device to store the tensors on.
        """
        
        self.capacity = capacity
        
        self.state = torch.zeros((capacity, 2), dtype=torch.float32).to(device)
        self.action = torch.zeros((capacity, 1), dtype=torch.int64).to(device)
        self.reward = torch.zeros((capacity, 1), dtype=torch.float32).to(device)
        self.next_state = torch.zeros((capacity, 2), dtype=torch.float32).to(device)
        self.done = torch.zeros((capacity, 1), dtype=torch.float32).to(device)
        
        self.ptr = 0
        self.size = 0
        self.batch_size = sample_size
        
    def add(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool) -> None:
        """
        Adds a new experience to the replay buffer.
        Args:
            state (torch.Tensor): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (torch.Tensor): The next state after taking the action.
            done (bool): Whether the episode has ended.
        """
        
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a batch of experiences from the replay buffer.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        indices = torch.randint(0, self.size, (self.batch_size,), device=self.state.device)
        return (
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.next_state[indices],
            self.done[indices]
        )
        
    def clear(self) -> None:
        """
        Clears the replay buffer.
        """
        self.ptr = 0
        self.size = 0
        
