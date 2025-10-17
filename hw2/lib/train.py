import torch
import numpy as np
from typing import Any
from torch.utils.data import DataLoader

class TrainingLoop:
    
    def __init__(self, model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: Any, device: torch.device, gamma: float) -> None:
        """
        Initializes the training loop with model, dataloader, optimizer, loss function, and device.

        Args:
            model (torch.nn.Module): The neural network model to be trained.
            dataloader (DataLoader): DataLoader providing the training data.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            loss_fn (Any): Loss function to compute the training loss.
            device (torch.device): Device to run the training on (CPU or GPU).
            gamma (float): Discount factor for future rewards.
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.gamma = gamma
        
    def train_epoch(self) -> float:
        """
        Trains the model for one epoch over the entire dataset.

        Returns:
            float: The average loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_count = 0

        for batch in self.dataloader:
            S, A, R, S_next, Done = [x.to(self.device) for x in batch]
            S = S.float()
            S_next = S_next.float()
            R = R.float()
            Done = Done.float()
            A = A.long()

            self.optimizer.zero_grad()

            # Q(s, ·), then select Q(s, a)
            q_all = self.model(S)
            q_sa = q_all.gather(1, A.unsqueeze(1)).squeeze(1)

            # Bellman targets without a target network (stop-grad on current net)
            with torch.no_grad():
                next_q_max = self.model(S_next).max(1).values
                y = R + self.gamma * (1.0 - Done) * next_q_max

            loss = self.loss_fn(q_sa, y)
            loss.backward()
            self.optimizer.step()

            batch_size = S.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

        return total_loss / total_count if total_count > 0 else 0.0
