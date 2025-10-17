from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

class CustomDataset(Dataset):
    def __init__(self, S: np.ndarray, A: np.ndarray, R: np.ndarray, S_next: np.ndarray, Done: np.ndarray) -> None:
        """
        Initializes the dataset with state, action, reward, next state, and done flag arrays.

        Args:
            S (np.ndarray): Array of states.
            A (np.ndarray): Array of actions.
            R (np.ndarray): Array of rewards.
            S_next (np.ndarray): Array of next states.
            Done (np.ndarray): Array of done flags.
        """
        self.S = S
        self.A = A
        self.R = R
        self.S_next = S_next
        self.Done = Done
        
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        try: 
            assert len(self.S) == len(self.A) == len(self.R) == len(self.S_next) == len(self.Done)
        except AssertionError:
            raise ValueError("All input arrays must have the same length.")
        
        return len(self.S)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves the sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            A tuple containing (state, action, reward, next state, done flag).
        """
        return self.S[idx], self.A[idx], self.R[idx], self.S_next[idx], self.Done[idx]