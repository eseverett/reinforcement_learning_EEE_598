import numpy as np
from typing import Tuple
import random

class DataGenerator:
    
    def __init__(self, env, epsilon: float, init: str = "zeros", Tmax: int = 100) -> None:
        """
        Initializes the GridWorld data generator with an internal Q-table.

        Args:
            env: The GridWorld environment instance.
            epsilon (float): Exploration rate for ε-greedy policy.
            init (str): Method to initialize the Q-table ("zeros" or "random").
            Tmax (int): Maximum number of steps per episode before reset.
        """
        
        self.env = env
        self.epsilon = epsilon
        self.Tmax = Tmax
        
        # Initialize Q-table same shape as GridWorld (3x4x4)
        if init == "zeros":
            self.q_table = np.zeros((3, 4, 4), dtype=float)
        elif init == "random":
            self.q_table = np.random.rand(3, 4, 4)
        else:
            raise ValueError("init must be 'zeros' or 'random'")

    def choose_action(self, state: Tuple[int, int]) -> int:
        """
        Selects an action using ε-greedy policy:
        chooses random action with probability ε, else argmax from Q-table.
        """
        r, c = state
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return int(np.argmax(self.q_table[r, c, :]))

    def generate_data(self, steps: int, seed: int = None) -> Tuple[np.ndarray, ...]:
        """
        Generates (s, a, r, s_next, done) transitions from the environment.

        Args:
            steps (int): Number of transitions to collect.
            seed (int): Optional random seed for reproducibility.

        Returns:
            Tuple of NumPy arrays: (S, A, R, S_next, Done)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        S, A, R, S_next, Done = [], [], [], [], []
        s = self.env.reset()
        t = 0

        for _ in range(steps):
            # ε-greedy action
            a = self.choose_action(s)
            
            # Step environment
            s_next, r, done = self.env.step(a)

            # Store transition
            S.append(s)
            A.append(a)
            R.append(r)
            S_next.append(s_next)
            Done.append(done)

            # Reset when episode ends or Tmax reached
            t += 1
            if done or t >= self.Tmax:
                s = self.env.reset()
                t = 0
            else:
                s = s_next

        # Convert lists to NumPy arrays
        return (
            np.array(S, dtype=np.int32),
            np.array(A, dtype=np.int32),
            np.array(R, dtype=np.float32),
            np.array(S_next, dtype=np.int32),
            np.array(Done, dtype=bool),
        )
