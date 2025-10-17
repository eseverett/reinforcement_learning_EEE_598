import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple

ARROWS = {0:"↑", 1:"→", 2:"↓", 3:"←"}

class Evaluator:
    
    def __init__(self, model: torch.nn.Module, device: torch.device, running_loss: list) -> None:
        
        """
        Initializes the evaluator with the model, device, and running loss.

        Args:
            model (torch.nn.Module): The trained neural network model.
            device (torch.device): Device to run the evaluation on (CPU or GPU).
            running_loss (list): List of loss values recorded during training.
        """
        self.model = model.to(device)
        self.device = device
        self.running_loss = running_loss
        
        
        
    def plot_running_loss(self) -> None:
        """
        Plots the running loss over training epochs.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.running_loss, label='Running Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Running Loss Over Epochs')
        plt.legend()
        plt.grid()
        plt.show()
        

    @torch.no_grad()
    def _greedy_action(self, s: np.ndarray) -> int:
        # s: shape (2,), e.g., [row, col]
        x = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.model(x)                      # [1, num_actions]
        return int(q.argmax(dim=1).item())

    def evaluate_episode(self, env, max_steps: int = 200) -> tuple[float, int, bool]:
        """Run one greedy episode (ε=0). Returns (total_reward, steps, success)."""
        s = env.reset()                        # expects [row, col]
        total_reward = 0.0
        success = False
        for t in range(1, max_steps + 1):
            a = self._greedy_action(s)
            s, r, done = env.step(a)   
            total_reward += float(r)
            if done:
                success = True
                break
        return total_reward, t, success

    def evaluate_policy(self, env, n_episodes: int = 20, max_steps: int = 200) -> dict:
        """Greedy evaluation over multiple episodes."""
        self.model.eval()
        returns = []
        lengths = []
        successes = 0
        with torch.no_grad():
            for _ in range(n_episodes):
                G, steps, ok = self.evaluate_episode(env, max_steps=max_steps)
                returns.append(G)
                lengths.append(steps)
                successes += int(ok)
        return {
            "avg_return": float(np.mean(returns)) if returns else 0.0,
            "std_return": float(np.std(returns)) if returns else 0.0,
            "avg_length": float(np.mean(lengths)) if lengths else 0.0,
            "success_rate": successes / max(1, n_episodes),
        }
        

    @torch.no_grad()
    def run_one_episode_with_trace(self, env, max_steps: int = 200, return_trace: bool = False):
        """Greedy (ε=0) episode with a step-by-step trace and an ASCII map."""
        self.model.eval()
        s = env.reset()  # expects [row, col]
        traj = []        # list of (state_tuple, action_int, reward_float)
        total_reward = 0.0

        for t in range(1, max_steps + 1):
            x = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = int(self.model(x).argmax(dim=1).item())
            s_next, r, done = env.step(a)
            traj.append((tuple(s), a, float(r)))
            total_reward += float(r)
            s = s_next
            if done:
                break

        # --- tabular trace ---
        print("Step |   State  | Act |   R")
        print("------------------------------")
        for i, (st, a, r) in enumerate(traj, 1):
            print(f"{i:>4} | {str(st):<8} |  {ARROWS.get(a, a)}  | {r:>4.1f}")
        print("------------------------------")
        print(f"Total reward: {total_reward:.2f}, steps: {len(traj)}")

        # --- ASCII path (arrows show action taken leaving that cell) ---
        if traj:
            rows = [st[0] for st, _, _ in traj] + [s[0]]
            cols = [st[1] for st, _, _ in traj] + [s[1]]
            H, W = max(rows) + 1, max(cols) + 1
            grid = np.full((H, W), "·", dtype=object)
            grid[traj[0][0]] = "S"
            for (st, a, _) in traj[:-1]:
                grid[st] = ARROWS.get(a, "?")
            grid[tuple(s)] = "G"  # final cell after last step
            print("\nPath (row 0 at top):")
            for r in range(H):
                print(" ".join(str(x) for x in grid[r]))

        if return_trace:
            return {"total_reward": total_reward, "steps": len(traj), "trajectory": traj}
        return {"total_reward": total_reward, "steps": len(traj)}