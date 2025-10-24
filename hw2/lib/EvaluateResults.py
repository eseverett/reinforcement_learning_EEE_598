import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple

class EvaluateResults:
    
    def __init__(self) -> None:
        pass
    
    def plot_loss_per_episode(self, losses: list, title: str, color: str = 'blue', label: str = 'Loss per Episode') -> None:
        """
        Plots the loss per episode.
        Args:
            losses (list): Loss values per episode.
            title (str): Title for the plot.
            color (str): Color for the plot lines.
            label (str): Label for the plot legend.
        """
        window = 10
        weights = np.ones(window) / window
        
        plt.plot(losses, label=label, alpha=0.3, color=color)
        plt.plot(np.convolve(losses, weights, mode='same'), linewidth=2, color=color)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('DQN ' + title + ' Loss per Episode')
        plt.legend()
        plt.grid(True)

    def plot_return_per_episode(self, returns: list, title: str, color: str = 'blue', label: str = 'Return per Episode') -> None:
        """
        Plots the episodic return.
        Args:
            returns (list): Cumulative reward per episode.
            title (str): Title for the plot.
            color (str): Color for the plot lines.
            label (str): Label for the plot legend.
        """
        window = 10
        weights = np.ones(window) / window

        plt.plot(returns, label=label, alpha=0.3, color=color)
        plt.plot(np.convolve(returns, weights, mode='same'), linewidth=2, color=color)
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('DQN ' + title + ' Return per Episode')
        plt.legend()
        plt.grid(True)

    def evaluate_policy_greedy(
        self,
        model: torch.nn.Module,
        env,
        device: torch.device,
        loss_fn: torch.nn.Module,
        discount_factor: float = 0.99,
        episodes: int = 10,
        max_steps: int = 50
    ) -> Tuple[float, List[List[Tuple[int, int]]], list, list]:
        """
        Runs greedy evaluation (epsilon = 0) for a number of episodes.
        Uses the same loss function as training for TD error evaluation.
        Args:
            model (torch.nn.Module): The trained DQN model.
            env: The GridWorld environment.
            device (torch.device): The device to run computations on.
            loss_fn (torch.nn.Module): The loss function used during training.
            discount_factor (float): Discount factor for future rewards.
            episodes (int): Number of evaluation episodes to run.
            max_steps (int): Maximum steps per episode.
        Returns:
            traces (List[List[Tuple[int, int]]]): List of state traces for each episode
            ep_returns (list): List of returns per episode.
            ep_losses (list): List of average TD losses per episode.
        """
        model.eval()
        traces = []
        total_return = 0.0
        ep_returns, ep_losses = [], []

        rows = max(1, getattr(env, "_rows", 1) - 1)
        cols = max(1, getattr(env, "_cols", 1) - 1)

        for _ in range(episodes):
            s_tuple = env.reset()
            trace = [s_tuple]
            ep_ret = 0.0
            step_losses = []

            for _ in range(max_steps):
                r, c = s_tuple
                s = torch.tensor([r / rows, c / cols], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(s)
                    a = int(torch.argmax(q_values, dim=1).item())

                s_next, reward, done = env.step(a)
                trace.append(s_next)
                ep_ret += float(reward)

                # Compute TD target and loss using the same loss function as training
                r_next, c_next = s_next
                s_next_t = torch.tensor([r_next / rows, c_next / cols], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    next_q = model(s_next_t)
                    target = reward + (0.0 if done else discount_factor * torch.max(next_q).item())

                target_t = torch.tensor([[target]], dtype=torch.float32, device=device)
                pred_t = q_values[0, a].unsqueeze(0).unsqueeze(0)
                loss_val = loss_fn(pred_t, target_t).item()
                step_losses.append(loss_val)

                s_tuple = s_next
                if done:
                    break

            traces.append(trace)
            total_return += ep_ret
            ep_returns.append(ep_ret)
            ep_losses.append(np.mean(step_losses) if step_losses else 0.0)


        return traces, ep_returns, ep_losses

    def render_trace_grid(
        self,
        env,
        trace: List[Tuple[int, int]]
    ) -> List[List[str]]:
        """
        Renders the given trace on the GridWorld environment as a list of strings.
        Args:
            env: The GridWorld environment.
            trace (List[Tuple[int, int]]): The list of states in the trace.
        Returns:
            List[List[str]]: A 2D list of strings representing the grid with the trace annotated.
        """
        grid = getattr(env, "grid", None)
        if grid is None:
            raise ValueError("Environment must expose a 'grid' ndarray.")

        rows, cols = grid.shape
        out = [[".."] * cols for _ in range(rows)]

        # Obstacles as ##
        for r in range(rows):
            for c in range(cols):
                if np.isnan(grid[r, c]):
                    out[r][c] = "##"
                elif grid[r, c] == 1.0:
                    out[r][c] = "T+"
                elif grid[r, c] < 0.0 and not np.isnan(grid[r, c]):
                    out[r][c] = "T-"

        # Annotate steps
        for idx, (r, c) in enumerate(trace):
            if np.isnan(grid[r, c]):
                continue
            tag = f"{idx+1:02d}"
            out[r][c] = tag

        return out

    def print_trace_grid(self, annotated_grid: List[List[str]]) -> None:
        """
        Prints the annotated grid returned by render_trace_grid.
        Args:
            annotated_grid (List[List[str]]): The annotated grid to print.
        """
        for row in annotated_grid:
            print(" | ".join(row))


    def print_policy_arrows(self, model: torch.nn.Module, env, device: torch.device) -> None:
        """
        Prints a grid showing the greedy action (argmax_a Q(s,a)) at every
        non-obstacle, non-terminal cell. Obstacles = '##', terminals = 'T+'/'T-'.
        Actions: 0=^, 1=>, 2=v, 3=<.
        Args:
            model (torch.nn.Module): The trained DQN model.
            env: The GridWorld environment.
            device (torch.device): The device to run computations on.
        """
        model.eval()
        grid = getattr(env, "grid", None)
        if grid is None:
            raise ValueError("Environment must expose a 'grid' ndarray.")

        rows, cols = grid.shape
        denom_r = max(1, rows - 1)
        denom_c = max(1, cols - 1)
        action_to_char = {0: "v", 1: ">", 2: "^", 3: "<"}


        # build printable grid
        for r in range(rows):
            row_strs = []
            for c in range(cols):
                val = grid[r, c]
                if np.isnan(val):
                    row_strs.append("##")
                elif val == 1.0:
                    row_strs.append("T+")
                elif val < 0.0 and not np.isnan(val):
                    row_strs.append("T-")
                else:
                    s = torch.tensor([r / denom_r, c / denom_c], dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        q = model(s)
                        a = int(torch.argmax(q, dim=1).item())
                    row_strs.append(action_to_char.get(a, ".."))
            print(" | ".join(row_strs))