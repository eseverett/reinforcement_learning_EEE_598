import copy
import torch
from typing import Tuple

class DQN_Agent:
    
    def __init__(self, rl_cfg: dict, training_cfg: dict) -> None:
        self.learning_rate = rl_cfg['learning_rate']
        self.discount_factor = rl_cfg['discount_factor']
        self.exploration_rate = rl_cfg['exploration_rate']
        
        self.num_episodes = training_cfg['num_episodes']
        self.max_steps_per_episode = training_cfg['max_steps_per_episode']
        self.exploration_decay_rate = training_cfg['exploration_decay_rate']
        self.min_exploration_rate = training_cfg['min_exploration_rate']    
        self.loss_function = training_cfg['loss_function']
        self.optimizer_class = training_cfg['optimizer']
        self.replay_buffer = training_cfg['buffer']
        self.environment = training_cfg['environment']
        self.device = training_cfg['device']


        self.target_update_steps = training_cfg['target_update_steps']
        self.warmup_steps = training_cfg['warmup_steps']
        self.gradient_steps_per_env_step = training_cfg['gradient_steps_per_env_step']

        self.model = training_cfg['model'].to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)

        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)
        
        self.running_loss = []
        self.running_return = []
        self._global_step = 0

    def _normalize_state(self, state_rc: Tuple[int, int]) -> torch.Tensor:
        r, c = state_rc
        rows = max(1, getattr(self.environment, "_rows", 1) - 1)
        cols = max(1, getattr(self.environment, "_cols", 1) - 1)
        return torch.tensor([r / rows, c / cols], dtype=torch.float32, device=self.device)

    def select_action(self, state: torch.Tensor) -> int:
        if torch.rand(1).item() < self.exploration_rate:
            return torch.randint(0, 4, (1,)).item()
        with torch.no_grad():
            q_values = self.model(state.unsqueeze(0))
            return torch.argmax(q_values).item()
            
    def update_exploration_rate(self) -> None:
        self.exploration_rate = max(self.exploration_rate * self.exploration_decay_rate, self.min_exploration_rate)

    def _sync_target_network(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def optimize_model(self) -> None:

        while self.replay_buffer.size < self.warmup_steps:
            s = self._normalize_state(self.environment.reset())
            done = False
            while not done:
                a = torch.randint(0, 4, (1,)).item()
                ns_tuple, r, done = self.environment.step(a)
                ns = self._normalize_state(ns_tuple)
                self.replay_buffer.add(s, a, r, ns, done)
                s = ns
                if self.replay_buffer.size >= self.warmup_steps:
                    break

        state_tuple = self.environment.reset()
        state = self._normalize_state(state_tuple)
        episode_return = 0.0
        episode_loss_sum = 0.0
        episode_loss_count = 0

        for _ in range(self.max_steps_per_episode):
            action = self.select_action(state)
            next_state_tuple, reward, done = self.environment.step(action)
            next_state = self._normalize_state(next_state_tuple)

            self.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += float(reward)
            self._global_step += 1

            if self.replay_buffer.size >= self.replay_buffer.batch_size:
                for _ in range(self.gradient_steps_per_env_step):
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample()

                    # Q(s,a)
                    q_values_all = self.model(states)
                    q_values = q_values_all.gather(1, actions)

                    # y = r + gamma * (1 - done) * max_a' Q_target(s', a')
                    with torch.no_grad():
                        next_q_all = self.target_model(next_states)
                        max_next_q = torch.max(next_q_all, dim=1, keepdim=True).values
                        targets = rewards + self.discount_factor * (1.0 - dones) * max_next_q

                    loss = self.loss_function(q_values, targets)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    episode_loss_sum += loss.item()
                    episode_loss_count += 1

            if self._global_step % self.target_update_steps == 0:
                self._sync_target_network()

            if done:
                break

        if episode_loss_count > 0:
            self.running_loss.append(episode_loss_sum / episode_loss_count)
        else:
            self.running_loss.append(0.0)

        self.running_return.append(episode_return)
        self.update_exploration_rate()

    def get_running_loss(self) -> list:
        return self.running_loss

    def get_running_return(self) -> list:
        return self.running_return
