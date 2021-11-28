from datetime import datetime
import torch
import torch.nn as nn
from torch import optim
import gym
from IPython import display
from torch.nn import functional as F
import collections
import typing
import random
import numpy as np
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import csv

Experience = collections.namedtuple('Experience', field_names=[
                                    'state', 'action', 'reward', 'next_state', 'done'])


class NeuralNetwork(nn.Module):
    def __init__(self, inputs, hidden_layer_1_units, hidden_layer_2_units, outputs):
        super().__init__()
        self.layer1 = nn.Linear(inputs, hidden_layer_1_units)
        self.layer2 = nn.Linear(hidden_layer_1_units, hidden_layer_2_units)
        self.layer3 = nn.Linear(hidden_layer_2_units, outputs)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ExperienceReplayBuffer:
    def __init__(self,
                 sample_size: int,
                 buffer_size: int = None) -> None:

        self._sample_size = sample_size
        self._buffer_size = buffer_size
        self._buffer = collections.deque(maxlen=buffer_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def append(self, state, action, reward, next_state, done) -> None:
        self._buffer.append(Experience(
            state, action, reward, next_state, done))

    def sample(self) -> typing.List[Experience]:
        return random.sample(self._buffer, k=self._sample_size)

    @property
    def has_enough_experience(self):
        return self.__len__() > self._sample_size


class Agent:
    # Hyperparameters
    ALPHA = 0.3
    GAMMA = 0.99
    MIN_EPSILON = 0.1
    MAX_EPSILON = 0.7
    EPSILON_DECAY_FACTOR = 0.001
    EPSILON_DECAY_ALGORITHM = 'LINEAR'
    # Replay Buffer
    EXPERIENCES_SAMPLE_SIZE = 64
    EXPERIENCE_REPLAY_BUFFER_SIZE = 100000
    # Neural Network
    UPDATE_NEURAL_NETWORK_FREQUENCY = 4
    NEURAL_NETWORK_LAYER_1_HIDDEN_UNITS = 64
    NEURAL_NETWORK_LAYER_2_HIDDEN_UNITS = 64
    # Optimizer
    OPTIMIZER_FUNCTION = optim.RMSprop
    OPTIMIZER_PARAMETERS = {
        "lr": 5e-4,
        "alpha": 0.99,
        "eps": 1e-08,
        "weight_decay": 0,
        "momentum": 0,
        "centered": False
    }

    def print_hyperparameters(self):
        print('Agent Hyperparameters:')
        print(f'\tAlpha:{self.ALPHA}')
        print(f'\tGamma:{self.GAMMA}')
        print(f'\tEpsilon:')
        print(f'\t\tDecay strategy: {self.EPSILON_DECAY_ALGORITHM}')
        print(f'\t\tDecay factor: {self.EPSILON_DECAY_FACTOR}')
        print(f'\t\tMax epsilon: {self.MAX_EPSILON}')
        print(f'\t\tMin epsilon: {self.MIN_EPSILON}')
        nn_state = self._local_neural_network.state_dict()
        print('\tNeural Network:')
        print(
            f'\t\tHidden Layer unites: {len(nn_state["layer1.weight"])}/{len(nn_state["layer2.weight"])}')
        print(f'\t\tUpdate frequency: {self.UPDATE_NEURAL_NETWORK_FREQUENCY}')

    @classmethod
    def set_hp_alpha(cls, alpha):
        cls.ALPHA = alpha
        return cls

    @classmethod
    def set_hp_gamma(cls, gamma):
        cls.GAMMA = gamma
        return cls

    @classmethod
    def set_hp_epsilon(cls, upper_bound, lower_bound, decay_factor, decay_algorithm):
        cls.MAX_EPSILON = upper_bound
        cls.MIN_EPSILON = lower_bound
        cls.EPSILON_DECAY_FACTOR = decay_factor
        cls.EPSILON_DECAY_ALGORITHM = decay_algorithm
        return cls

    @classmethod
    def set_hp_optimizer(cls, function, parameters):
        cls.OPTIMIZER_FUNCTION = function,
        cls.OPTIMIZER_PARAMETERS = parameters
        return cls

    @classmethod
    def set_hp_replay_buffer(cls, sample_size: int, buffer_size: int):
        cls.EXPERIENCES_SAMPLE_SIZE = sample_size,
        cls.EXPERIENCE_REPLAY_BUFFER_SIZE = buffer_size
        return cls

    @classmethod
    def set_hp_neural_network(cls, hidden_layer_1_units: int, hidden_layer_2_units: int, update_frequency: int):
        cls.UPDATE_NEURAL_NETWORK_FREQUENCY = update_frequency
        cls.NEURAL_NETWORK_LAYER_1_HIDDEN_UNITS = hidden_layer_1_units
        cls.NEURAL_NETWORK_LAYER_2_HIDDEN_UNITS = hidden_layer_2_units
        return cls

    def __init__(self, name: str, environment: str, seed: int = 42):
        self._name = name
        self._env = gym.make(environment)
        self._seed = seed
        self._setup_random_generators()
        self._state_size = self._env.observation_space.shape[0]
        self._action_size = self._env.action_space.n
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self._memory = ExperienceReplayBuffer(
            self.EXPERIENCES_SAMPLE_SIZE, self.EXPERIENCE_REPLAY_BUFFER_SIZE)

        self._local_neural_network = NeuralNetwork(
            self._state_size, self.NEURAL_NETWORK_LAYER_1_HIDDEN_UNITS, self.NEURAL_NETWORK_LAYER_2_HIDDEN_UNITS, self._action_size)
        self._target_neural_network = NeuralNetwork(
            self._state_size, self.NEURAL_NETWORK_LAYER_1_HIDDEN_UNITS, self.NEURAL_NETWORK_LAYER_2_HIDDEN_UNITS, self._action_size)
        self._local_neural_network.to(self._device)
        self._target_neural_network.to(self._device)
        self._training_scores = None
        self._training_time = None

        self._optimizer = self.OPTIMIZER_FUNCTION(
            self._local_neural_network.parameters(), **self.OPTIMIZER_PARAMETERS)
        self.print_hyperparameters()

    def _setup_random_generators(self):
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self._env.seed(self._seed)

    def _decay_epsilon(self, episode: int) -> float:
        if self.EPSILON_DECAY_ALGORITHM == 'POWER':
            return max(self.EPSILON_DECAY_FACTOR ** episode * self.MAX_EPSILON, self.MIN_EPSILON)
        else:
            return max((1 - self.EPSILON_DECAY_FACTOR * episode) * self.MAX_EPSILON, self.MIN_EPSILON)

    def learn(self, experiences: typing.List[Experience]):
        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(self._device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(self._device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(self._device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(self._device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self._device)

        Q_targets_next = self._target_neural_network(
            next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))

        Q_expected = self._local_neural_network(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._update_target_neural_network_parameters()

    def _update_target_neural_network_parameters(self) -> None:
        for target_parameter, parameter in zip(self._target_neural_network.parameters(), self._local_neural_network.parameters()):
            target_parameter.data.copy_(
                self.ALPHA * parameter.data + (1 - self.ALPHA) * target_parameter.data)

    def _train_local_neural_network(self, state: np.array):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._local_neural_network.eval()
        with torch.no_grad():
            action_values = self._local_neural_network(state)
        self._local_neural_network.train()
        return action_values

    def choose_action(self, state: np.array, epsilon: float) -> int:
        action_values = self._train_local_neural_network(state)
        if random.uniform(0, 1) < epsilon or not self._memory.has_enough_experience:
            return np.random.randint(self._action_size)
        else:
            return action_values.cpu().argmax().item()

    def _is_time_to_learn(self, time_step):
        return time_step % self.UPDATE_NEURAL_NETWORK_FREQUENCY == 0

    def save_checkpoint(self, scores):
        checkpoint = {
            'environment': self._env.unwrapped.spec.id,
            'seed': self._seed,
            'memory': {
                'experience_replay_buffer': self._memory,
                'buffer_size': self.EXPERIENCE_REPLAY_BUFFER_SIZE,
                'sample_size': self.EXPERIENCES_SAMPLE_SIZE
            },
            'training': {
                'scores': self._training_scores,
                'time': self._training_time
            },
            'alpha': self.ALPHA,
            'gamma': self.GAMMA,
            'epsilon': {
                'lower_bound': self.MIN_EPSILON,
                'upper_bound': self.MAX_EPSILON,
                'decay_factor': self.EPSILON_DECAY_FACTOR,
                'decay_algorithm': self.EPSILON_DECAY_ALGORITHM
            },
            'neural_networks': {
                'update_frequency': self.UPDATE_NEURAL_NETWORK_FREQUENCY,
                'local_neural_network_state': self._local_neural_network.state_dict(),
                'target_neural_network_state': self._target_neural_network.state_dict(),
                'optimizer_state': self._optimizer.state_dict()
            }
        }
        torch.save(checkpoint, f'checkpoints/{self._name}.checkpoint')

    @classmethod
    def from_checkpoint(cls, name):
        checkpoint = torch.load(f'checkpoints/{name}.checkpoint')
        cls = cls.set_hp_alpha(checkpoint['alpha']).set_hp_gamma(
            checkpoint['gamma'])
        hidden_layer_1_units = len(
            checkpoint['neural_networks']['local_neural_network_state']['layer1.weight'])
        hidden_layer_2_units = len(
            checkpoint['neural_networks']['local_neural_network_state']['layer2.weight'])
        cls.set_hp_neural_network(hidden_layer_1_units, hidden_layer_2_units,
                                  checkpoint['neural_networks']['update_frequency'])
        # cls.set_hp_alpha(checkpoint['alpha']).set_hp_gamma(
        #     checkpoint['gamma']).set_hp_epsilon(**checkpoint['epsilon']).set_hp_replay_buffer(checkpoint['memory']['sample_size'], checkpoint['memory']['buffer_size'])

        agent = cls(name, checkpoint['environment'], checkpoint['seed'])
        agent._local_neural_network.load_state_dict(
            checkpoint['neural_networks']['local_neural_network_state'])
        agent._target_neural_network.load_state_dict(
            checkpoint['neural_networks']['target_neural_network_state'])
        agent._optimizer.load_state_dict(
            checkpoint['neural_networks']['optimizer_state'])
        agent._memory = checkpoint['memory']['experience_replay_buffer']
        agent._training_scores = checkpoint['training']['scores']
        agent._training_time = checkpoint['training']['time']
        return agent

    def train(self,
              target_score: float,
              total_episodes: int,
              recent_scores_size: int) -> typing.List[float]:
        scores = []
        training_started_at = datetime.now()
        recent_scores = collections.deque(maxlen=recent_scores_size)
        epsilon = self.MAX_EPSILON
        # Loop for each episode
        for episode in range(total_episodes):
            # Initialize state
            state = self._env.reset()
            score = 0
            done = False
            # Loop for each time step of the episode
            time_step = 0
            while not done:
                time_step += 1
                # Choose action from state using policy derived from Q
                action = self.choose_action(state, epsilon)
                # Take action and observe reward and next state
                next_state, reward, done, _ = self._env.step(action)
                # Save observation in memory
                self._memory.append(
                    state, action, reward, next_state, done)
                # Learn from the observation
                if self._is_time_to_learn(time_step) and self._memory.has_enough_experience:
                    experiences = self._memory.sample()
                    self.learn(experiences)
                state = next_state
                score += reward
            # Decay epsilon
            epsilon = self._decay_epsilon(episode)
            scores.append(score)
            recent_scores.append(score)

            average_score = np.mean(recent_scores)
            print(
                f'\rEpisode {episode + 1}\tAverage Score: {average_score:.2f} (e: {epsilon:.2f})', end="")
            if target_score is not None and average_score >= target_score:
                print(
                    f"\nTraining completed in {episode:d} episodes!\tAverage Score: {average_score:.2f}")
                break
            if (episode + 1) % 100 == 0:
                print(
                    f"\rEpisode {episode + 1}\tAverage Score: {average_score:.2f}")
        training_ended_at = datetime.now()
        self._training_time = (training_ended_at -
                               training_started_at).total_seconds()
        self._training_scores = scores
        self.save_checkpoint(scores)
        return scores

    def play(self, episodes: int = 1):
        scores = []
        for _ in range(episodes):
            state = self._env.reset()
            done = False
            score = 0
            while not done:
                action = self.choose_action(state, 0)
                state, reward, done, _ = self._env.step(action)
                score += reward
            scores.append(score)
        return scores


def simulate(agent: Agent, env: gym.Env, ax: plt.Axes) -> None:
    state = env.reset()
    img = ax.imshow(env.render(mode='rgb_array'))
    done = False
    while not done:
        action = agent.choose_action(state, 0)
        img.set_data(env.render(mode='rgb_array'))
        plt.axis('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)
        state, _, done, _ = env.step(action)
    env.close()


def plot_scores(scores):
    episodes = [episode + 1 for episode in range(len(scores))]
    plt.plot(episodes, scores, label='Rewards')
    plt.axhline(200, color='k', linestyle="dashed", label="Target Rewards")
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.legend(loc='best')
    plt.show()


def explore_alphas(name, gamma):
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    for alpha in alphas:
        name = f'{name}_explore_alpha_{str(alpha).replace(".", "-")}'
        agent = Agent.set_hp_alpha(alpha).set_hp_gamma(
            gamma)(name, 'LunarLander-v2')
        agent.train(None, 2000, 100)


def explore_gammas(name, alpha):
    gammas = [0.995]
    for gamma in gammas:
        filename = f'{name}_explore_alpha_{str(alpha).replace(".", "-")}_gamma_{str(gamma).replace(".", "-")}'
        agent = Agent.set_hp_alpha(alpha).set_hp_gamma(
            gamma)(filename, 'LunarLander-v2')
        agent.train(None, 2000, 100)


def plot_eploration_graphs(label, values, scores, training_episodes, training_times):
    plt.plot(values, scores, label='Rewards')
    plt.ylabel('Rewards')
    plt.xlabel(label)
    plt.legend(loc='best')
    plt.show()
    plt.plot(values, training_episodes, label='Training Episodes')
    plt.ylabel('Training Episodes')
    plt.xlabel(label)
    plt.legend(loc='best')
    plt.show()
    plt.plot(values, training_times, label='Training Time')
    plt.ylabel('Training time')
    plt.xlabel(label)
    plt.legend(loc='best')
    plt.show()


def alpha_gamma_search(name):
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    gammas = [0.98, 0.99, 0.995]
    for alpha in alphas:
        for gamma in gammas:
            filename = f'{name}_explore_alpha_{str(alpha).replace(".", "-")}_gamma_{str(gamma).replace(".", "-")}'
            agent = Agent.set_hp_alpha(alpha).set_hp_gamma(
                gamma)(filename, 'LunarLander-v2')
            agent.train(None, 2000, 100)


def alpha_gamma_analyze(name):
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    gammas = [0.98, 0.99, 0.995]
    rows = [['alpha | gamma'] + gammas]
    for alpha in alphas:
        row = [alpha]
        for gamma in gammas:
            filename = f'{name}_explore_alpha_{str(alpha).replace(".", "-")}_gamma_{str(gamma).replace(".", "-")}'
            agent = Agent.from_checkpoint(filename)
            _total_scores = []
            for _ in range(10):
                _scores = agent.play(100)
                _total_scores.append(np.mean(_scores))
            row.append(np.mean(_total_scores))
        rows.append(row)
    with open(f'checkpoints/{name}_alpha_gamma.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def hidden_layer_search(name):
    pairs = [(64, 64), (128, 128), (64, 32), (128, 64), (256, 128)]
    for pair in pairs:
        filename = f'{name}_explore_hidden_layer_{pair[0]}-{pair[1]}'
        agent = Agent.set_hp_alpha(0.05).set_hp_gamma(
            0.995).set_hp_neural_network(pair[0], pair[1], 4)(filename, 'LunarLander-v2')
        agent.train(None, 2000, 100)


def hidden_layer_analyze(name):
    pairs = [(64, 64), (128, 128), (64, 32), (128, 64), (256, 128)]
    rows = [['units', 'rewards']]
    for pair in pairs:
        filename = f'{name}_explore_hidden_layer_{pair[0]}-{pair[1]}'
        agent = Agent.from_checkpoint(filename)
        _total_scores = []
        for i in range(10):
            print(f'Simulation {i + 1}/10')
            _scores = agent.play(100)
            _total_scores.append(np.mean(_scores))
        rows.append([f'{pair[0]}/{pair[1]}', np.mean(_total_scores)])
    with open(f'checkpoints/{name}_hidden_layer.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def update_frequency_search(name):
    frequencies = [4, 8, 16, 32, 64]
    for frequency in frequencies:
        filename = f'{name}_explore_update_frequency_{frequency}'
        agent = Agent.set_hp_alpha(0.05).set_hp_gamma(
            0.995).set_hp_neural_network(64, 64, frequency)(filename, 'LunarLander-v2')
        agent.train(None, 2000, 100)


def update_frequency_analyze(name):
    frequencies = [4, 8, 16, 32, 64]
    rows = [['frequency', 'rewards']]
    for frequency in frequencies:
        filename = f'{name}_explore_update_frequency_{frequency}'
        agent = Agent.from_checkpoint(filename)
        _total_scores = []
        for _ in range(10):
            _scores = agent.play(100)
            _total_scores.append(np.mean(_scores))
        rows.append([frequency, np.mean(_total_scores)])
    with open(f'checkpoints/{name}_update_frequency.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Land the spaceship!')
    parser.add_argument('--action', type=str, choices=[
                        'train', 'analyze', 'alpha-gamma-search', 'alpha-gamma-analyze', 'hidden-layer-search', 'hidden-layer-analyze', 'update-frequency-search', 'update-frequency-analyze', 'explore-alphas', 'explore-gammas', 'analyze-explored-alphas', 'analyze-explored-gammas'], help='Select an action to execute')
    parser.add_argument(
        '--name', type=str, help='The filepath prefix in order to store/load the agent')
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--epsilon-decay-strategy',
                        type=str, choices=['POWER', 'LINEAR'])
    parser.add_argument('--epsilon-decay-factor',
                        type=float)
    parser.add_argument('--stop-at-score', type=float)
    args = parser.parse_args()

    if args.action == 'train':
        AgentClass = Agent
        if args.alpha is not None:
            AgentClass = AgentClass.set_hp_alpha(args.alpha)
        if args.gamma is not None:
            AgentClass = AgentClass.set_hp_gamma(args.gamma)
        if args.epsilon_decay_strategy is not None and args.epsilon_decay_factor is not None:
            AgentClass = AgentClass.set_hp_epsilon(
                1, 0.1, args.epsilon_decay_factor, args.epsilon_decay_strategy)
        agent = AgentClass(args.name, 'LunarLander-v2')
        scores = agent.train(args.stop_at_score, 2000, 100)
    elif args.action == 'analyze':
        agent = Agent.from_checkpoint(args.name)
        plot_scores(agent._training_scores)
        plot_scores(agent.play(100))
    elif args.action == 'alpha-gamma-analyze':
        alpha_gamma_analyze(args.name)
    elif args.action == 'alpha-gamma-search':
        alpha_gamma_search(args.name)
    elif args.action == 'hidden-layer-search':
        hidden_layer_search(args.name)
    elif args.action == 'hidden-layer-analyze':
        hidden_layer_analyze(args.name)
    elif args.action == 'update-frequency-search':
        update_frequency_search(args.name)
    elif args.action == 'update-frequency-analyze':
        update_frequency_analyze(args.name)
    elif args.action == 'explore-alphas':
        explore_alphas(args.name, args.gamma)
    elif args.action == 'explore-gammas':
        explore_gammas(args.name, args.alpha)
    elif args.action == 'analyze-explored-alphas':
        analyze_alpha_exploration(args.name)
    elif args.action == 'analyze-explored-gammas':
        analyze_gamma_exploration(args.name)


if __name__ == "__main__":
    main()

# agent = Agent('LunarLander-v2')

# _, ax = plt.subplots(1, 1, figsize=(10, 8))
# # simulate(deep_q_agent, env, ax)


# scores = agent.train(-10, 2000, 100, 'trained_model')

# print(difference)
# simulate(agent, agent._env, ax)