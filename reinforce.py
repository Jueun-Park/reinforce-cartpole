import numpy as np
import torch
from torch import nn
from gym import Env


class MlpPolicy(nn.Module):
    def __init__(self, obs_size, action_size) -> None:
        super().__init__()
        self.linear1 = nn.Linear(obs_size, 32)
        self.linear2 = nn.Linear(32, action_size)

    def forward(self, obs):
        latent = self.linear1(obs)
        latent = torch.relu(latent)
        action_prob = self.linear2(latent)
        action_prob = torch.relu(action_prob)
        action_prob = torch.softmax(action_prob, dim=0)
        return action_prob


class Reinforce:
    def __init__(self, env: Env, lr=1e-3, verbose=1) -> None:
        self.env = env
        self.verbose = verbose
        self.policy = MlpPolicy(
            obs_size=self.env.observation_space.shape[0],  # Box
            action_size=self.env.action_space.n,  # Discrete
        )
        self.policy.double()
        self.optimizer = torch.optim.Adam(
            params=self.policy.parameters(), lr=lr,
        )
        self.gamma = 0.99

    def predict(self, obs):
        prob_tensor = self.policy(torch.tensor(obs).double())
        prob = prob_tensor.detach().numpy()
        action = np.random.choice(range(self.env.action_space.n), p=prob)
        return action

    def learn(self, num_episodes):
        episode_reward_list = []
        for episode in range(num_episodes):
            episode_reward = 0
            reward_trajectory = []
            prob_trajectory = []
            done = False
            obs = self.env.reset()
            while not done:
                prob_tensor = self.policy(torch.tensor(obs).double())
                prob = prob_tensor.detach().numpy()
                action = np.random.choice(range(self.env.action_space.n), p=prob)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward

                reward_trajectory.append(reward)
                action_probability = prob_tensor[action]
                prob_trajectory.append(action_probability)

            self._train(reward_trajectory, prob_trajectory)
            episode_reward_list.append(episode_reward)
            if len(episode_reward_list) > 100:
                del episode_reward_list[0]
            if episode % 100 == 0 and self.verbose > 0:
                avg_reward = np.mean(episode_reward_list)
                print(
                    f"episode {episode:6d} | reward {episode_reward:6.1f} | avg reward {avg_reward:6.2f}")

    def __discount_rewards(self, rwd_traj):
        discounted_rewards = np.zeros(len(rwd_traj))
        running_add = 0
        for t in reversed(range(0, len(rwd_traj))):
            running_add = running_add * self.gamma + rwd_traj[t]
            discounted_rewards[t] = running_add
        return discounted_rewards


    def _train(self, rwd_traj, prob_traj):
        self.optimizer.zero_grad()
        loss = 0
        discounted_rwd = self.__discount_rewards(rwd_traj)
        for rr, pp in zip(discounted_rwd, prob_traj):
            loss += -rr * torch.log(pp)
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    torch.random.manual_seed(2021)
    import gym
    env = gym.make("CartPole-v1")
    agent = Reinforce(env)
    agent.learn(2000)
