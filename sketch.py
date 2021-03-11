import gym

env = gym.make("CartPole-v1")
print(env.observation_space)
print(env.observation_space.shape)
print(env.action_space)
print(env.action_space.n)

for _ in range(10):
    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
    print(obs, action)
