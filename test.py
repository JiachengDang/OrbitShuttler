import gym
import random


if __name__ == "__main__":
    env= gym.make('SpaceInvaders-v0')
    print(env.observation_space.shape)
    print(env.action_space.n)