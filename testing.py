import gym
import pickle
import neat
import numpy as np
import os
import sys


def test_mountain_climbing():
    # load the winner
    with open('winner-mountain-car', 'rb') as f:
        winner = pickle.load(f)


    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.abspath('')
    config_path = os.path.join(local_dir, 'config-mountain-car')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    env = gym.make('MountainCarContinuous-v0')
    observation = env.reset()
    fitness = 0.0
    done = False
    while not done:
        action = net.activate(observation)
        observation, reward, done, info = env.step(action)
        env.render()
        fitness += reward
    env.close()
    print('The score/fitness for this random sample run is', fitness) # Frames/Steps this run last

def test_pole_balancing():
    # load the winner
    with open('winner-pole-balance', 'rb') as f:
        winner = pickle.load(f)


    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.abspath('')
    config_path = os.path.join(local_dir, 'config-pole-balance')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    env = gym.make('CartPole-v1')
    observation = env.reset()
    fitness = 0.0
    done = False
    while not done:
        action = np.argmax(net.activate(observation)) 
        observation, reward, done, info = env.step(action)
        env.render()
        fitness += reward
    env.close()
    print('The score/fitness for this random sample run is', fitness) # Frames/Steps this run last


if __name__ == '__main__':
    test_pole_balancing()
    test_mountain_climbing()