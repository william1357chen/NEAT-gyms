import multiprocessing
import neat 
import gym
import numpy as np
import os
import sys
import pickle

runs_per_net = 2

class PoleBalance:
    @staticmethod
    # Evaluate each genome/neural network and calculate fitness by running OpenAI environment
    def eval_genome(genome, config):
        # Create our neural network 
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # Create OpenAI environment
        env = gym.make('CartPole-v1')
        # We want to have several runs for each genome and average out the performance
        fitnesses = []
        for run in range(runs_per_net):
            # Initialize the environment
            observation = env.reset()
            # Fitness for this run
            fitness = 0.0
            # Loop until we reach termination
            done = False
            while not done:
                # Pass in observation as input for neural network and get output/action
                action = np.argmax(net.activate(observation)) # argmax because we have two output nodes, left or right
                # Feed action into the environment, and get results like new observation, reward, etc
                observation, reward, done, info = env.step(action) # take action given by neural network
                fitness += reward
            fitnesses.append(fitness)
        env.close()

        # The genome's fitness is its average across all runs.
        return np.mean(fitnesses)
    @staticmethod
    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = PoleBalance.eval_genome(genome, config)
    @staticmethod
    def run(config_file):
        # Load the config file, which is assumed to live in
        # the same directory as this script.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)

        pop = neat.Population(config)
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.StdOutReporter(True))
        # Redirect stdout
        actual_stdout = sys.stdout
        sys.stdout = open('pole-balance-output.txt', 'w') 

        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), PoleBalance.eval_genome)
        winner = pop.run(pe.evaluate)

        # Restore stdout 
        sys.stdout = actual_stdout

        return winner, config
