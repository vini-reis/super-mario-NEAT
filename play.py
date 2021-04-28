import retro
import retro.data
import os
import neat
import pickle

from rominfo import *
from utils import *

# Constants
RADIUS = 6
SAVING_FOLDER = "savings"

#Global variables
best_genome_file = "best-genome-"
winner_file = "winner.pkl"
world_folder = "Yoshi"
env = None

def run(config, file):
    global env

    genome = pickle.load(open(file,'rb'))                                      # Load the best Mario (genome)
    net = neat.nn.FeedForwardNetwork.create(genome, config)                    # Create a feedfoward network
    env.reset()                                                                # Reset the game

    try:
        reward = 0                                                             # Initialize the reward
        done = stuck = False
        timeout = maxX = 0

        while not done and not stuck:
            inputs, x, y = getInputs(getRam(env),RADIUS)                       # Get a 13x13 matrix with the actual state of the game
            output = net.activate(inputs)                                      # Activate the network from the state acquired
            sn, rew, done, info = performAction(getAction(output),env)         # Perform the action with the highest score on the output
            reward += rew                                                      # Increment the reward

            done = getRam(env)[0x13D6] <= 0                                    # Check for Mario's death or finishing the level
            stuck = (timeout % 50 == 0 and maxX > x) or info['lives'] < 4        # Check if Mario's stuck

            maxX = x if x > maxX else maxX                                     # Save rightmost distance

            env.render()                                                       # Render game screen
            timeout+=1

        env.close()                                                            # Close the game
        if stuck: print("Mario needs to train more...")
        print("Distance: ",x," pixels")                                        # Prints the final results
        print("Reward: ", reward)
        input("Press anything to close...")                                    # Interrupt for checking the results
        exit()                                                                 # Exits the program

    except KeyboardInterrupt:
        env.close()                                                            # Close the game
        exit()                                                                 # Exits the program

def main(level, winner=0):
    global env, best_genome_file, world_folder

    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland{0}'.format(level), players=1)      # Start the game in selected level

    localDir = os.path.dirname(__file__)                                                                # Get local dir
    world_folder += str(level)
    saving_path = os.path.join(localDir, SAVING_FOLDER, world_folder)

    if winner > 0 and os.path.isfile(os.path.join(saving_path,best_genome_file) + f"{level}.pkl"):
        best_genome_file = os.path.join(saving_path,best_genome_file) + f"{level}.pkl"                  # Create filename according to selected level
    else:
        best_genome_file = os.path.join(saving_path, winner_file)


    configPath = os.path.join(localDir, "config")                                                       # Get config filepat

    config = neat.config.Config(neat.DefaultGenome,                                                     # Instantiate a config object
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                configPath)

    run(config, best_genome_file)                                                                       # Run the best Mario for now
