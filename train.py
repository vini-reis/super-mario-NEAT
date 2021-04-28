import retro
import retro.data
import os
import neat
import pickle
import progressbar
import shutil

from rominfo import *
from utils import *
import checkpoint2 as ck

# Constants
RADIUS = 6
FINAL_DISTANCE = 4780
LASTEST_SUFFIX = "-lastest"
PKL_EXTENSION = ".pkl"
SAVING_FOLDER = "savings"

# Global Variables
pop = None
stats = None
checkpointer = None
env = None
rdr = True
world_folder = "Yoshi"
best_genome_file = "best-genome-"
winner_file = "winner.pkl"
checkpoint_file = "checkpoint-" 
lastest_checkpoint_file = checkpoint_file + "lastest"

def eval_genomes(genomes, config):
    """
    Function used to evaluate each genome in a generation.
    :param genomes: Genomes list
    :param config: Config object
    """

    global env, pop, stats, rdr, best_genome_filepath

    bar = progressbar.ProgressBar(max_value=len(genomes))
    pBarCount = 0

    for genome_id, genome in genomes:
        try:
            env.reset()                                                         # Reset game
            net = neat.nn.FeedForwardNetwork.create(genome, config)             # Create a net with actual genome
            done = False
                                                                                # and the whole screen is mapped
            timeout = 0                                                         # Reset a timeout counter
            maxX = 0                                                            # Reset the maximum distance achieved
            reward = 0                                                          # Reset the level reward

            while not done:
                inputs, x, y = getInputs(getRam(env),RADIUS)                    # Get game inputs - 13x13 array where Mario is in the middle
                output = net.activate(inputs)                                   # Ativate the net to get the score for each action
                sn, rew, done, info = performAction(getAction(output),env)      # Perform the action with highest score
                reward += rew                                                   # Increment level reward with the reward won in the previous action

                done = info['lives'] < 4 or getRam(env)[0x13D6] <= 0            # Game over check
                done = timeout % 40 == 0 and x <= maxX                          # Stuck verification

                if x > maxX: maxX = x                                           # Rightmost x coordinade saving

                if not rdr: env.render()                                        # Render all actions taken in screen
                timeout += 1                                                    # Increment our "timer"

            pBarCount+=1
            bar.update(pBarCount)                                               # Update the progressbar

            genome.fitness = 0.0
            genome.fitness += maxX if abs(maxX - x) > 50 else x                             # Measuring distance and punishing for coming back too much
            genome.fitness -= timeout/4                                                     # Punishing for time wasted
            genome.fitness = (-1.0) if maxX < 500.0 else genome.fitness                     # Punishing for getting stuck at the begining
            genome.fitness = maxX + reward if maxX >= FINAL_DISTANCE else genome.fitness    # Rewarding for finishing the level

            if checkpointer.current_generation > 0:                                         # Check if is the first generation
                                                                                            # because if itś not, thereś no best genome
                if genome.fitness > pop.best_genome.fitness:                                # Check if this genome beats the best until now
                    print("\n\nWinner is beaten, saving...")
                    pickle.dump(pop.best_genome, open(winner_file, 'wb'))                   # If it does, saves it

        except KeyboardInterrupt:
            if not pop.best_genome is None:
                print("\nSaving the best genome until now...")
                pickle.dump(pop.best_genome, open(winner_file, 'wb'))                       # Save the best genome for now

            print("\nSaved! Mario out!!")
            env.close()                                                                     # Close oue game environment
            exit()                                                                          # Exits application

    if (checkpointer.current_generation % 5 == 0 and checkpointer.current_generation > 0):                                  # Checkpoint at the end of each 5 generations
        checkpointer.save_checkpoint(pop, stats, checkpoint_file + str(checkpointer.current_generation))
    checkpointer.save_checkpoint(pop, stats, lastest_checkpoint_file)
    print("\nSaving the best genome until now...")
    pickle.dump(pop.best_genome, open(best_genome_file + str(checkpointer.current_generation) + PKL_EXTENSION, 'wb'))       # Save the best genome of each generation
    pickle.dump(pop.best_genome, open(winner_file, 'wb'))

def main(gen, level, norender=False, restart=False, checkpoint=0):
    global env, checkpointer, stats, pop, rdr, best_genome_file, checkpoint_file, winner_file, lastest_checkpoint_file, world_folder

    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland{0}'.format(level), players=1)   # Start the game
    rdr = norender

    localDir = os.path.dirname(__file__)
    world_folder += str(level)
    saving_path = os.path.join(localDir, SAVING_FOLDER, world_folder)
    best_genome_file = os.path.join(saving_path,best_genome_file)
    winner_file = os.path.join(saving_path, winner_file)
    checkpoint_file = os.path.join(saving_path, checkpoint_file)
    lastest_checkpoint_file = os.path.join(saving_path, lastest_checkpoint_file)
    configPath = os.path.join(localDir, "config")

    config = neat.config.Config(neat.DefaultGenome,                                                          # Instatiate a config object
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                configPath)

    pop = neat.Population(config)                                                                            # Instantiate a population
    stats = neat.StatisticsReporter()                                                                        # Instantiate a stats reporter

    if True in [f.startswith("checkpoint") for f in os.listdir(saving_path)] and not restart:                # Check if a checkpoint exists
        if checkpoint > 0:
            pop, stats = ck.Checkpointer.restore_checkpoint(checkpoint_file + str(checkpoint))               # Loads the population selected from checkpoint
            pop.best_genome = pickle.load(open(best_genome_file + str(checkpoint) + PKL_EXTENSION, 'rb'))    # Loads the best genome from population above
        else:
            pop, stats = ck.Checkpointer.restore_checkpoint(lastest_checkpoint_file)                         # Loads the population from lastest checkpoint
            pop.best_genome = pickle.load(open(winner_file,'rb'))                                            # Loads the best genome for now
    elif restart:
        shutil.rmtree(saving_path)                                                                           # Remove the saving directory
        os.makedirs(saving_path)                                                                             # Create a new one

    pop.add_reporter(neat.StdOutReporter(True))                                                              # Enable report on StdOut
    pop.add_reporter(stats)                                                                                  # Add our stats reporter to the population
    checkpointer = ck.Checkpointer()                                                                         # Instatiate and checkpointes
    pop.add_reporter(checkpointer)                                                                           # Add our checkpointer to the population

    pop.run(eval_genomes,gen)                                                                                # Starts the evolve process
