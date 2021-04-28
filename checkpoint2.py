"""Uses `pickle` to save and restore populations (and other aspects of the simulation state)."""
from __future__ import print_function

import gzip
import random
import time
import neat
import pickle

from neat.population import Population
from neat.reporting import BaseReporter


class Checkpointer(BaseReporter):
    """
    A reporter class that performs checkpointing using `pickle`
    to save and restore populations (and other aspects of the simulation state).
    """
    def __init__(self, generation_interval=1, time_interval_seconds=300,
                 filename_prefix='checkpoint'):
        """
        Saves the current state (at the end of a generation) every ``generation_interval`` generations or
        ``time_interval_seconds``, whichever happens first.

        :param generation_interval: If not None, maximum number of generations between save intervals
        :type generation_interval: int or None
        :param time_interval_seconds: If not None, maximum number of seconds between checkpoint attempts
        :type time_interval_seconds: float or None
        :param str filename_prefix: Prefix for the filename (the end will be the generation number)
        """
        self.generation_interval = generation_interval
        self.time_interval_seconds = time_interval_seconds
        self.filename_prefix = filename_prefix

        self.current_generation = None
        self.last_generation_checkpoint = -1
        self.last_time_checkpoint = time.time()

        self.config = None
        self.pop = None
        self.species = None
        self.current_generation = None

    def start_generation(self, generation):
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
        self.config = config
        self.pop = population
        self.species = species_set

    def save_checkpoint(self, population, stats, filename):
        """ Save the current simulation state in a determined filename."""
        print("Checkpointing...")
        with gzip.open(filename, 'w', compresslevel=5) as f:
            data = (self.current_generation, population.config, population.population, population.species, random.getstate(), stats)
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore_checkpoint(filename):
        """Resumes the simulation from a previous saved point."""
        with gzip.open(filename) as f:
            generation, config, population, species_set, rndstate, stats = pickle.load(f)
            random.setstate(rndstate)
            return neat.Population(config, (population, species_set, generation)), stats
