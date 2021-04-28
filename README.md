# Super Mario World with Python-NEAT

## Introduction

This project was developed in Artificial Inteligence course of 'Universidade Federal do ABC', from teacher FFabrício Olivetti de França, in the first quarter of 2021.

## Instructions

### Installing libraries

- Install the required libraries with the following command:

```bash
python -m pip install -r requirements.txt
```

- The game ROM needs to be in the *site-packages/retro/data/stable/SuperMarioWorldSnes/* directory, and named as *rom.sfc* (if you're using Anaconda, it should be in *~/anaconda3/lib/python3.6/*)

### Training

To train the the agent, you can use the following command:

```bash
python main.py train [--gen generations] [--level level] [--norender] [--restart] [--checkpoint checkpoint]
```

#### Training args

- ```--gen generations```
Choose the number of generations for training. *Default: 100 generations.*

- ```--level level```
Choose the level which Mario will train or play. *Default: Yoshiland2*

- ```--norender```
Select this option to disable game rendering during training.

- ```--restart```
Select this option to restart Mario training from beggining.

- ```--checkpoint```
Select this option to load a checkpoint from a specific generation. *Default: most recent checkpoint.*

**Obs.:** The checkpoint is taken every 5 generations, so, only multiples of five numbers generations can be loaded.

### Playing with best agent

Use the following command to play with the best agent until now:

```bash
python main.py play [--level level] [--winner gen_number]
```

#### Playing args

- ```--level level```
Choose the level which Mario will play. *Default: Yoshiland 2*

- ```--winner gen_number```
Choose the best agent from the generation selected. *Default: Winner agent*

## Help

To get help while running the program, type:

```bash
python main.py [-h | --help]
```

## References

- [NEAT-Python](https://neat-python.readthedocs.io/en/latest/), Python library used to implement neural networks
- [Gym-Retro](https://retro.readthedocs.io/en/latest/index.html), Python library to run SNES games
- [ProgressBar2](https://pypi.org/project/progressbar2/), library to implement progress bars
- [Super Mario Neat](https://github.com/vivek3141/super-mario-neat), algorithm that uses NEAT to play Super Mario World
- [Super Mario Neat](https://github.com/gustavohb/super-mario-neat), inteligent agent that plays Super Mario World using RLE (Retro Learning Environment), bu @gustavogh
- [Argparse](https://docs.python.org/3/howto/argparse.html) tutorial
- [Flappybird AI](https://github.com/techwithtim/NEAT-Flappy-Bird), AI implemented to evolve a neural network to play Flappy Bird, by youtuber [Tech with Tim](https://www.youtube.com/channel/UC4JX40jDee_tINbkjycV4Sg)
