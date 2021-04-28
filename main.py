import argparse
import train
import play

p = argparse.ArgumentParser(description="This program is an AI designed to play Super Mario World on world YoshiIsland using the NEAT library in Python.")
p.add_argument("mode", metavar="mode", type=str, choices=['train','play'], 
               help="Select 'train' to train the model from the best until now or from scratch. Or, 'play' to play with the best Mario trained. Press Ctrl + C anytime to exit.")
p.add_argument("--gen", metavar="generations", type=int, default=100,
               help="Quantity of generations to run. Default: 100 generations.")
p.add_argument("--level", metavar="level", type=int, default=2,
               help="Select which level from Yoshiland you want. E.g.: Yoshiland{level} Default: Yoshiland2")
p.add_argument("--checkpoint", metavar="checkpoint", type=int, default=0,
               help="Select which generation to load from. Obs.: The checkpoints are made at each 5 generations. Please, select a multiple of 5.")
p.add_argument("--winner", metavar="winner", type=int, default=0,
               help="Select a winner from a generation to play.")
p.add_argument("--norender", action="store_true",
               help="Select this option to disable the screen render during the training.")
p.add_argument("--restart", help="Select this option to restart the training.", action="store_true")

args = p.parse_args()

if args.mode.upper() == "TRAIN":
    if args.restart and args.checkpoint > 0: 
        p.error("You can't choose the checkpoint if the training will restart.")
        exit()

    train.main(args.gen, args.level, args.norender, args.restart, args.checkpoint)

if args.mode.upper() == "PLAY":
    play.main(args.level, args.winner)