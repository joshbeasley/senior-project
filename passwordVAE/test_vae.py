import tensorflow as tf
from tensorflow import keras
import numpy as np
import vae
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train and test VAE neural network for password cracking")
    parser.add_argument("--test", action="store_true", dest="test", help="generate passwords using trained or loaded VAE")
    parser.add_argument("--num", "-n", action="store", dest="num", type=int, default=30, help="number of passwords to generate")
    parser.add_argument("--train", "-t", action = "store", dest="train", help="train VAE using data from file")
    parser.add_argument("--save", action="store", dest="save", help="save trained VAE")
    parser.add_argument("--load", action="store", dest="load", help="load trained VAE")
    vae.passwordVAE.add_arguments(parser)
    args = parser.parse_args()

    if args.train is None and args.load is None:
        parser.error("must choose --train or --load")
    if args.train is not None and args.load is not None:
        parser.error("cannot choose both --train and --load")

    model = vae.passwordVAE()

    # train or load VAE
    if args.train is not None:
        with open(args.train) as infile:
            model.train(infile, args)
    else:
        model.load(args.load)

    # save VAE
    if args.save is not None:
        model.save(args.save)

    # evaluate VAE
    if args.test:
        model.generate_passwords("output.txt", args.num)

""" Example Commands
    python3 test_vae.py --train 100K_rockyou_train.txt --save the_first_vae --epochs 10 --batch 64 --gpus 2
    python3 test_vae.py --test --num 100 --load the_first_vae
"""