import tensorflow as tf
from tensorflow import keras
import numpy as np
import gan
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train and test GAN recurrent neural network for password cracking")
    parser.add_argument("--test", action="store_true", dest="test", help="generate passwords using trained or loaded GAN")
    parser.add_argument("--num", "-n", action="store", dest="num", type=int, default=100, help="number of passwords to generate")
    parser.add_argument("--train", "-t", action = "store", dest="train", help="train GAN using data from file")
    parser.add_argument("--save", action="store", dest="save", help="save trained GAN")
    parser.add_argument("--load", action="store", dest="load", help="load trained GAN")
    gan.passwordGAN.add_arguments(parser)
    args = parser.parse_args()

    if args.train is None and args.load is None:
        parser.error("must choose --train or --load")
    if args.train is not None and args.load is not None:
        parser.error("cannot choose both --train and --load")

    model = gan.passwordGAN()

    # train or load GAN
    if args.train is not None:
        with open(args.train) as infile:
            model.train(infile, args)
    else:
        model.load(args.load)

    # TODO
    # if args.test:
    #     print(model.complete_text("p", "output.txt", args.chars, args.temp))
    #     model.get_accuracy("output.txt", "rockyou_test.txt")

""" Example Commands
    python3 test_gan.py --train ../datasets/100K_rockyou_train.txt --save 100K_GAN_model --epochs 1 --iterations 1562 --batch 64 --gpus 2
    python3 test_gan.py --test --num 100 --load my_model2
"""





