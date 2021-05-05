import tensorflow as tf
from tensorflow import keras
import numpy as np
import lstm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train and test LSTM recurrent neural network for password cracking")
    parser.add_argument("--test", action="store_true", dest="test", help="generate passwords using trained or loaded RNN")
    parser.add_argument("--chars", "-c", action="store", dest="chars", type=int, default=100, help="number of characters to generate")
    parser.add_argument("--temp", action="store", dest="temp", type=float, default=1, help="temperature for character generation")
    parser.add_argument("--train", "-t", action = "store", dest="train", help="train RNN using data from file")
    parser.add_argument("--save", action="store", dest="save", help="save trained RNN")
    parser.add_argument("--load", action="store", dest="load", help="load trained RNN")
    lstm.passwordLSTM.add_arguments(parser)
    args = parser.parse_args()

    if args.train is None and args.load is None:
        parser.error("must choose --train or --load")
    if args.train is not None and args.load is not None:
        parser.error("cannot choose both --train and --load")

    model = lstm.passwordLSTM()

    # train or load RNN
    if args.train is not None:
        with open(args.train) as infile:
            model.train(args.save, infile, args)
    else:
        model.load(args.load)

    # save RNN
    if args.save is not None:
        model.save(args.save)

    # evaluate RNN
    if args.test:
        print(model.complete_text("p", "output.txt", args.chars, args.temp))
        model.get_accuracy("output.txt", "rockyou_test.txt")

""" Example Commands
    python3 test_lstm.py --train ../datasets/100K_rockyou_train.txt --save 100K_LSTM_model --epochs 1 --batch 512 --gpus 2
    python3 test_lstm.py --test --chars 0.5 --temp 1 --load my_model2
"""