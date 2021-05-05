# Applications of Deep Learning to Password Cracking

A Senior Project by Josh Beasley in partial fulfillment of the requirements for the degree of Bachelors of Science in Computer Science.

Yale School of Engineering and Applied Science, Yale University, May 13, 2021

**Abstract:** Passwords have become the ubiquitous method of online authentication in the modern digital age, yet their flawed nature leaves them open to many security concerns. This project explores the applications of deep learning to password cracking and password security, specifically, by analyzing the way in which the dictionary of likely passwords is constructed for a conventional dictionary attack. Current state-of-the-art password cracking tools like John the Ripper and HashCat rely on a rule-based approach to apply transformations to the existing dictionary and expand its size. I aim to develop and experiment with the effectiveness of various deep learning techniques including recurrent neural networks, generative adversarial networks, and variational autoencoders to test their effectiveness in learning the underlying structure of a leaked password dataset. These models can then be used to expand the size of existing password dictionaries to improve password cracking ability, and in turn, increase password security when users are tasked with creating a password.

## Overview of Repository

The repo contains four main directories: **datasets, passwordGAN, passwordLSTM, and passwordVAE**

The datasets directory contains all of the datasets used in this project for training and testing the neural networks

The models directories each contain a class with all relevant functions for the model, while the test file creates an ArgParse and an instance of the class to train and test from the command line

## Setup

`git clone https://github.com/joshbeasley/senior-thesis.git`

All packages and their corresponding versions are located in the `requirements.txt` file.
Install all packages in your desired conda or venv python environment and you should be good to go!

## passwordLSTM
`cd passwordLSTM`

**Relevant Arguments:**
- `"--test", action="store_true", dest="test", help="generate passwords using trained or loaded RNN"`
- `"--chars", "-c", action="store", dest="chars", type=int, default=100, help="number of characters to generate"`
- `"--temp", action="store", dest="temp", type=float, default=1, help="temperature for character generation"`
- `"--train", "-t", action = "store", dest="train", help="train RNN using data from file"`
- `"--save", action="store", dest="save", help="save trained RNN"`
- `"--load", action="store", dest="load", help="load trained RNN"`
- `"--epochs", action="store", type=int, default=10, help="number of epochs to train for"`
- `"--batch", action="store", type=int, default=256, help="training batch size"`
- `"--gpus", action="store", type=int, default=1, help="number of GPUs that are being used to train the model"`

**Example Commands:**

`python3 test_lstm.py --train ../datasets/100K_rockyou_train.txt --save 100K_LSTM_model --epochs 1 --batch 512 --gpus 2`

`python3 test_lstm.py --test --chars 0.5 --temp 1 --load my_model2`

 

## passwordGAN

`cd passwordGAN`

**Relevant Arguments:**
- `"--test", action="store_true", dest="test", help="generate passwords using trained or loaded RNN"`
- `"--num", "-n", action="store", dest="num", type=int, default=30, help="number of passwords to generate"`
- `"--train", "-t", action = "store", dest="train", help="train RNN using data from file"`
- `"--save", action="store", dest="save", help="save trained RNN"`
- `"--load", action="store", dest="load", help="load trained RNN"`
- `"--epochs", action="store", type=int, default=10, help="number of epochs to train for"`
- `"--batch", action="store", type=int, default=256, help="training batch size"`
- `"--gpus", action="store", type=int, default=1, help="number of GPUs that are being used to train the model"`
- `"--iterations", action="store", type=int, default=1000, help="number of iterations to train the model"`
- `"--n_critic", action="store", type=int, default=10, help="critic updates per generator update"`
- `"--checkpoints", action="store", type=int, default=5000, help="number of iterations before each update to the checkpoint model"`

**Example Commands:**

`python3 test_gan.py --train ../datasets/100K_rockyou_train.txt --save 100K_GAN_model --epochs 1 --iterations 1562 --batch 64 --gpus 2`

`python3 test_gan.py --test --num 100 --load gan_model`


## passwordVAE

`cd passwordVAE`

**Relevant Arguments:**
- `"--test", action="store_true", dest="test", help="generate passwords using trained or loaded RNN"`
- `"--num", "-n", action="store", dest="num", type=int, default=30, help="number of passwords to generate"`
- `"--train", "-t", action = "store", dest="train", help="train RNN using data from file"`
- `"--save", action="store", dest="save", help="save trained RNN"`
- `"--load", action="store", dest="load", help="load trained RNN"`
- `"--epochs", action="store", type=int, default=10, help="number of epochs to train for"`
- `"--batch", action="store", type=int, default=256, help="training batch size"`
- `"--gpus", action="store", type=int, default=1, help="number of GPUs that are being used to train the model"`


**Example Commands:**

`python3 test_vae.py --train ../datasets/100K_rockyou_train.txt --save the_first_vae --epochs 2 --batch 64 --gpus 2`

`python3 test_vae.py --test --num 100 --load the_first_vae`


## References

The code for the LSTM model was written with the assistance of the book *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems* by Aurelien Goren.

The code for the GAN model was written based off of the [PassGAN paper](https://arxiv.org/abs/1709.00440) and using code from [Riathoir's repository](https://github.com/Riathoir/PASSGAN-IWGAN-Tensorflow-2) as a jumping off point.

The code for the VAE was written using code from Andreas Pogiatzis' article [Demystifying Generative Models by Generating Passwords — Part 2](https://towardsdatascience.com/demystifying-generative-models-by-generating-passwords-part-2-38ad3c325a46) as a jumping off point.

