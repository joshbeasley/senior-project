import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import pandas as pd
import random
import argparse
import pickle

tf.compat.v1.disable_eager_execution()

class passwordVAE:
    def __init__(self):
        """ Initializes model and other key variables """
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.max_password_len = 32
        self.padding_char = "-"
        self.batch_size = None
        self.epochs = None
        self.vocab_size = None
        self.charset = None
        self.epsilon_std = 1.0
        self.layer_sizes = [16, 10, 6]
        self.train_dataset = None
        self.val_dataset = None

    @staticmethod
    def add_arguments(parser):
        """ Adds the below arguments to the given parser.  
            parser -- an argparse object
        """
        parser.add_argument("--epochs", action="store", type=int, default=10, help="number of epochs to train for")
        parser.add_argument("--batch", action="store", type=int, default=256, help="training batch size")
        parser.add_argument("--gpus", action="store", type=int, default=1, help="number of GPUs that are being used to train the model")

    def tokenize_training_data(self, infile):
        """ Reads the infile, and tokenizes all unique characters.
            Pads all tokenized passwords so that they are of consistent
            length for the LSTM cells.
            Encodes each password using a one-hot format

            infile -- a file containing appropriately formatted training data
        """
        padded_passwords = []
        self.charset = set(self.padding_char)    
        for p in infile.readlines():
            p = p[:-1]
            padded_passwords.append(p.ljust(self.max_password_len, self.padding_char))
            self.charset |= set(p)

        # split passwords into a training and validation set
        train_size = len(padded_passwords) * 90 // 100
        train_dataset = padded_passwords[:train_size]
        val_dataset = padded_passwords[train_size:]

        # Convert characters to integers 
        self.vocab_size = len(self.charset)
        char2id = dict((c, i) for i, c in enumerate(self.charset))

        # One hot encode the passwords
        encoded_train_passwords = [[char2id[c] for c in password] for password in train_dataset]
        encoded_val_passwords = [[char2id[c] for c in password] for password in val_dataset]

        one_hot_encoded_train = np.array([to_categorical(p, num_classes=self.vocab_size) for p in encoded_train_passwords])
        one_hot_encoded_train = one_hot_encoded_train[:((len(one_hot_encoded_train) // 64) * 64)]

        one_hot_encoded_val = np.array([to_categorical(p, num_classes=self.vocab_size) for p in encoded_val_passwords])
        one_hot_encoded_val = one_hot_encoded_val[:((len(one_hot_encoded_val) // 64) * 64)]

        return one_hot_encoded_train, one_hot_encoded_val

    def train(self, infile, args):
        """ Trains the RNN using data read from the given infile.

            infile -- a file containing appropriately formatted training data
            args -- if not None, the arguments returned by the parser passed
                    to add_arguments
        """
        self.batch_size = args.batch
        self.epochs = args.epochs
        self.train_dataset, self.val_dataset = self.tokenize_training_data(infile)
        layer_sizes = self.layer_sizes
        epsilon_std = self.epsilon_std

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(layer_sizes[-1],),
                                    mean=0., stddev=epsilon_std)
            return z_mean + K.exp(.5 * z_log_sigma) * epsilon

        # ENCODER MODEL
        encoder_input = tf.keras.layers.Input(batch_shape=(self.batch_size, self.max_password_len, self.vocab_size))

        x = encoder_input
        for idx, layer_size in enumerate(self.layer_sizes):
            ret_seq = (idx != len(self.layer_sizes) - 1) 
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(layer_size, return_sequences=ret_seq))(x)
        encoder_output = tf.keras.layers.Dense(self.layer_sizes[-1], activation="relu")(x)

        z_mean = tf.keras.layers.Dense(self.layer_sizes[-1])(encoder_output)
        z_log_sigma = tf.keras.layers.Dense(self.layer_sizes[-1])(encoder_output)
        z = tf.keras.layers.Lambda(sampling, output_shape=(self.layer_sizes[-1],))([z_mean, z_log_sigma])
        self.encoder = tf.keras.Model(encoder_input, z_mean, name="Encoder")

        # DECODER MODEL
        bottleneck_size = self.layer_sizes[-1]
        decoder_input = tf.keras.layers.Input((bottleneck_size,))
        layer = tf.keras.layers.RepeatVector(self.max_password_len)

        x = layer(z)
        _x = layer(decoder_input)
        for layer_size in self.layer_sizes[::-1][1:]:
            layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(layer_size, return_sequences=True))
            x = layer(x)
            _x = layer(_x)

        layer =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_size, activation="softmax"))
        decoder_output = layer(x)
        _decoder_output = layer(_x)
        self.decoder = tf.keras.Model(decoder_input, _decoder_output, name="Decoder")

        # Create autoencoder model
        self.vae = tf.keras.Model(encoder_input, decoder_output, name="Autoencoder")

        # Variational autoencoder custom loss categorical entropy loss + KL loss
        def vae_loss(x, x_decoded_mean):
            xent_loss = tf.keras.losses.categorical_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
            xent_loss = K.sum(xent_loss, axis=-1)
            return xent_loss + kl_loss

        self.vae.compile(loss=vae_loss, optimizer="adam", metrics=['categorical_accuracy'])
        self.vae.summary()

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=2)
        callbacks = [early_stopping_cb]

        history = self.vae.fit(self.train_dataset, 
                               self.train_dataset, 
                               epochs=self.epochs, 
                               batch_size=self.batch_size, 
                               validation_data=(self.val_dataset, self.val_dataset), 
                               callbacks=callbacks)

        # plot train and validation loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
        plt.savefig('plots/{}.png'.format(args.save))


    def load(self, filename):
        """ Loads the neural network and tokenizer that defines from filename

            filename -- the name of a file containing data saved by the save
                        method for a compatible policy
        """
        # self.vae = tf.keras.models.load_model(filename)
        self.decoder = tf.keras.models.load_model("models/{}_decoder".format(filename))
        with open('models/{}_charset.pickle'.format(filename), 'rb') as handle:
            self.charset = pickle.load(handle)

    def save(self, filename):
        """ Saves the neural network and tokenizer that defines to filename

            filename -- the name of a writable file
        """
        self.vae.save("models/{}".format(filename))
        self.decoder.save("models/{}_decoder".format(filename))
        with open('models/{}_charset.pickle'.format(filename), 'wb') as handle:
            pickle.dump(self.charset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def unpad(self, text):
        """ removes padding from a given a piece of text

            text -- a piece of padded text (padded with self.padding_char)
        """
        return text.replace("_", "")

    def one_hot_decode(self, vectors):
        """ converts a one hot vector back to the corresponding character

            vectors -- an array of one hot vectors
        """
        #self.charset = {'C', '|', '<', "'", 's', '$', 'M', 'e', ')', '=', '_', 'i', 'k', 'Z', '[', '+', 'j', '2', 'W', '3', 'X', 'K', 'd', 'q', 'Y', '\\', 'R', 'v', 'm', '&', 'S', 'b', 'h', ';', 'g', 'N', 'H', '1', '-', 'J', 'P', ':', '"', '*', ' ', '0', 'I', 'O', '{', '5', '?', '/', 'L', 'U', '@', 'B', 'w', 'c', 'V', '~', '4', '#', 'G', '>', 'T', 'r', 'f', '(', 'E', 'n', 'u', 'p', '}', '8', ',', 'Q', 'x', 'A', '6', 't', 'y', '`', 'o', 'l', 'F', '.', '^', 'a', ']', '9', 'D', '7', 'z', '%', '!'}
        return "".join([list(self.charset)[np.argmax(vec)] for vec in vectors])

    def reconstruct_passwords(self, num):
        """ Reconstructs passwords from the original training set
            by providing them as input to the full VAE model

            num -- number of batches to reconstruct
        """
        reconstructed_passwords = self.vae.predict(self.train_dataset[0:(num * self.batch_size)], batch_size=self.batch_size)
        reconstructed_passwords = [self.unpad(self.one_hot_decode(p)) for p in reconstructed_passwords]

        for pw in reconstructed_passwords:
            print(pw)

    def generate_passwords(self, outfile, num=30):
        """ Generates passwords from a random normal distribution using the decoder
            side of the trained VAE

            num -- number of passwords to generate
            outfile -- file location to output these passwords to
        """
        f = open(outfile, "w")
        mu, sigma = 0, 3
        new_passwords = []
        for i in range(num):
            latent_sample = np.array([np.random.normal(mu, sigma, 6)])
            new_password_vec = self.decoder.predict(latent_sample)
            new_password_str = self.unpad(self.one_hot_decode(new_password_vec[0]))
            new_passwords.append(new_password_str)
            print(new_password_str)
            f.write(new_password_str)
        f.close()



