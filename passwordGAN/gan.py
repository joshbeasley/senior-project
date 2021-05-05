import os
from datetime import datetime
from tqdm.autonotebook import tqdm
import shutil
from functools import partial
import numpy as np
import pickle

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.backend as K
from tensorflow.python.keras import metrics
from tensorflow import random

from gan_submodels import ResBlock, Discriminator, Generator

tf.config.experimental_run_functions_eagerly(True)

class passwordGAN:
    def __init__(self):
        """ Initializes model and tokenizer """
        self.G = Generator(layer_dim=128, seq_len=10)
        self.D = Discriminator(layer_dim=128, seq_len=10)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, lower=False)

    @staticmethod
    def add_arguments(parser):
        """ Adds the below arguments to the given parser.  
            parser -- an argparse object
        """
        parser.add_argument("--epochs", action="store", type=int, default=10, help="number of epochs to train for")
        parser.add_argument("--batch", action="store", type=int, default=256, help="training batch size")
        parser.add_argument("--gpus", action="store", type=int, default=1, help="number of GPUs that are being used to train the model")
        parser.add_argument("--iterations", action="store", type=int, default=10000, help="number of iterations to train the GAN for per epoch")
        parser.add_argument("--n_critic", action="store", type=int, default=10, help="number of critic updates per generator update")
        parser.add_argument("--checkpoints", action="store", type=int, default=5000, help="Number of iterations per checkpoint")

    def tokenize_training_data(self, infile, batch_size):
        """ Reads the infile, and tokenizes all unique characters.
            Tokenized passwords are used to create dataset which is
            returned.

            infile -- a file containing appropriately formatted training data
            args -- if not None, the arguments returned by the parser passed
                    to add_arguments
        """
        passwords = infile.read()

        # create a character level encoding
        self.tokenizer.fit_on_texts(passwords)
    
        # number of distinct chars
        max_id = len(self.tokenizer.word_index)
        # number of chars
        dataset_size = self.tokenizer.document_count

        [encoded] = np.array(self.tokenizer.texts_to_sequences([passwords])) - 1
        train_dataset = tf.data.Dataset.from_tensor_slices(encoded)

        train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset

    def train(self, infile, args):
        """ Trains the GAN using data read from the given infile.

            infile -- a file containing appropriately formatted training data
            args -- if not None, the arguments returned by the parser passed
                    to add_arguments
        """
        dataset = self.tokenize_training_data(infile, args.batch)
        g_train_loss = metrics.Mean()
        d_train_loss = metrics.Mean()
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_directory = "./checkpoints/training_checkpoints"
        g_checkpoint_prefix = os.path.join(checkpoint_directory + "/generator", "ckpt")
        d_checkpoint_prefix = os.path.join(checkpoint_directory + "/discriminator", "ckpt")


        for epoch in tf.range(args.epochs):
            epoch = tf.cast(epoch, dtype=tf.int64, name=epoch)
            bar = self.progress_bar(self.tokenizer.document_count, args.batch, epoch, args.epochs)
            for iteration, batch in zip(range(args.iterations), dataset):
                for _ in tf.range(args.n_critic):
                    real = tf.reshape(tf.dtypes.cast(batch, tf.float32), [2, 1, 32])
                    self.train_discriminator(real)
                    d_loss = self.train_discriminator(real)
                    d_train_loss(d_loss)

                g_loss = self.train_generator()
                g_train_loss(g_loss)
                self.train_generator()

                bar.postfix['g_loss'] = f'{g_train_loss.result():6.3f}'
                bar.postfix['d_loss'] = f'{d_train_loss.result():6.3f}'
                bar.update(args.batch)

                if iteration % args.checkpoints == 0 and iteration > 0:
                    generator_checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.G)
                    generator_checkpoint.save(file_prefix=g_checkpoint_prefix)

                    discriminator_checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.D)
                    discriminator_checkpoint.save(file_prefix=d_checkpoint_prefix)

            tf.saved_model.save(self.G, './models/generator/' + args.save)
            tf.saved_model.save(self.D, './models/discriminator/' + args.save)
            with open('./models/tokenizer/{}.pickle'.format(args.save), 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            g_train_loss.reset_states()
            d_train_loss.reset_states()

            bar.close()
            del bar

    @tf.function
    def train_generator(self):
        """ Trains the generator submodel
        """
        z = tf.random.normal([2, 1, 32], dtype=tf.dtypes.float32)
        with tf.GradientTape() as t:
            t.watch(z)
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            loss = self.wasserstein_loss(fake_logits)
        grad = t.gradient(loss, self.G.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.G.trainable_variables))
        return loss

    @tf.function
    def train_discriminator(self, real):
        """ Trains the discriminator submodel
        """
        z = tf.random.normal([2, 1, 32], dtype=tf.dtypes.float32)
        with tf.GradientTape() as t:
            t.watch(z)
            real_logits = self.D(real, training=True)
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            cost = self.discriminator_loss(fake_logits, real_logits)
            gp = self.gradient_penalty(partial(self.D, training=True), real, x_fake)
            cost += 10.0 * gp
        grad = t.gradient(cost, self.D.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.D.trainable_variables))
        return cost

    def gradient_penalty(self, f, real, fake):
        """ A function that applies the gradient penalty for training the discriminator
        """
        alpha = random.uniform([2, 1, 32], 0., 1.)
        diff = fake - real
        inter = real + (alpha * diff)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = f(inter)
        grad = t.gradient(pred, [inter])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2]))
        gp = tf.reduce_mean((slopes - 1.) ** 2)
        return gp
    
    def discriminator_loss(self, f_logit, r_logit):
        """ Loss function for the discriminator submodel
        """
        f_loss = tf.reduce_mean(f_logit)
        r_loss = tf.reduce_mean(r_logit)
        return f_loss - r_loss

    def wasserstein_loss(self, f_logit):
        """ Function that calculates wasserstein loss
        """
        f_loss = -tf.reduce_mean(f_logit)
        return f_loss

    def load(self, filename):
        """ Loads the neural network and tokenizer that defines from filename

            filename -- the name of a file containing data saved by the save
                        method for a compatible policy
        """
        self.G = tf.keras.models.load_model("models/generator/{}".format(filename))
        self.D = tf.keras.models.load_model("models/discriminator/{}".format(filename))
        with open('models/tokenizer/{}.pickle'.format(filename), 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    @tf.function
    def generate_samples(self, num):
        """ Generates 'num' number of samples using the generator

            num -- number of samples to generate
        """
        output = []
        for i in range(num):
            z = tf.convert_to_tensor(tf.random.normal([2, 1, 32], dtype=tf.dtypes.float32))
            samples = self.G(z, training=False)
            samples = samples.numpy()
            samples = np.argmax(samples, axis=2)
            for i in range(len(samples)):
                decoded = []
                for j in range(len(samples[i])):
                    decoded.append([samples[i][j]])
                print(tokenizer.sequences_to_texts(decoded)[0], end="")
    
    def get_terminal_width(self):
        """ Gets the terminal width to properly display the progress bar
        """
        width = shutil.get_terminal_size(fallback=(200, 24))[0]
        if width == 0:
            width = 120
        return width

    def progress_bar(self, total_passwords, batch_size, epoch, epochs):
        """ Creates an instance of the progress bar class

            total_passwords -- total number of passwords in the dataset
            batch_size -- batch size for the training process
            epoch -- current epoch
            epochs -- total number of epochs
        """
        bar = tqdm(total=total_passwords * epochs,
                ncols=int(self.get_terminal_width() * .9),
                desc=tqdm.write(f'Epoch {epoch + 1}/{epochs}'),
                postfix={
                    'g_loss': f'{0:6.3f}',
                    'd_loss': f'{0:6.3f}',
                    1: 1
                },
                bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  '
                            'ETA: {remaining}  Elapsed Time: {elapsed}  '
                            'G Loss: {postfix[g_loss]}  D Loss: {postfix['
                            'd_loss]}',
                unit=' passwords',
                miniters=10)
        return bar
