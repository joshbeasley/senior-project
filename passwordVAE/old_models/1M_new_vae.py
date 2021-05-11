import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, Lambda, Layer, Bidirectional
from tensorflow.keras.losses import categorical_crossentropy
from matplotlib import pyplot as plt
import pandas as pd
import random
import argparse
import pickle

tf.compat.v1.disable_eager_execution()

infile = open("../datasets/1M_rockyou_train.txt")

padded_passwords = []
charset = set("_")    
for p in infile.readlines():
    p = p[:-1]
    padded_passwords.append(p.ljust(32, "_"))
    charset |= set(p)

# split passwords into a training and validation set
train_size = len(padded_passwords) * 90 // 100
train_dataset = padded_passwords[:train_size]
val_dataset = padded_passwords[train_size:]

# Convert characters to integers 
vocab_size = len(charset)
char2id = dict((c, i) for i, c in enumerate(charset))

# One hot encode the passwords
encoded_train_passwords = [[char2id[c] for c in password] for password in train_dataset]
encoded_val_passwords = [[char2id[c] for c in password] for password in val_dataset]

one_hot_encoded_train = np.array([to_categorical(p, num_classes=vocab_size) for p in encoded_train_passwords])
one_hot_encoded_train = one_hot_encoded_train[:((len(one_hot_encoded_train) // 64) * 64)]

one_hot_encoded_val = np.array([to_categorical(p, num_classes=vocab_size) for p in encoded_val_passwords])
one_hot_encoded_val = one_hot_encoded_val[:((len(one_hot_encoded_val) // 64) * 64)]

def create_lstm_vae(timesteps, layer_sizes, vocab_size, epsilon_std=1.,
                    batch_size=64):

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(layer_sizes[-1],),
                                mean=0., stddev=epsilon_std)
        return z_mean + K.exp(.5 * z_log_sigma) * epsilon
    
    # Create encoder model
    enc_input = Input(batch_shape=(batch_size, timesteps, vocab_size))
    x = enc_input
    for idx, layer_size in enumerate(layer_sizes):
        ret_seq = (idx != len(layer_sizes) - 1) # False for the last layer_size
        x = Bidirectional(LSTM(layer_size, return_sequences=ret_seq))(x)
    enc_output = Dense(layer_sizes[-1], activation="relu")(x)
    z_mean = Dense(layer_sizes[-1])(enc_output)
    z_log_sigma = Dense(layer_sizes[-1])(enc_output)
    z = Lambda(sampling, output_shape=(layer_sizes[-1],))([z_mean, z_log_sigma])
    encoder = tf.keras.Model(enc_input, z_mean, name="Encoder")

    # Create decoder model
    bottleneck_size = layer_sizes[-1]
    dec_input = Input((bottleneck_size,))
    layer = RepeatVector(timesteps)
    x = layer(z)
    _x = layer(dec_input)
    for layer_size in layer_sizes[::-1][1:]:
        layer = Bidirectional(LSTM(layer_size, return_sequences=True))
        x = layer(x)
        _x = layer(_x)
    layer =  TimeDistributed(Dense(vocab_size, activation="softmax"))
    dec_output = layer(x)
    _dec_output = layer(_x)
    decoder = tf.keras.Model(dec_input, _dec_output, name="Decoder")

    # Create autoencoder model
    autoencoder = tf.keras.Model(enc_input, dec_output, name="Autoencoder")

    # Variational autoencoder custom loss categorical entropy loss + KL loss
    def vae_loss(x, x_decoded_mean):
        xent_loss = categorical_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        xent_loss = K.sum(xent_loss, axis=-1)
        return xent_loss + kl_loss

    autoencoder.compile(loss=vae_loss, optimizer="adam", metrics=['categorical_accuracy'])
    return encoder, decoder, autoencoder

variational_encoder, variational_decoder, variational_autoencoder = create_lstm_vae(32, [16, 10, 6], vocab_size)
print(variational_encoder.summary())


variational_autoencoder.fit(one_hot_encoded_train, one_hot_encoded_train, epochs=25, batch_size=64)


variational_autoencoder.save("1M_new_vae")
variational_decoder.save("1M_new_vae_decoder")
with open('1M_new_vae_charset.pickle', 'wb') as handle:
    pickle.dump(charset, handle, protocol=pickle.HIGHEST_PROTOCOL)


unpad = lambda text: text.replace("_", "")
one_hot_decode = lambda one_hot_vectors: "".join([list(charset)[np.argmax(vec)] for vec in one_hot_vectors])

mu, sigma = 0, 3
for i in range(1000000):
    latent_sample = np.array([np.random.normal(mu, sigma, 6)])
    new_password_vec = variational_decoder.predict(latent_sample)
    new_password_str = unpad(one_hot_decode(new_password_vec[0]))
    print(new_password_str)
