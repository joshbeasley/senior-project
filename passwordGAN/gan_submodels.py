import tensorflow as tf

class ResBlock(tf.keras.Model):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.res_block = tf.keras.Sequential([
            tf.keras.layers.ReLU(True),
            tf.keras.layers.Conv1D(dim, dim, 5, padding='same'),
            tf.keras.layers.ReLU(True),
            tf.keras.layers.Conv1D(dim, dim, 5, padding='same'),
        ])

    def call(self, input, **kwargs):
        output = self.res_block(input)
        return input + (0.3 * output)

class Discriminator(tf.keras.Model):
    def __init__(self, layer_dim, seq_len):
        super(Discriminator, self).__init__()
        dim = layer_dim
        self.dim = layer_dim
        self.seq_len = seq_len

        self.block = tf.keras.Sequential([
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
        ])
        self.conv1d = tf.keras.layers.Conv1D(dim, 32, 1, padding='valid')
        self.linear = tf.keras.layers.Dense(seq_len * dim, activation='linear')

    def call(self, input, **kwargs):
        output = tf.transpose(input, [0, 2, 1])
        output = self.conv1d(output)
        output = self.block(output)
        output = tf.reshape(output, (-1, 64, 4))
        output = self.linear(output)
        return output

class Generator(tf.keras.Model):
    def __init__(self, layer_dim, seq_len):
        super(Generator, self).__init__()
        dim = layer_dim
        self.dim = layer_dim
        self.seq_len = seq_len

        self.fc1 = tf.keras.layers.Dense(128, activation='linear', input_shape=(dim * seq_len,))
        self.block = tf.keras.Sequential([
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
        ])
        self.conv1 = tf.keras.layers.Conv1D(64, 32, 1, padding='valid')
        self.softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, noise, **kwargs):
        output = self.fc1(noise)
        output = tf.reshape(output, (-1, 2, 128))
        output = self.block(output)
        output = tf.reshape(output, [1, 32, 8])
        output = self.conv1(output)
        output = tf.transpose(output, [0, 2, 1])
        output = self.softmax(output)
        return tf.reshape(output, [2, 1, 32])