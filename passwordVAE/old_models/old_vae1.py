import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

# (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

# def preprocess_images(images):
#     images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
#     return np.where(images > .5, 1.0, 0.0).astype('float32')

# train_images = preprocess_images(train_images)
# test_images = preprocess_images(test_images)

# train_size = 60000
# batch_size = 32
# test_size = 10000

# train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size)
# test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size)

# print(next(iter(train_dataset))[0])

batch_size = 64

infile = open("../100K_rockyou_train.txt")
passwords = infile.read()

tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, lower=False)

# create a character level encoding
tokenizer.fit_on_texts(passwords)

# number of distinct chars
max_id = len(tokenizer.word_index)
# number of chars
dataset_size = tokenizer.document_count

[encoded] = np.array(tokenizer.texts_to_sequences([passwords])) - 1
train_dataset = tf.data.Dataset.from_tensor_slices(encoded)
train_dataset = train_dataset.window(10, drop_remainder=True)
train_dataset = train_dataset.flat_map(lambda window: window.batch(10))



train_dataset = train_dataset.batch(batch_size, drop_remainder=False)

initial = next(iter(train_dataset))

train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self,
                latent_dim=32,
                intermediate_dim=64,
                name='encoder',
                **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self,
                original_dim,
                intermediate_dim=64,
                name='decoder',
                **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = layers.Dense(original_dim, activation='sigmoid')

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                original_dim,
                intermediate_dim=64,
                latent_dim=32,
                name='autoencoder',
                **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                            intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        # self._set_inputs(inputs)
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = - 0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed


vae = VariationalAutoEncoder(10, 8, 6)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

for epoch in range(1):
    print('Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # !!! uncomment the following two lines to use workaround and skip !!!
            # if step == 0 and epoch == 0:
            #   vae._set_inputs(x_batch_train)
            reconstructed = vae(x_batch_train)
            # Compute reconstruction loss
            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vae.losses)  # Add KLD regularization loss

        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        loss_metric(loss)

        if step % 100 == 0:
            print('step %s: mean loss = %s' % (step, loss_metric.result()))

print(vae(initial))
















# class CVAE(tf.keras.Model):

#     def __init__(self, latent_dim):
#         super(CVAE, self).__init__()
#         self.latent_dim = latent_dim
#         self.encoder = tf.keras.Sequential([
#             tf.keras.layers.InputLayer(input_shape=(10,)),
#             tf.keras.layers.Conv2D(
#                 filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
#             tf.keras.layers.Conv2D(
#                 filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
#             tf.keras.layers.Flatten(),
#             # No activation
#             tf.keras.layers.Dense(latent_dim + latent_dim),
#         ])

#         self.decoder = tf.keras.Sequential([
#             tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
#             tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
#             tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
#             tf.keras.layers.Conv2DTranspose(
#                 filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
#             tf.keras.layers.Conv2DTranspose(
#                 filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
#             # No activation
#             tf.keras.layers.Conv2DTranspose(
#                 filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
#         ])

#     @tf.function
#     def sample(self, eps=None):
#         if eps is None:
#             eps = tf.random.normal(shape=(100, self.latent_dim))
#         return self.decode(eps, apply_sigmoid=True)

#     def encode(self, x):
#         mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
#         return mean, logvar

#     def reparameterize(self, mean, logvar):
#         eps = tf.random.normal(shape=mean.shape)
#         return eps * tf.exp(logvar * .5) + mean

#     def decode(self, z, apply_sigmoid=False):
#         logits = self.decoder(z)
#         if apply_sigmoid:
#             probs = tf.sigmoid(logits)
#             return probs
#         return logits

# optimizer = tf.keras.optimizers.Adam(1e-4)


# def log_normal_pdf(sample, mean, logvar, raxis=1):
#     log2pi = tf.math.log(2. * np.pi)
#     return tf.reduce_sum(
#         -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
#         axis=raxis)


# def compute_loss(model, x):
#     mean, logvar = model.encode(x)
#     z = model.reparameterize(mean, logvar)
#     x_logit = model.decode(z)
#     cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
#     logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
#     logpz = log_normal_pdf(z, 0., 0.)
#     logqz_x = log_normal_pdf(z, mean, logvar)
#     return -tf.reduce_mean(logpx_z + logpz - logqz_x)


# @tf.function
# def train_step(model, x, optimizer):
#     """Executes one training step and returns the loss.

#     This function computes the loss and gradients, and uses the latter to
#     update the model's parameters.
#     """
#     with tf.GradientTape() as tape:
#         loss = compute_loss(model, x)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# epochs = 10
# # set the dimensionality of the latent space to a plane for visualization later
# latent_dim = 2
# num_examples_to_generate = 16

# # keeping the random vector constant for generation (prediction) so
# # it will be easier to see the improvement.
# random_vector_for_generation = tf.random.normal(
#     shape=[num_examples_to_generate, latent_dim])
# model = CVAE(latent_dim)

# def generate_and_save_images(model, epoch, test_sample):
#     mean, logvar = model.encode(test_sample)
#     z = model.reparameterize(mean, logvar)
#     predictions = model.sample(z)
#     fig = plt.figure(figsize=(4, 4))

#     for i in range(predictions.shape[0]):
#         plt.subplot(4, 4, i + 1)
#         plt.imshow(predictions[i, :, :, 0], cmap='gray')
#         plt.axis('off')

#     # tight_layout minimizes the overlap between 2 sub-plots
#     plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
#     plt.show()

# def get_terminal_width():
#         width = shutil.get_terminal_size(fallback=(200, 24))[0]
#         if width == 0:
#             width = 120
#         return width

# def progress_bar(total_passwords, batch_size, epoch, epochs):
#     bar = tqdm(total=total_passwords,
#             ncols=int(get_terminal_width() * .9),
#             desc=tqdm.write(f'Epoch {epoch + 1}/{epochs}'),
#             bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  '
#                         'ETA: {remaining}  Elapsed Time: {elapsed}',
#             unit=' passwords',
#             miniters=10)
#     return bar

# # Pick a sample of the test set for generating output images
# assert batch_size >= num_examples_to_generate
# for test_batch in test_dataset.take(1):
#     test_sample = test_batch[0:num_examples_to_generate, :, :, :]

# generate_and_save_images(model, 0, test_sample)


# for epoch in range(0, epochs):
#     # bar = progress_bar(train_size, batch_size, epoch, epochs)
#     start_time = time.time()
#     for train_x in train_dataset:
#         train_step(model, train_x, optimizer)
#         # bar.update(batch_size)
#     end_time = time.time()

#     loss = tf.keras.metrics.Mean()
#     for test_x in test_dataset:
#         loss(compute_loss(model, test_x))
#     elbo = -loss.result()
#     print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
#             .format(epoch, elbo, end_time - start_time))
#     generate_and_save_images(model, epoch, test_sample)