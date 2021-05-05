import tensorflow as tf
import numpy as np
import itertools
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import nltk.data
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import reuters
from nltk.corpus import gutenberg
from nltk.corpus import brown
from nltk.tokenize import sent_tokenize
from gensim.models import KeyedVectors
import pickle

batch_size = 64

with open('w2v.pickle', 'rb') as handle:
    w2v = pickle.load(handle)

# w2v = KeyedVectors.load_word2vec_format('lexvec.commoncrawl.300d.W.pos.vectors')

# with open('w2v.pickle', 'wb') as handle:
#     pickle.dump(w2v, handle, protocol=pickle.HIGHEST_PROTOCOL)

def split_into_sent (text):
    strg = ''
    for word in text:
        strg += word
        strg += ' '
    strg_cleaned = strg.lower()
    for x in ['\xd5d','\n','"',"!", '#','$','%','&','(',')','*','+',',','-','/',':',';','<','=','>','?','@','[','^',']','_','`','{','|','}','~','\t']:
        strg_cleaned = strg_cleaned.replace(x, '')
    sentences = sent_tokenize(strg_cleaned)
    return sentences

def vectorize_sentences(sentences):
    vectorized = []
    for sentence in sentences:
        byword = sentence.split()
        concat_vector = []
        for word in byword:
            try:
                concat_vector.append(w2v[word])
            except:
                pass
        vectorized.append(concat_vector)
    return vectorized

data_concat = []

for t in [brown.words(), reuters.words(), gutenberg.words()]:
    text = split_into_sent(t)
    vect = vectorize_sentences(text)
    data = [x for x in vect if len(x) == 10]
    for x in data:
        data_concat.append(list(itertools.chain.from_iterable(x)))

# wiki_tokens = vectorize_sentences(wiki_tokens)
# wikidata = [x for x in wiki_tokens if len(x) == 10]
# for x in wikidata:
#     data_concat.append(list(itertools.chain.from_iterable(x)))

data_array = np.array(data_concat)
np.random.shuffle(data_array)

print(data_array[0])
print(data_array.shape)
print(len(data_array[0]))
train = data_array[:8000]
test = data_array[8000:10000]

# infile = open("../100K_rockyou_train.txt")
# passwords = infile.read()

# tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, lower=False)

# # create a character level encoding
# tokenizer.fit_on_texts(passwords)

# # number of distinct chars
# max_id = len(tokenizer.word_index)
# # number of chars
# dataset_size = tokenizer.document_count

# [encoded] = np.array(tokenizer.texts_to_sequences([passwords])) - 1
# train_dataset = tf.data.Dataset.from_tensor_slices(encoded)
# train_dataset = train_dataset.window(10, drop_remainder=True)
# train_dataset = train_dataset.flat_map(lambda window: window.batch(10))
# train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
# train_dataset = train_dataset.unbatch()
# # train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
# train_dataset = np.stack(list(train_dataset))

# print(train_dataset.shape)

# batch_size = 500
# original_dim = 3000
# intermediate_dim = 1200
# latent_dim = 1000
# epsilon_std = 1.0

# input_ = tf.keras.layers.Input(batch_shape=(batch_size, original_dim))
# encoder1 = tf.keras.layers.Dense(intermediate_dim, activation="relu")(input_)
# encoder_mean = tf.keras.layers.Dense(latent_dim)(encoder1)
# encoder_log_var = tf.keras.layers.Dense(latent_dim)(encoder1)

# def sampling(args):
#     z_mean, z_log_var = args
#     epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
#                               stddev=epsilon_std)
#     return z_mean + K.exp(z_log_var / 2) * epsilon

# encoder = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,))([encoder_mean, encoder_log_var])

# # we instantiate these layers separately so as to reuse them later
# decoder_h = tf.keras.layers.Dense(intermediate_dim, activation='relu')
# decoder_mean = tf.keras.layers.Dense(original_dim, activation='sigmoid')
# h_decoded = decoder_h(encoder)
# x_decoded_mean = decoder_mean(h_decoded)

# # placeholder loss
# def zero_loss(y_true, y_pred):
#     return K.zeros_like(y_pred)

# # Custom loss layer
# class CustomVariationalLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         self.is_placeholder = True
#         super(CustomVariationalLayer, self).__init__(**kwargs)

#     def vae_loss(self, x, x_decoded_mean, encoder_mean, encoder_log_var):
#         xent_loss = original_dim * tf.keras.metrics.binary_crossentropy(x, x_decoded_mean)
#         kl_loss = - 0.5 * K.sum(1 + encoder_log_var - K.square(encoder_mean) - K.exp(encoder_log_var), axis=-1)
#         return K.mean(xent_loss + kl_loss)

#     def call(self, inputs):
#         x = inputs[0]
#         x_decoded_mean = inputs[1]
#         encoder_mean = inputs[2]
#         encoder_log_var = inputs[3]
#         loss = self.vae_loss(x, x_decoded_mean, encoder_mean, encoder_log_var)
#         self.add_loss(loss, inputs=inputs)
#         # we don't use this output, but it has to have the correct shape:
#         return K.ones_like(x)

# loss_layer = CustomVariationalLayer()([input_, x_decoded_mean, encoder_mean, encoder_log_var])
# vae = tf.keras.Model(input_, [loss_layer])
# vae.compile(optimizer='rmsprop', loss=[zero_loss])
# vae.summary()
# vae.fit(train, train, epochs=3, batch_size=batch_size)


