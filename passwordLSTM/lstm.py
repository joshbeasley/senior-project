import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pickle

class passwordLSTM:
    def __init__(self):
        """ Initializes model and tokenizer """
        self._model = None
        self._tokenizer = keras.preprocessing.text.Tokenizer(char_level=True, lower=False)

    @staticmethod
    def add_arguments(parser):
        """ Adds the below arguments to the given parser.  
            parser -- an argparse object
        """
        parser.add_argument("--epochs", action="store", type=int, default=10, help="number of epochs to train for")
        parser.add_argument("--batch", action="store", type=int, default=256, help="training batch size")
        parser.add_argument("--gpus", action="store", type=int, default=1, help="number of GPUs that are being used to train the model")

    def tokenize_training_data(self, infile, args):
        """ Reads the infile, and tokenizes all unique characters.
            Tokenized passwords are used to create dataset which is
            returned.

            infile -- a file containing appropriately formatted training data
            args -- if not None, the arguments returned by the parser passed
                    to add_arguments
        """
        # import and open the password file
        passwords = infile.read()
        print("Successfully loaded password file")

        # create a character level encoding
        self._tokenizer.fit_on_texts(passwords)
    
        # number of distinct chars
        max_id = len(self._tokenizer.word_index)
        # number of chars
        dataset_size = self._tokenizer.document_count

        [encoded] = np.array(self._tokenizer.texts_to_sequences([passwords])) - 1
        print("Dataset tokenized")

        train_size = dataset_size * 90 // 100
        train_dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
        val_dataset = tf.data.Dataset.from_tensor_slices(encoded[train_size:])

        # window size
        # TODO:
        # - adjust window size so it is optimal dependent on password length
        n_steps = 10
        window_length = n_steps + 1
        train_dataset = train_dataset.window(window_length, shift=1, drop_remainder=True)
        val_dataset = val_dataset.window(window_length, shift=1, drop_remainder=True)

        # convert the nested dataset into a flat dataset of tensors for training
        train_dataset = train_dataset.flat_map(lambda window: window.batch(window_length))
        val_dataset = val_dataset.flat_map(lambda window: window.batch(window_length))

        # shuffle the windows and separate the inputs from the targets
        batch_size = args.batch
        train_dataset = train_dataset.shuffle(10000).batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        train_dataset = train_dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
        val_dataset = val_dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

        # encode input features as one-hot vectors
        train_dataset = train_dataset.map(
            lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
        val_dataset = val_dataset.map(
            lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))

        train_dataset = train_dataset.prefetch(1)
        val_dataset = val_dataset.prefetch(1)

        return train_dataset, val_dataset

    def train(self, name, infile, args):
        """ Trains the RNN using data read from the given infile.

            infile -- a file containing appropriately formatted training data
            args -- if not None, the arguments returned by the parser passed
                    to add_arguments
        """
        train_dataset, val_dataset = self.tokenize_training_data(infile, args)

        n_gpus = args.gpus
        device_type = 'GPU'
        devices = tf.config.experimental.list_physical_devices(
                device_type)
        devices_names = [d.name.split('e:')[1] for d in devices]
        print("Using the following {} GPUs: {}".format(n_gpus, devices_names))
        strategy = tf.distribute.MirroredStrategy(
                devices=devices_names[:n_gpus])

        with strategy.scope():
            self._model = keras.models.Sequential([
                keras.layers.LSTM(1024, return_sequences=True, input_shape=[None, len(self._tokenizer.word_index)], dropout=0.2),
                keras.layers.LSTM(1024, return_sequences=True, dropout=0.2),
                keras.layers.TimeDistributed(keras.layers.Dense(len(self._tokenizer.word_index), activation="softmax")),
                keras.layers.TimeDistributed(keras.layers.Dense(len(self._tokenizer.word_index), activation="softmax"))
            ])

            opt = keras.optimizers.Adam(learning_rate=(0.001 * n_gpus))
            self._model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=[keras.metrics.SparseCategoricalAccuracy()])
        print("Model Created")

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=2)
        model_checkpoint_cb = keras.callbacks.ModelCheckpoint("models/{}".format(name), save_best_only=True, save_weights_omly=False)
        callbacks = [early_stopping_cb, model_checkpoint_cb]

        history = self._model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=callbacks)
        print("Training Complete")

        # plot train and validation loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
        plt.savefig('plots/{}.png'.format(name))

    def load(self, filename):
        """ Loads the neural network and tokenizer that defines from filename

            filename -- the name of a file containing data saved by the save
                        method for a compatible policy
        """
        self._model = keras.models.load_model("models/{}".format(filename))
        with open('models/{}.pickle'.format(filename), 'rb') as handle:
            self._tokenizer = pickle.load(handle)

        trainable_count = np.sum([K.count_params(w) for w in self._model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in self._model.non_trainable_weights])

        print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))


    def save(self, filename):
        """ Saves the neural network and tokenizer that defines to filename

            filename -- the name of a writable file
        """
        # self._model.save(filename) not needed anymore since I'm using the ModelCheckpoint callback function
        with open('models/{}.pickle'.format(filename), 'wb') as handle:
            pickle.dump(self._tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def preprocess(self, text):
        """ Preprocesses textual data so it can be used as input to the 
            network for testing purposes

            text -- the textual data (password)
        """
        X = np.array(self._tokenizer.texts_to_sequences(text)) - 1
        return tf.one_hot(X, len(self._tokenizer.word_index))

    def next_char(self, text, temperature=1):
        """ Given a piece of text, rpedicts the next character using the LSTM model

            text -- the textual data (password)
            temperature -- used to control the randomness of predictions by scaling the logits before applying softmax
        """
        text = text[-10:]
        X_new = self.preprocess([text])
        y_proba = self._model.predict(X_new)[0, -1:, :]
        rescaled_logits = tf.math.log(y_proba) / temperature
        char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
        return self._tokenizer.sequences_to_texts(char_id.numpy())[0]

    def complete_text(self, text, outfile, n_chars=50, temperature=1):
        """ Completes a given pice of text using the appropriate temperature and outputs n_chars characters to outfile

            text -- textual data (password)
            outfile -- output file for generated passwords
            n_chars -- number of characters to generate
            temperature -- used to control the randomness of predictions by scaling the logits before applying softmax
        """
        for _ in range(n_chars):
            text += self.next_char(text, temperature)
        with open(outfile, "w") as f:
            f.write(text)
        return text

    def get_accuracy(self, output, test_dataset):
        """ Outputs the accuracy for a given model (i.e. # of matched passwords) for a given test dataset

            output -- output file from a complete_text() call
            test_dataset -- the testing dataset to compare the generated passwords to
        """
        output_file = open(output, "r")
        test_file = open(test_dataset, "r")

        output_lines = output_file.readlines()
        test_lines = test_file.readlines()

        matches = list(set(output_lines).intersection(test_lines))
        num_matches = len(matches)

        print("Found {}/{} matches.".format(num_matches, len(test_lines)))
        for match in matches:
            print(match)

        