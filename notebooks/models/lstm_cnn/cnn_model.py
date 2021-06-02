import numpy as np
from keras.layers import Dense, Dropout, Input, Flatten, Concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
 
from keras.models import Model
from keras.utils import to_categorical
from tensorflow.keras.metrics import Recall


class CNN_Model():
    """
    """
    # training arguments
    batch_size = 64 #16
    epoch_no = 10
    hidden_dims = 50

    def __init__(self, *args, **kwargs):
        self._model = self._get_cnn_model(*args, **kwargs)

    def train(self, *args, **kwargs):
        self._train_model(*args, **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self._model.predict(np.array(feature))

    def _get_cnn_model(self, dropout, optimizer,
                         max_sequence_length, embedding_layer, kernel_size=8, num_filters=32):
        '''
           https://github.com/karimkhanp/CNN-Sentiment-Analysis-Keras/blob/master/sentiment_cnn.py
        '''
        
        sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        x = embedded_sequences
        x = Dropout(dropout[0])(x)

        # Convolutional block
        conv_blocks = []
        for sz in kernel_size:
            conv = Conv1D(filters=num_filters,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(x)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        x = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        x = Dropout(dropout[1])(x)
        x = Dense(self.hidden_dims, activation="relu")(x)
        output = Dense(3, activation="sigmoid")(x)


        model_cnn = Model(inputs=sequence_input, outputs=output)


        # compile network
        model_cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[Recall()])
        
        model_cnn.summary()
        return model_cnn
    
    def _train_model(self, *args):

        if len(args) > 1:
            x_train = np.array(args[0])
            y_train = np.array(args[1])
            x_val = np.array(args[2])
            y_val = np.array(args[3])

            #weights = {0: 1 / y_train[:, 0].mean(), 1: 1 / y_train[:, 1].mean()}
            self._model.fit(
                x_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.epoch_no,
                validation_data=(x_val, y_val),
                shuffle=True,
                #class_weight=weights,
                verbose=0)
        else:
            dataset = args[0]
            x_train, y_train_ = dataset.format_sklearn()

            if y_train_.ndim == 1:
                y_train = to_categorical(np.asarray(y_train_))
            else:
                y_train = y_train_

            #weights = {0: 1 / y_train[:, 0].mean(), 1: 1 / y_train[:, 1].mean()}

            self._model.fit(
                x_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.epoch_no,
                shuffle=True,
                #class_weight=weights,
                verbose=0)
