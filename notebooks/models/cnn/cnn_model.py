import numpy as np


from keras.layers import Dense, Dropout, Input, Flatten, Concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Model
from tensorflow.keras.models import load_model

class CNN_Model():
    def __init__(self, *args, **kwargs):
        if len(kwargs)>0:
            self._model = self._get_cnn_model(*args, **kwargs)

    def train(self, *args, **kwargs):
        self._train_model(*args, **kwargs)

    def save(self, model_fp):
        self._model.save(model_fp)

    
    def predict(self, feature, *args, **kwargs):
        return self._model.predict(np.array(feature))

    def _get_cnn_model(self, dropout, optimizer,
                         max_sequence_length, embedding_layer, kernel_size=8, num_filters=32, hidden_dims=50):
        '''
           inspired from https://github.com/karimkhanp/CNN-Sentiment-Analysis-Keras/blob/master/sentiment_cnn.py
        '''
        
        sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        x = embedded_sequences
        
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
        x = Dense(hidden_dims, activation="relu")(x)
        output = Dense(3, activation="sigmoid")(x)

        
        model_cnn = Model(inputs=sequence_input, outputs=output)

        # compile network
        model_cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model_cnn.summary()
        return model_cnn
    
    def _train_model(self, *args):

        x_train = np.array(args[0])
        y_train = np.array(args[1])
        x_val = np.array(args[2])
        y_val = np.array(args[3])
        batch_size = args[4]
        epoch_no = args[5]

        #weights = {0: 1 / y_train[:, 0].mean(), 1: 1 / y_train[:, 1].mean()}
        self._model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epoch_no,
            validation_data=(x_val, y_val),
            shuffle=True,
            #class_weight=weights,
            verbose=0)

    def _load_model(self, model_fp):
        """Load the model from the given file path
            Parameters
            ----------
            model_fp: str
                    file path of the trained model
        """
        self._model = load_model(model_fp)
        
    def predict_model(self, model_fp, features):
        """Load a trained model and make a prediction
            Parameters
            ----------
            model_fp: str
                file path of the trained model
            features: dataframe
                X_test 
                
        """
        self._load_model(model_fp)
        preds = self.predict(features)
        return preds

