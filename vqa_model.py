from keras.models import Sequential, Model
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import LSTM, Dense, Embedding, Input, Concatenate, Flatten
from keras.layers.merge import concatenate
from constants import Constants
import tensorflow as tf

class VQA():

    def get_model(self, embedding_matrix, vocab_size, question_len=15, img_feat=2048, embed_dim=300, ):
        number_of_hidden_units_LSTM = 512
        number_of_dense_layers      = 3
        number_of_hidden_units      = 1024
        activation_function         = 'tanh'
        dropout_pct                 = 0.5

        # Image model - loading image features and reshaping
        model_image = Sequential()
        model_image.add(Reshape((img_feat,), input_shape=(img_feat,)))

        # Language Model - 3 LSTMs
        model_language = Sequential()
        # model_language.add(Embedding(vocab_size, embedding_matrix.shape[1], input_length=question_len,
                                        # weights=[embedding_matrix], trainable=False))
        model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True, input_shape=(question_len, embed_dim)))
        model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True))
        model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=False))

        # combined model
        model = Sequential()
    
        model.add(concatenate([model_language, model_image]))

        for _ in range(number_of_dense_layers):
            model.add(Dense(number_of_hidden_units, kernel_initializer='uniform'))
            model.add(Activation(activation_function))
            model.add(Dropout(dropout_pct))

        model.add(Dense(1000))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        
        return model

    def get_model_functional(self, embedding_matrix, vocab_size, hidden_units=16, question_len=15, img_feat=2048, embed_dim=300):
        number_of_hidden_units_LSTM = 512
        number_of_dense_layers      = 3
        number_of_hidden_units      = 1024
        activation_function         = 'tanh'
        dropout_pct                 = 0.5

        # Image model - loading image features and reshaping
        input_image = Input(shape=(img_feat,))

        # Language Model - 3 LSTMs
        input_lang = Input(shape=(question_len,))
        output_embedding = Embedding(vocab_size, embedding_matrix.shape[1], input_length=question_len,
                                        weights=[embedding_matrix], trainable=False)(input_lang)
        #output_lstm_1 = LSTM(units=hidden_units, return_sequences=True, unroll=True)(output_embedding)
        #output_lstm_2 = LSTM(units=hidden_units, return_sequences=True, unroll=True)(output_lstm_1)
        output_lstm_3 = LSTM(units=hidden_units, return_sequences=False, unroll=True)(output_embedding)


        # combined model
        merged = Concatenate()([input_image, output_lstm_3])

        dense_layers, activation_layers, dropout_layers = [], [], []

        for i in range(number_of_dense_layers):
            if i is 0:
                dense_layers.append(Dense(number_of_hidden_units, kernel_initializer='uniform')(merged))
            else:
                dense_layers.append(Dense(number_of_hidden_units, kernel_initializer='uniform')(dropout_layers[-1]))
            activation_layers.append(Activation(activation_function)(dense_layers[i]))
            dropout_layers.append(Dropout(dropout_pct)(activation_layers[i]))
        
        final_dense = Dense(Constants.NUM_CLASSES, activation="softmax")(dropout_layers[-1])
        model = Model(inputs=[input_image, input_lang], outputs=final_dense)
        model.compile(loss="categorical_crossentropy", optimizer="rmsprop" , metrics=['accuracy'] )#   tf.keras.losses.CategoricalCrossentropy()
        # model.summary()     
        return model
