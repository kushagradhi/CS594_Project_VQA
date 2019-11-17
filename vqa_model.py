from keras.models import Sequential
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import LSTM, Dense, Embedding
from keras.layers.merge import concatenate

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
        model_language.add(Embedding(vocab_size, embedding_matrix.shape[1], input_length=question_len,
                                        weights=[embedding_matrix], trainable=False))
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
