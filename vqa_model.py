from keras.models import Sequential, Model
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import LSTM, Dense, Embedding, Input, Concatenate, Flatten, Lambda, Conv1D, multiply
from keras.layers.merge import concatenate
from constants import Constants
import tensorflow as tf
import numpy as np

class VQA():

    def get_model(self, embedding_matrix, vocab_size, question_len=15, img_feat=2048, embed_dim=300):
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

    def get_sketch_matrix(self, h, s,v,n=2048):
        y=np.zeros(n)
        for i in range(n):
            y[h[i]]+=s[i]*v[i]
        return y
             

    def get_mcb_layer(self, v1, v2, h_s, d=128, n1=2048, n2=2048):
        #sketch_v1 = tf.transpose(tf.sparse_tensor_dense_matmul(self.get_sketch_matrix(h_s[0], h_s[1]), v1, adjoint_a=True, adjoint_b=True))
        #sketch_v2 = tf.transpose(tf.sparse_tensor_dense_matmul(self.get_sketch_matrix(h_s[2], h_s[3]), v2, adjoint_a=True, adjoint_b=True))


        sketch_v1 = self.get_sketch_matrix(h_s[0], h_s[1],v1)
        sketch_v2 = self.get_sketch_matrix(h_s[2], h_s[3],v2)

        fft_1, fft_2 = tf.fft(sketch_v1), tf.fft(sketch_v2)
        fft_product = tf.multiply(fft_1, fft_2)
        inv_fft = tf.ifft(fft_product)
        sgn_sqrt = tf.sign(inv_fft) * tf.sqrt(tf.abs(inv_fft))
        l2_norm = tf.keras.backend.l2_normalize(sgn_sqrt)
        return l2_norm 

    def set_hs(self,d=128,n=2048):
        s1=np.random.choice([-1,1],size=n)
        h1=np.random.randint(low=0,high=d,size=n)
        s2=np.random.choice([-1,1],size=n)
        h2=np.random.randint(low=0,high=d,size=n)
        print(s1)
        print(h1)
        return [h1,s1,h2,s2]


    def get_model_attention(self, embedding_matrix, vocab_size, h_s_img_text_1, h_s_img_text_2, hidden_units_LSTM=1024, question_len=15, img_feat=2048, embed_dim=300, ):

        input_image = Input(shape=(img_feat,))

        # Language Model - 2 LSTMs
        input_lang = Input(shape=(question_len,))
        output_embedding = Embedding(vocab_size, embedding_matrix.shape[1], input_length=question_len,
                                        weights=[embedding_matrix], trainable=False)(input_lang)
        output_lstm_1 = LSTM(units=hidden_units_LSTM, return_sequences=True, unroll=True)(output_embedding)
        output_lstm_2 = LSTM(units=hidden_units_LSTM, return_sequences=False, unroll=True)(output_lstm_1)
        concatenated_lang_features = Concatenate()([output_lstm_1[:,-1,:], output_lstm_2])

        mcb_1 = self.get_mcb_layer(v1=input_image, v2=concatenated_lang_features, h_s=h_s_img_text_1, 
                                d=128, n1=2048, n2=2048)
        conv_1 = Conv1D(filters=1, kernel_size=32, activation='relu', padding='same')(mcb_1)
        conv_2 = Conv1D(filters=1, kernel_size=32, activation='softmax', padding='same')(conv_1)
        weighted_sum = multiply([conv_2, input_image])

        mcb_2 = self.get_mcb_layer(v1=weighted_sum, v2=concatenated_lang_features, h_s=h_s_img_text_2, d=128,
                                    n1=2048, n2=2048)
        final_fc = Dense(Constants.NUM_CLASSES, activation="softmax")(mcb_2)
        model = Model(inputs=[input_image, input_lang], outputs=final_fc)
        model.compile(loss="categorical_crossentropy", optimizer="rmsprop" , metrics=['accuracy'] )   
        # model.summary()     
        return model           









