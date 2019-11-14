import pandas as pd
import nltk
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


class QuestionFeatures:
    def __init__(self, embedding_size=300):
        self.glove_embeddings = self.get_glove_embeddings(embedding_size)


    ## Read the GloVe embeddings
    def get_glove_embeddings(self, embedding_size):
        embeddings={}
        embedding_file = {50:"glove_50", 100:"glove_100", 300:"glove_300"}
        with open(embedding_file[embedding_size], encoding='UTF-8') as f:
            for line in f:
                data = line.split()
                word = data[0]
                coefficients = np.asarray(data[1:], dtype='float32')
                embeddings[word] = coefficients
        print(f'Reading GloVe {embedding_size}d, found {len(embeddings)} word vectors.')
        return embeddings

    def tokenize_corpus(self, text, sequence_len=50):    
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text)

        sequences = tokenizer.texts_to_sequences(text)
        padded_sequences = pad_sequences(sequences, maxlen=sequence_len, padding='post')
        
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        return padded_sequences

    def get_embedding_matrix(self, word_index, embeddings, embedding_dim):
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    ## LSTM
    def model_builder_LSTM(self, vocab_size, embedding_dim, embedding_matrix, learning_rate, hidden_units, max_length, trainable):
        input_initial = layers.Input(shape=(max_length,), dtype='int32')
        output_embedding = layers.Embedding(vocab_size, embedding_matrix.shape[1], input_length=max_length,
                                        weights=[embedding_matrix])(input_initial)
        output_lstm = layers.LSTM(units=hidden_units, return_sequences=False, unroll=True)(output_embedding)

        model = keras.Model(inputs=input_initial, outputs=output_lstm)
        loss = tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model