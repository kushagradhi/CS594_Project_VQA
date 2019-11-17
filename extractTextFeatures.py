import pandas as pd
import nltk
import numpy as np
import json
import pickle
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from constants import Constants


class QuestionFeatures:
    def __init__(self, embedding_size=300, initialize=False):
        if initialize:
            self.glove_embeddings = self.get_glove_embeddings(embedding_size)


    ## Read the GloVe embeddings
    def get_glove_embeddings(self, embedding_size=300):
        embeddings={}
        embedding_file = os.path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["glove_300"]) #{50:"glove_50", 100:"glove_100", 300:"glove_300"}
        with open(embedding_file, encoding='UTF-8') as f:
            for line in f:
                data = line.split()
                word = data[0]
                coefficients = np.asarray(data[1:], dtype='float32')
                embeddings[word] = coefficients
        print(f'Reading GloVe {embedding_size}d, found {len(embeddings)} word vectors.')
        return embeddings

    def intialize_tokenizer(self, text):
        self.tokenizer = Tokenizer(oov_token = True)
        self.tokenizer.fit_on_texts(text)        
        self.word_index = self.tokenizer.word_index
        self.vocab_size = len(self.word_index) + 1
        print('Found %s unique tokens ' % len(self.word_index))

    def save_tokenizer(self, filename):
        # filename = os.path.join(Constants.DIRECTORIES["root"], filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_tokenizer(self, filename):
        with open(filename, 'rb') as f:
            self.tokenizer = pickle.load(f)  
        self.word_index = self.tokenizer.word_index
        self.vocab_size = len(self.word_index) + 1

    def tokenize(self, text, sequence_len=15):
        sequences = self.tokenizer.texts_to_sequences(text)
        padded_sequences = pad_sequences(sequences, maxlen=sequence_len, padding='pre')   
        return padded_sequences

    def get_embedding_matrix(self, glove_embeddings=None, embedding_dim=300):
        if glove_embeddings is None:
            glove_embeddings = self.glove_embeddings
        embedding_matrix = np.zeros((len(self.word_index) + 1, embedding_dim))
        for word, i in self.word_index.items():
            embedding_vector = glove_embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def save_ndarray(self, filename, array):
        # filename = os.path.join(Constants.DIRECTORIES["root"], filename)
        np.save(filename, array)

    def load_ndarray(self, filename):
        return np.load(filename)
        


    # ## LSTM
    # def model_builder_LSTM(self, embedding_matrix, learning_rate=0.001, hidden_units=32, max_length=50, trainable=False, embedding_dim=300):
    #     input_initial = layers.Input(shape=(max_length,), dtype='int32')
    #     output_embedding = layers.Embedding(self.vocab_size, embedding_matrix.shape[1], input_length=max_length,
    #                                     weights=[embedding_matrix])(input_initial)
    #     output_lstm = layers.LSTM(units=hidden_units, return_sequences=False, unroll=True)(output_embedding)

    #     model = keras.Model(inputs=input_initial, outputs=output_lstm)
    #     loss = tf.keras.losses.BinaryCrossentropy()
    #     optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    #     model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    #     return model


    def get_questions(self, json_filename, save=False):
        '''
        returns dictionary with image_id, question_id, question (each is a list)
        '''
        with open (json_filename, 'r') as fr:
            json_data = json.load(fr)

        questions={"image_id":[], "question_id":[], "questions":[]}
        for i in range(len(json_data["questions"])):
            questions["image_id"].append(json_data["questions"][i]["image_id"])
            questions["question_id"].append(json_data["questions"][i]["question_id"])
            questions["questions"].append(json_data["questions"][i]["question"])

        if save is True:
            output_filename = json_filename[:-4] + "txt"
            with open(output_filename, 'w') as fwriter:
                for i in range(len(json_data["questions"])): 
                    fwriter.write(str(json_data["questions"][i]["image_id"]) + "\t" + str(json_data["questions"][i]["question_id"]) + "\t" 
                                    + json_data["questions"][i]["question"] + "\n")
        print("read " + str(len(questions["questions"])) + " questions from file " + json_filename)
        return questions


def getAnnotations(json_filename, save=False):
    '''
    returns dictionary with question_id, most_frequent_answer (each is a list) 
    '''
    with open (json_filename, 'r') as fr:
        json_data = json.load(fr)

    answers = {"question_id":[], "multiple_choice_answer":[]}
    for i in range(len(json_data["annotations"])):
        answers["question_id"].append(json_data["annotations"][i]["question_id"])
        answers["multiple_choice_answer"].append(json_data["annotations"][i]["multiple_choice_answer"])

    if save:
        output_filename = json_filename[:-4] + "txt"
        with open(output_filename, 'w') as fwriter:
            for i in range(len(answers["question_id"])):
                fwriter.write(str(answers["question_id"][i]) + "\t " + answers["multiple_choice_answer"][i] + "\n")
        print("wrote " + str(len(answers["question_id"])) + "answers to file " + output_filename)
    return answers
        


if __name__ == "__main__":
    qObj = QuestionFeatures(initialize=True)
    # glove_embeddings = qObj.get_glove_embeddings()
    
    train_questions = qObj.get_questions(os.path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["training_questions"]))  
    validation_questions = qObj.get_questions(os.path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["validation_questions"]))  
    testing_questions = qObj.get_questions(os.path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["testing_questions"]))  
    testing_questions_dev = qObj.get_questions(os.path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["testing_questions_dev"]))  


    qObj.intialize_tokenizer(train_questions["questions"])

    tokenized_train_questions = qObj.tokenize(train_questions["questions"])
    tokenized_validation_questions = qObj.tokenize(validation_questions["questions"])
    tokenized_testing_questions = qObj.tokenize(testing_questions["questions"])
    tokenized_testing_questions_dev = qObj.tokenize(testing_questions_dev["questions"])

    word_embeddings = qObj.get_embedding_matrix()

    
    qObj.save_tokenizer(os.path.join(Constants.DIRECTORIES["root"], "tokenizer.pkl"))
    qObj.save_ndarray(os.path.join(Constants.DIRECTORIES["root"], "embeddings"), word_embeddings)
    qObj.save_ndarray(os.path.join(Constants.DIRECTORIES["root"], "tokenized_train_questions"), tokenized_train_questions)
    qObj.save_ndarray(os.path.join(Constants.DIRECTORIES["root"], "tokenized_validation_questions"), tokenized_validation_questions)
    qObj.save_ndarray(os.path.join(Constants.DIRECTORIES["root"], "tokenized_testing_questions"), tokenized_testing_questions)
    qObj.save_ndarray(os.path.join(Constants.DIRECTORIES["root"], "tokenized_testing_questions_dev"), tokenized_testing_questions_dev)
