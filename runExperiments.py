from extractTextFeatures import QuestionFeatures
from constants import Constants
from vqa_model import VQA
import os
import pickle
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from utils import read_answers, get_n_frequent_answers


def main():
    tf.logging.set_verbosity(tf.logging.FATAL)
    epochs = 1
    batch_size = 100

    # load image features
    image_file_training = "D:\\CS\\DLNLP_Project\\data\\img_features\\imgfeature_1000_final.pkl"
    with open(image_file_training, 'rb') as f:
        image_features = pickle.load(f)

    # load text features and tokenizer
    textObj = QuestionFeatures(initialize=False)
    textObj.load_tokenizer(os.path.join(Constants.DIRECTORIES["root"], "tokenizer.pkl"))
    word_embeddings = textObj.load_ndarray(os.path.join(Constants.DIRECTORIES["root"], "embeddings.npy") )

    train_questions = textObj.get_questions(os.path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["training_questions"])) 
    train_answers = read_answers(os.path.join(Constants.DIRECTORIES["root"], "v2_mscoco_train2014_annotations.txt"))
    #val_answers = read_answers(os.path.join((Constants.DIRECTORIES["root"], "v2_mscoco_val2014_annotations.txt")))

    top_train_answers, top_answers = get_n_frequent_answers(train_answers)  # has the questions with answers in the top 1000 only
    top_question_ids = top_train_answers["question_id"]
    print(f'ans={len(train_answers["multiple_choice_answer"])}')
    num_traning_ex = len(top_train_answers["multiple_choice_answer"])
    num_batches = int(num_traning_ex / batch_size) +1
    print(f'len_top_100={len(top_answers)}, num_train_ex={len(top_train_answers["multiple_choice_answer"])}, num_batches={num_batches}')
    
    model = VQA().get_model_functional(embedding_matrix=word_embeddings, vocab_size=textObj.get_vocab_size())
    
    #data description! ):
    #image_features={"image_id":[], "features":[]}
    #questions={"image_id":[], "question_id":[], "questions":[]}
    #answers = {question_id":[], "multiple_choice_answer":[]} 
    print("Starting training of the VQA model ...")
    for epoch in range(epochs):        
        for batch in range(0,num_batches):
            start_index = batch*batch_size
            stop_index = min(num_traning_ex, (batch+1)*batch_size)
            X_image, X_text, y = [], [], []
            print(f'Training batch {batch} from {start_index}:{stop_index}')
            for i in range(start_index, stop_index):
                q_id = int(top_question_ids[i])
                img_id = train_questions["image_id"][train_questions["question_id"].index(q_id)]
                image_feat_index = image_features[0][img_id]
                X_image.append(image_features[1][image_feat_index])
                
                q_index = train_questions["question_id"].index(q_id)
                X_text.append(train_questions["questions"][q_index])

                y_i = np.zeros(Constants.NUM_CLASSES)
                a_index = top_train_answers["question_id"].index(str(q_id))
                class_i = top_answers.index(top_train_answers["multiple_choice_answer"][a_index])
                y_i[class_i] = 1
                y.append(y_i)
            
            X_text=textObj.tokenize(X_text)
            loss = model.train_on_batch([X_image, X_text], y)
        print("Completed training for epoch " + str(epoch) + "\n\n")

if __name__ == "__main__":
    main()

        

