from extractTextFeatures import QuestionFeatures
from constants import Constants
from vqa_model import VQA
import pickle
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import image
from utils import read_answers, get_n_frequent_answers
from keras.models import load_model
from modelpredict_pg import prediction

def get_images_for_batch(root_dir, image_ids, prefix="COCO_train2014_000000"):
    images = np.ndarray(shape=(len(image_ids), 224, 224, 3))
    for idx, img_id in enumerate(image_ids):
        img_fname = prefix + "0"*(6-len(str(img_id))) + str(img_id) + ".jpg"
        img_path = os.path.join(root_dir, img_fname)
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        images[idx,:,:,:] = img
    return images

def run_exp(last_epoch=-1, fname=None, filename_loss=None, filename_acc=None):
    start_from_epoch = last_epoch + 1
    epochs = 150
    batch_size = 64

    # load text features and tokenizer
    textObj = QuestionFeatures(initialize=False)
    textObj.load_tokenizer(os.path.join(Constants.DIRECTORIES["root"], "tokenizer.pkl"))
    word_embeddings = textObj.load_ndarray(os.path.join(Constants.DIRECTORIES["root"], "embeddings.npy") )

    train_questions = textObj.get_questions(os.path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["training_questions"])) 
    train_answers = read_answers(os.path.join(Constants.DIRECTORIES["root"], "v2_mscoco_train2014_annotations.txt"))

    top_train_answers, top_answers = get_n_frequent_answers(train_answers)  # has the questions with answers in the top 1000 only
    top_question_ids = top_train_answers["question_id"]
    print(f'ans={len(train_answers["multiple_choice_answer"])}')
    num_traning_ex = len(top_train_answers["multiple_choice_answer"])
    num_batches = int(num_traning_ex / batch_size) + 1
    print(f'len_top_100={len(top_answers)}, num_train_ex={len(top_train_answers["multiple_choice_answer"])}, num_batches={num_batches}')
    loadedmodel=""
    if fname==None:
        model = VQA().get_model_cnn_attention(embedding_matrix=word_embeddings, vocab_size=textObj.get_vocab_size())
        loss_np = np.ndarray(shape=(epochs,num_batches,2))
        acc_np = np.ndarray(shape=(epochs,2))
    else:
        model = load_model(fname)
        loadedmodel = 'M'+fname.replace('.h5','').replace('model_','') + '_'
        #replace with last saved metrics
        loss_np = np.load(filename_loss)
        acc_np = np.load(filename_acc)        
        print(f'loadedm {loadedmodel}')
        print("Model loaded")
    print("Starting training of the VQA model ...")

    #data description! ):
    #image_features={"image_id":[], "features":[]}
    #questions={"image_id":[], "question_id":[], "questions":[]}
    #answers = {question_id":[], "multiple_choice_answer":[]} 
    for epoch in range(start_from_epoch, epochs):
        # if epoch%2==0:
        #     time.sleep(300)
        for batch in range(0, num_batches):
            start_index = batch*batch_size
            stop_index = min(num_traning_ex, (batch+1)*batch_size)
            images_in_batch = []
            if stop_index-start_index < batch_size:
                X_text, y = [], np.ndarray(shape=(stop_index-start_index, Constants.NUM_CLASSES))
            else:
                X_text, y = [], np.ndarray(shape=(batch_size, Constants.NUM_CLASSES))
            print(f'Training batch {batch+1} from {start_index}:{stop_index}')
            for i in range(start_index, stop_index):
                q_id = int(top_question_ids[i])
                index_in_questions = train_questions["question_id"].index(q_id)
                img_id = train_questions["image_id"][index_in_questions]
                images_in_batch.append(img_id)
                
                X_text.append(train_questions["questions"][index_in_questions])

                y_i = np.zeros(Constants.NUM_CLASSES)
                a_index = top_train_answers["question_id"].index(str(q_id))
                class_i = top_answers.index(top_train_answers["multiple_choice_answer"][a_index])
                y_i[class_i] = 1
                y[i-start_index] = y_i
            X_image = get_images_for_batch(root_dir=os.path.join(Constants.DIRECTORIES["root"], "train2014"), image_ids=images_in_batch)
            X_text=textObj.tokenize(X_text)
            loss = model.train_on_batch([X_image, X_text], y)
            print(loss)
            loss_np[epoch][batch][0]=loss[0]
            loss_np[epoch][batch][1]=loss[1]
        print("Completed training for epoch " + str(epoch) + "\n\n")
        save_model_name=loadedmodel + 'attn_model_' + str(epoch) +'.h5'
        model.save(save_model_name)
        save_epoch_name=loadedmodel + 'attn_loss_' + str(epoch) +'_' + str(batch)
        np.save(save_epoch_name, loss_np)
        print("Model saved for epoch: " + str(epoch))
        # print("Validation...")
        # acc = predict_attn(model=model)
        # save_epoch_name=loadedmodel + 'attn_VALACC_' + str(epoch) +'_' + str(batch)
        # acc_np[epoch][0]=acc[0]
        # acc_np[epoch][1]=acc[1]
        # np.save(save_epoch_name, acc_np)
    model.save("final_attn_model.h5")
    np.save('final_attn_loss',loss_np)
    np.save('final_attn_val',acc_np)
            

if __name__ == "__main__":
    run_exp()

