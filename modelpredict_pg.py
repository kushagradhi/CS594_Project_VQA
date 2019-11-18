#from runExperiments import load_saved_model
from extractTextFeatures import QuestionFeatures
from constants import Constants
from keras.preprocessing.text import Tokenizer
from utils import read_answers, get_n_frequent_answers
import pickle, os, time
import numpy as np
import tensorflow as tf
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def load_saved_model(fname):
    m=load_model(fname)
    return m

def prediction(fname=None, model=None, mode='not_r'):    
    start_time = time.time()
    tf.logging.set_verbosity(tf.logging.FATAL)
    batch_size = 1000

    
    '''
    q_idlist=[]
    for q_id in test_questions["question_id"]:
        a_index = test_answers["question_id"].index(str(q_id))
        if  test_answers["multiple_choice_answer"][a_index] in top_answers:
            q_idlist.append(q_id)

    #image_features={"image_id":[], "features":[]}
    #questions={"image_id":[], "question_id":[], "questions":[]}
    #answers = {"question_id":[], "multiple_choice_answer":[]}        
    '''
    if model==None:
        model=load_saved_model(fname)   
        # model.summary()

   
    if mode is 'r':
        print("Reading saved values...")
        X_image = np.load("X_image_val.npy")
        X_text = np.load("X_text_val.npy")
        y = np.load("y_val.npy")
        print(f'{X_image.shape}  {X_text.shape}  {y.shape}')

    else:
        image_file_test = "D:\\CS\\DLNLP_Project\\data\\img_features\\imgfeature_val20141000_final.pkl"
        # image_file_test = "C:\SIM\MS CS\CS594_Deep_learning_in_NLP\Project\Run\data\img_features\\imgfeature_val20141000_final.pkl"
        #image_file_test = "drive/My Drive/Colab Notebooks/data/img_features/imgfeature_val20141000_final.pkl"
        with open(image_file_test, 'rb') as f:
            image_features = pickle.load(f)
        print("Image features loaded")
        # load text features and tokenizer
        textObj = QuestionFeatures(initialize=False)
        textObj.load_tokenizer(os.path.join(Constants.DIRECTORIES["root"], "tokenizer.pkl"))
        # word_embeddings = textObj.load_ndarray(os.path.join(Constants.DIRECTORIES["root"], "embeddings.npy") )

        test_questions = textObj.get_questions(os.path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["validation_questions"])) 
        test_answers = read_answers(os.path.join(Constants.DIRECTORIES["root"], "v2_mscoco_val2014_annotations.txt"))

        #train_questions = textObj.get_questions(os.path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["training_questions"])) 
        train_answers = read_answers(os.path.join(Constants.DIRECTORIES["root"], "v2_mscoco_train2014_annotations.txt"))
        top_train_answers, top_answers = get_n_frequent_answers(train_answers)  # has the questions with answers in the top 1000 only
        top_question_ids = top_train_answers["question_id"]
        print("All files loaded")
        top_test_answers = {"question_id":[], "multiple_choice_answer":[]} 
        top_test_question_ids = top_test_answers["question_id"]

        for i in range(len(test_answers["multiple_choice_answer"])):
            if test_answers["multiple_choice_answer"][i] in top_answers:
                top_test_answers["question_id"].append(test_answers["question_id"][i])
                top_test_answers["multiple_choice_answer"].append(test_answers["multiple_choice_answer"][i])
        print(f'{len(top_train_answers["question_id"])} {len(top_answers)}')
        print(f'{len(test_answers["question_id"])}   {len(top_test_answers["question_id"])}')

        num_test_ex = len(top_test_answers["multiple_choice_answer"])
        num_batches = int(num_test_ex / batch_size) +1
        print(f'len_top_100={len(top_answers)}, num_train_ex={len(top_test_answers["multiple_choice_answer"])}, num_batches={num_batches}')

        X_image, X_text , y = [], [], np.ndarray(shape=(num_test_ex, Constants.NUM_CLASSES))
        for i in range(num_test_ex):
            q_id = int(top_test_question_ids[i])

            index_in_questions = test_questions["question_id"].index(q_id)
            img_id = test_questions["image_id"][index_in_questions]
            image_feat_index = image_features[0][img_id]
            X_image.append(image_features[1][image_feat_index])
            
            X_text.append(test_questions["questions"][index_in_questions])

            y_i = np.zeros(Constants.NUM_CLASSES)
            a_index = top_test_answers["question_id"].index(str(q_id))
            class_i = top_answers.index(top_test_answers["multiple_choice_answer"][a_index])
            y_i[class_i] = 1
            y[i] = y_i
            if i%5000 == 0:
                print(i)
        X_text = textObj.tokenize(X_text)
        np.save("X_image_val", X_image)
        np.save("X_text_val", X_text)
        np.save("y_val", y)
    
    
    print("Running prediction:")
    sol = model.evaluate([X_image, X_text], y)
    #acc=give_stats(sol,y)
    print(sol)
    print(f'Took {round(time.time() - start_time,1)}')
    return sol


def give_stats(sol,y):
    ct=0.0
    s=sol.shape
    s=s[0]
    for i in range(0,s):
        a_index=np.where(sol[i] == np.amax(sol[i]))
        #print(sol[i][a_index[0]])
        if y[i][a_index[0]]==1:
            ct+=1
    acc=ct/s
    print("Accuracy : " + str(acc) )
    return acc

if __name__=='__main__':
    # Give either fname or model
    _ = prediction(fname="C:\\Users\\kusha\\Google Drive\\CSii\\Courses\\s03\\s03-CS594 Deep Learning for NLP\\Project\\code\\model_0.h5", 
                    mode='r')