#from runExperiments import load_saved_model
from extractTextFeatures import QuestionFeatures
from constants import Constants
from keras.preprocessing.text import Tokenizer
from utils import read_answers, get_n_frequent_answers
import pickle, os
import numpy as np


def load_saved_model(fname):
    m=load_model(fname)
    return m

def prediction(fname=None,model=None):
    #image_file_training = "D:\\CS\\DLNLP_Project\\data\\img_features\\imgfeature_val20141000_final.pkl"
    image_file_test = "C:\SIM\MS CS\CS594_Deep_learning_in_NLP\Project\Run\data\img_features\\imgfeature_val20141000_final.pkl"
    #image_file_test = "drive/My Drive/Colab Notebooks/data/img_features/imgfeature_val20141000_final.pkl"
    with open(image_file_test, 'rb') as f:
        image_features = pickle.load(f)
    print("Image features loaded")
    # load text features and tokenizer
    textObj = QuestionFeatures(initialize=False)
    textObj.load_tokenizer(os.path.join(Constants.DIRECTORIES["root"], "tokenizer.pkl"))
    word_embeddings = textObj.load_ndarray(os.path.join(Constants.DIRECTORIES["root"], "embeddings.npy") )

    test_questions = textObj.get_questions(os.path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["validation_questions"])) 
    test_answers = read_answers(os.path.join(Constants.DIRECTORIES["root"], "v2_mscoco_val2014_annotations.txt"))

    #train_questions = textObj.get_questions(os.path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["training_questions"])) 
    train_answers = read_answers(os.path.join(Constants.DIRECTORIES["root"], "v2_mscoco_train2014_annotations.txt"))
    top_train_answers, top_answers = get_n_frequent_answers(train_answers)  # has the questions with answers in the top 1000 only
    top_question_ids = top_train_answers["question_id"]
    print("All files loaded")
    
    '''
    q_idlist=[]
    for q_id in test_questions["question_id"]:
        a_index = test_answers["question_id"].index(str(q_id))
        if  test_answers["multiple_choice_answer"][a_index] in top_answers:
            q_idlist.append(q_id)
            
    
    '''
    if model==None:
        model=load_saved_model(fname)
    i=0
    X_image, X_text , y=[],[],np.ndarray(shape=(len(test_questions["question_id"]), Constants.NUM_CLASSES))
    #for q_id in test_questions["question_id"]:
    for q_id in test_questions["question_id"]:
        #q_id = q["question_id"]
        
        a_index = test_answers["question_id"].index(str(q_id))
        if  test_answers["multiple_choice_answer"][a_index] in top_answers:

            img_id = test_questions["image_id"][test_questions["question_id"].index(q_id)]
            image_feat_index = image_features[0][img_id]
            X_image.append(image_features[1][image_feat_index])
                
            q_index = test_questions["question_id"].index(q_id)
            X_text.append(test_questions["questions"][q_index])

            y_i = np.zeros(Constants.NUM_CLASSES)
            a_index = test_answers["question_id"].index(str(q_id))
            class_i = top_answers.index(test_answers["multiple_choice_answer"][a_index])
            y_i[class_i] = 1
            y[i] = y_i
            i+=1
            #if i==2:
            #    break
                # y.append(y_i)
    print('Number of questions: ' + str(i))
    y=y[0:i]        
    X_text=textObj.tokenize(X_text)
    print("Running prediction:")
    sol = model.predict([X_image, X_text])
    give_stats(sol,y)
    
def give_stats(sol,y):
    ct=0.0
    s=sol.shape
    s=s[0]
    for i in range(0,s):
        a_index=np.where(sol[i] == np.amax(sol[i]))
        #print(sol[i][a_index[0]])
        if y[i][a_index[0]]==1:
            ct+=1
    print("Accuracy : " + str(ct/s) )

if __name__=='__main__':
    # Give either fname or model
    prediction()