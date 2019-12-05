#from runExperiments import load_saved_model
from extractTextFeatures import QuestionFeatures
from image_features_final import image_features
from constants import Constants
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import image
from utils import read_answers, get_n_frequent_answers
import pickle, os, time
import numpy as np
import tensorflow as tf
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

def get_resnet_model():
    model=ResNet50(weights='imagenet')
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    # model.summary()
    return model

# reads 224*224*3 for now, needs to change to handle other resoutions
def get_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    return img

def read_labels(labels_file):
    labels = []
    with open(labels_file, 'r') as f:
        for line in f.readlines():
            label = line.split(":")[1]
            labels.append(label.replace("'", "").replace(" ",""))
    return labels

def run_demo(model_file, img_file, labels_file):
    tf.logging.set_verbosity(tf.logging.FATAL)

    # read labels
    labels = read_labels(labels_file)
    # print(labels) #KB            
 
    # load model
    vqa_model = load_model(model_file)
    print("VQA Model loaded..\n")

    # load text features and tokenizer
    textObj = QuestionFeatures(initialize=False)
    textObj.load_tokenizer(os.path.join(Constants.DIRECTORIES["root"], "tokenizer.pkl"))

    # image model and feature pre-processing
    img_model = get_resnet_model()
    img = image.load_img(img_file, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img=resnet_preprocess_input(img)

    img_features = np.zeros((1, 2048), dtype=np.float32)
    img_features = img_model.predict(img)
    print(f'Image @{img_file} loaded...')

    while True:
        print("\n\nNext Question:")
        curr_question = [str(input())]
        if curr_question[0].lower() == 'q':
            print(f'Thank you for trying the VQA demo, bye!')
            break

        # convert question to token
        question_features = textObj.tokenize(curr_question)
        prediction = vqa_model.predict([img_features, question_features])
        print(f'\nAnswer:{np.argmax(prediction)} {labels[np.argmax(prediction)]}')


def generate_results_test_set(model_file, img_file_test, labels_file, batch_size=1000):
    tf.logging.set_verbosity(tf.logging.FATAL)

    # read labels
    labels = read_labels(labels_file)            
 
    # load model
    vqa_model = load_model(model_file)
    print("VQA Model loaded..\n")

    # load text features and tokenizer
    textObj = QuestionFeatures(initialize=False)
    textObj.load_tokenizer(os.path.join(Constants.DIRECTORIES["root"], "tokenizer.pkl"))
    test_questions = textObj.get_questions(os.path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["testing_questions"]))  
                                                                    # dictionary with image_id, question_id, question (each is a list)

    # load image features
    with open(img_file_test, 'rb') as f:
        image_features = pickle.load(f)

    num_questions = len(test_questions["question_id"])

    #build the input
    X_image, X_text = [], []
    for i in range(num_questions):
        # q_id = test_questions["question_id"][i]
        img_id = test_questions["image_id"][i]
        
        index_in_featureList = image_features[0][img_id]
        X_image.append(image_features[1][index_in_featureList])

        X_text.append(test_questions["questions"][i])
    
    X_text = textObj.tokenize(X_text)
    predictions = vqa_model.predict([X_image, X_text], batch_size=1000, verbose=1)

    results = []
    for i in range(num_questions):
        result = {}
        result["question_id"] = test_questions["question_id"][i]
        result["answer"] = labels[np.argmax(predictions[i])][:-1]
        results.append(result)
        
    with open("results_2lstm_" + str(time.time()) + ".json", 'w+') as f:
        json.dump(results, f)
    

if __name__ == "__main__":
    # run_demo("M0_model_13.h5", "D:\\CS\\DLNLP_Project\\data\\test2015\\COCO_test2015_000000000128.jpg", "data/labelsTop1000.txt") #elephant!
    # run_demo("M0_model_13.h5", "D:\\CS\\DLNLP_Project\\data\\test2015\\COCO_test2015_000000000057.jpg", "data/labelsTop1000.txt")

    generate_results_test_set(model_file="D:\\CS\\DLNLP_Project\\data\\model_results\\2LSTM\\MMMM0_14_19_23_model_26.h5", 
                            img_file_test="D:\\CS\\DLNLP_Project\\data\\img_features\\imgfeature_test20151000_final.pkl",   # image features saved by image_features_final.py
                            labels_file="data/labelsTop1000.txt")


    


