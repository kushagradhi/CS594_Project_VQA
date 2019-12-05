#from runExperiments import load_saved_model
from extractTextFeatures import QuestionFeatures
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

def get_resnet_model():
    model=ResNet50(weights='imagenet')
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    # model.summary()
    return model

# reads 224*224*3 for now, needs to change o handle other resoutions
def get_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)

def run_demo(model_file, img_file, labels_file):
    tf.logging.set_verbosity(tf.logging.FATAL)

    # read labels
    labels = []
    with open(labels_file, 'r') as f:
        for line in f.readlines():
            label = line.split(":")[1]
            labels.append(label.replace("'", "").replace(" ",""))
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


if __name__ == "__main__":
    # run_demo("M0_model_13.h5", "D:\\CS\\DLNLP_Project\\data\\test2015\\COCO_test2015_000000000128.jpg", "data/labelsTop1000.txt") #elephant!
    run_demo("M0_model_13.h5", "D:\\CS\\DLNLP_Project\\data\\test2015\\COCO_test2015_000000000057.jpg", "data/labelsTop1000.txt")



    


