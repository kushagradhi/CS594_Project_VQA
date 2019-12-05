from collections import defaultdict
from constants import Constants
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def get_n_frequent_answers(answers, to_keep=Constants.NUM_CLASSES):
    '''
    truncates the answers dict to retain only the top to_keep answers, 
    returns rebuilt dictionary with question_id, multiple_choice_answer (both lists)
    '''
    answers_counts = defaultdict(int)
    answers_counts = defaultdict(int)
    for answer in answers["multiple_choice_answer"]:
        answers_counts[answer] += 1
    top_answers = [a for a, v in sorted(answers_counts.items(), key=lambda x: x[1], reverse=True)][:to_keep]
    # top_answer_counts = {w:c for w, c in sorted(answers_counts.items(), key=lambda x: x[1], reverse=True)[:to_keep] }
    # top=""
    # for w,c in top_answer_counts.items():
    #     for i in range(c):
    #         top += w + "\n"
    # with open("top_answers.txt", 'w') as f:
    #     f.write(top)
    # print(top_answer_counts)
    new_answers={"question_id":[], "multiple_choice_answer":[]}
    for i in range(len(answers["multiple_choice_answer"])):
        if answers["multiple_choice_answer"][i] in top_answers:
            new_answers["question_id"].append(answers["question_id"][i])
            new_answers["multiple_choice_answer"].append(answers["multiple_choice_answer"][i])

    print(f'nfq ta={len(top_answers)} top_old={len(answers["multiple_choice_answer"])} top_new={len(new_answers["multiple_choice_answer"])}')
    return new_answers, top_answers

# def prepare_Ximage_and_y(images, questions, answers, start, stop):
#     #images={"image_id":[], "features":[]}
#     #questions={"image_id":[], "question_id":[], "question":[]}
#     #answers = {question_id":[], "multiple_choice_answer":[]}
    
#     X_image_features = []
#     y = []
#     for i in range(start, stop):
#         image_index = images["image_id"]
#         X_image_features.append(images["features"][image_index])   
             
#         y_encoding = np.zeros(Constants.NUM_CLASSES)
#         y_encoding[ answers["multiple_choice_answer"].index(answers["multiple_choice_answer"][i]) + 1 ] = 1
#         y.append(y_encoding)

#     return X_image_features, y

def read_answers(filename):
    answers={"question_id":[], "multiple_choice_answer":[]}
    with open(filename, 'r') as f:
        for line in f.readlines():            
            # answers[line.split()[0]] = line.split()[1]
            answers["question_id"].append(line.split()[0])
            answers["multiple_choice_answer"].append(line.split()[1])

    return answers

def plot_metrics(loss, acc, filename=None):
    if filename is None:
        filename = str(time.time()) + ".png"
    elif str(filename[-4:]) != ".png":
        filename += ".png"
        
    fig, axes = plt.subplots(figsize=(18,6), nrows=1, ncols=2)
    fig.tight_layout()
    axes[0].plot([i+1 for i in range(len(loss[0]))], loss[0])
    axes[0].plot([i+1 for i in range(len(loss[1]))], loss[1])
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].set_title('Loss')
    axes[0].legend(['train', 'val'], loc='upper right')
             
    axes[1].plot([i+1 for i in range(len(acc[0]))], acc[0])
    axes[1].plot([i+1 for i in range(len(acc[1]))], acc[1])
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('acc')
    axes[1].set_title('Accuracy')
    axes[1].legend(['train', 'val'], loc='lower right')
    
    plt.savefig(os.path.join(Constants.DIRECTORIES["root"], filename))
    plt.show()

def draw():
    tr = np.load("MM0_13_loss_14_404.npy")
    val = np.load("MM0_13_VALACC_14_404.npy")

    epochs = 14

    loss, acc = np.ndarray(shape=(2,epochs)), np.ndarray(shape=(2,epochs))
    for i in range(epochs):
        loss[0][i] = tr[i][404][0]
        loss[1][i] = val[i][0]

        acc[0][i] = tr[i][404][1] * 100
        acc[1][i] = val[i][1] * 100
    
    plot_metrics(loss, acc)
