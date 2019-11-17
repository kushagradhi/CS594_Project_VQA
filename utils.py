from collections import defaultdict
from constants import Constants
import numpy as np

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
    # print(answers_counts)
    # print(top_answers)
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


