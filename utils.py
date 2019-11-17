from collections import defaultdict
from constants import Constants

def get_n_frequent_answers(answers, to_keep=Constants.NUM_CLASSES):
    '''
    truncates the answers dict to retain only the top to_keep answers, 
    returns rebuilt dictionary with question_id, multiple_choice_answer (both lists)
    '''
    answers_counts = defaultdict(int)
    for answer in answers["multiple_choice_answer"]:
        answers_counts[answer] += 1
    top_answers = [a for a, v in sorted(answers_counts.items(), key=lambda x: x[1])][:to_keep]
    new_answers={"question_id":[], "multiple_choice_answer":[]}
    for i in range(answers["question_id"]):
        if answers["multiple_choice_answer"][i] in top_answers:
            new_answers["question_id"].append(answers["question_id"][i])
            new_answers["multiple_choice_answer"].append(answers["multiple_choice_answer"][i])
    return new_answers

def prepare_Ximage_and_y(images, questions, answers, start, stop):
    #images={"image_id":[], "features":[]}
    #questions={"image_id":[], "question_id":[], "question":[]}
    #answers = {question_id":[], "multiple_choice_answer":[]}
    
    X_image_features = []
    y = []
    for i in range(start, stop):
        image_index = images["image_id"]
        X_image_features.append(images["features"][image_index])   
             
        y_encoding = [0] * Constants.NUM_CLASSES
        y_encoding[ answers["multiple_choice_answer"].index(answers["multiple_choice_answer"][i]) + 1 ] = 1
        y.append(y_encoding)

    return X_image_features, y


