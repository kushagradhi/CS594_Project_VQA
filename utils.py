from collections import defaultdict

def get_n_frequent_answers(answers, to_keep=1000):
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


    

