import json

def dumpQuestionsToText(filename):
    with open (filename, 'r') as fr:
        json_data = json.load(fr)
    
    output_filename = filename[:-4] + "txt"
    with open(output_filename, 'w') as fwriter:
        for i in range(len(json_data["questions"])): 
            fwriter.write(str(json_data["questions"][i]["image_id"]) + "\t" + str(json_data["questions"][i]["question_id"]) + "\t" 
                            + json_data["questions"][i]["question"] + "\n")

dumpQuestionsToText("D:\\CS\\DLNLP_Project\\data\\v2_OpenEnded_mscoco_train2014_questions.json")
dumpQuestionsToText("D:\\CS\\DLNLP_Project\\data\\v2_OpenEnded_mscoco_val2014_questions.json")
dumpQuestionsToText("D:\\CS\\DLNLP_Project\\data\\v2_OpenEnded_mscoco_test-dev2015_questions.json")
dumpQuestionsToText("D:\\CS\\DLNLP_Project\\data\\v2_OpenEnded_mscoco_test2015_questions.json")

