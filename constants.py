#from os import path


class Constants:
    '''File locations - Root data directory contains the following directories and files:
    1> glove.6B - directory containing all the glove embeddings
    2> train2014 - directory containing COCO Training images
    3> val2014 - directory containing COCO Validation images
    4> test2015 - directory containing COCO Testing images
    5> 7 json files obtained by unzipping 7 non-image zip files in the Balanced Real Images section 

    '''
    DIRECTORIES = {
        "root": "D:\\CS\\DLNLP_Project\\data",
        
        ## Word Embeddings - Glove
        "glove_300": "glove.6B\\glove.6B.300d.txt",
        "glove_100": "glove.6B\\glove.6B.100d.txt",
        "glove_50": "glove.6B\\glove.6B.50d.txt",

        ## VQA Annotations
        "training_annotations": "v2_mscoco_train2014_annotations.json",
        "validation_annotations": "v2_mscoco_val2014_annotations.json",

        ## VQA Input Questions
        "training_questions": "v2_OpenEnded_mscoco_train2014_questions.json",
        "validation_questions": "v2_OpenEnded_mscoco_val2014_questions.json",
        "testing_questions": "v2_OpenEnded_mscoco_test2015_questions.json",
        "testing_questions_dev": "v2_OpenEnded_mscoco_test-dev2015_questions.json",
        
        ## VQA Input Images
        "training_images": "train2014",
        "validation_images": "val2014",
        "testing_images": "test2015",

        ## Complementary Pairs List
        "training_complimentary_pairs": "v2_mscoco_train2014_complementary_pairs.json",
        "validation_complimentary_pairs": "v2_mscoco_val2014_complementary_pairs.json"
    }

    ## CONSTANTS
    GLOVE_EMBEDDING_DIM = 300
    
