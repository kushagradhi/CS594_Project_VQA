import numpy as np, os
import pickle 


class check_feats:
    def __init__(self):
       data_directory=os.path.join('Datasets','train2014') 

    def check_img_feats(self,fname):
        name,feat=pickle.load(open(fname,'rb'))
        print("Filename: " + fname)
        print("Number of images loaded: " + str(len(name)))
        print('-------------------------------------------')
        print("Feature numpy shape: ", feat.shape)
        print('-------------------------------------------')
        print("Feature of last image:")
        print(feat[len(name)-1])
        print("Length of features of last image:")
        print(len(feat[len(name)-1]))
        print('-------------------------------------------')
        print("First image feature:")
        print(feat[0])
        print('-------------------------------------------')
        print("Feature of the LAST POSSIBLE IMAGE:")
        print(feat[-1])

if __name__=='__main__':
    obj=check_feats()
    ####--- Enter name of the file
    # obj.check_img_feats('imgfeature1000_19.pkl')
    obj.check_img_feats('D:\\CS\\DLNLP_Project\\data\\imgfeature_test20151000_81.pkl')
    