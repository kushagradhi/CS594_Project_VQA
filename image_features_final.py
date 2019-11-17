from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
import numpy as np,os
import keras
import tensorflow as tf
import pickle 
import time


class image_features:
    def __init__(self):
       #data_directory=os.path.join('Datasets','train2014') 
        pass

    def get_data(self, filename):
        data_directory=os.path.join('Datasets',filename)
        filelist=[]
        i=0
        for dir_, _, files in os.walk(data_directory):
            for fileName in files:
                filelist.append(fileName)
                i+=1
        return filelist, data_directory 

    def get_model(self):
        model = VGG16(weights='imagenet')
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        model.summary()
        return model
    
    def get_resnet_model(self):
        model=ResNet50(weights='imagenet')
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        model.summary()
        return model

    def get_feats(self,mname,batch_size,filename):
        fname,data_directory=self.get_data(filename)
        n=len(fname)
        print(n)
        
        if mname=='V':
            mod=self.get_model()
            features = np.zeros((n, 4096), dtype=np.float32)
        else:
            mod=self.get_resnet_model()
            features = np.zeros((n, 2048), dtype=np.float32)
        vgg16_feature=[]
        imgname={}
        b=int(n/batch_size)+1
        bn=0
        for i in range(0,b):
            batch = np.zeros((batch_size, 224, 224, 3))
            bn=0
            for k in range(i*batch_size,min((i+1)*batch_size,n)):
                img_path=os.path.join(data_directory, fname[k])
                f=fname[k].replace('.jpg','')
                f=int(f[-7:])
                imgname[f]=k
                img = image.load_img(img_path, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                if mname=='V':
                    img = preprocess_input(img)
                else:
                    img=resnet_preprocess_input(img)
                batch[bn, :, :, :] = img
                bn+=1
                if k%batch_size==0 and k>1:
                    print(str(k) + " Done")
            start = time.time()        
            print("Predicting now at " + str(start))
            if n < (i+1)*batch_size:
                batch=batch[0:bn]
            features[i * batch_size : min((i + 1) * batch_size,n), :] = mod.predict(batch)
            end=round(time.time() - start,1)
            print("Prediction done in " + str(end))
            print(features[i * batch_size : min((i + 1) * batch_size,n), :])
            s='imgfeature' + str(batch_size) + '_' + str(i) +'.pkl'
            pickle.dump([imgname,features],open(s,'wb'))
                
        #print(features)
        s='imgfeature_' + str(data_directory) + '_' + str(batch_size) + '_final.pkl'
        pickle.dump([imgname,features],open(s,'wb'))
        return imgname, features

if __name__=='__main__':
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)

    '''Use the below to import features.. Call function get_feats with parameters modelname (rather, initial) and batch size
    Filepath for training images is set as Datasets/train2014
    Saved pkl file has dictionary, followed by features.. Key for the dictionary is the name of the image file and the value is the position at which its feature will be found
    '''
    obj=image_features()
    modelname='R'  # R for Resnet, V for VGG16
    batch_size=1000 # Runs prediction on 'batch_size' number of features at a time.
    filename='train2014' #--- Validation set: val2014, training: train2014, test: train2014
    imgname, features=obj.get_feats(modelname,batch_size,filename)  # Creates files with features --- final one: imgfeature_batchsize_final.pkl
    