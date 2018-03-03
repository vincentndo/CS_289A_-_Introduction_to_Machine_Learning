import os
import numpy as np
from numpy.random import random
import cv2

import copy
import glob

import _pickle as pickle
import IPython



class data_manager(object):
    def __init__(self,classes,image_size,compute_features = None, compute_label = None):

        #Batch Size for training
        self.batch_size = 40
        #Batch size for test, more samples to increase accuracy
        self.val_batch_size = 400

        self.classes = classes
        self.num_class = len(self.classes)
        self.image_size = image_size



        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))



        self.cursor = 0
        self.t_cursor = 0
        self.epoch = 1

        self.recent_batch = []

        if compute_features == None:
            self.compute_feature = self.compute_features_baseline

        else: 
            self.compute_feature = compute_features

        if compute_label == None:
            self.compute_label = self.compute_label_baseline
        else: 
            self.compute_label = compute_label


        self.load_train_set()
        self.load_validation_set()


    def get_train_batch(self):

        '''

        Compute a training batch for the neural network 
        The batch size should be size 40

        '''
        train_batch = []
        for i in range(self.batch_size):
            index = int( random() * len(self.train_data) )
            train_batch.append(self.train_data[index])
        
        return train_batch



    def get_empty_state(self):
        images = np.zeros((self.batch_size, self.image_size,self.image_size,3))
        return images

    def get_empty_label(self):
        labels = np.zeros((self.batch_size, self.num_class))
        return labels

    def get_empty_state_val(self):
        images = np.zeros((self.val_batch_size, self.image_size,self.image_size,3))
        return images

    def get_empty_label_val(self):
        labels = np.zeros((self.val_batch_size, self.num_class))
        return labels



    def get_validation_batch(self):

        '''
        Compute a training batch for the neural network 

        The batch size should be size 400

        '''
        #FILL IN
        val_batch = []
        for i in range(self.val_batch_size):
            index = int( random() * len(self.val_data) )
            val_batch.append(self.val_data[index])

        return val_batch



    def compute_features_baseline(self, image):
        '''
        computes the featurized on the images. In this case this corresponds
        to rescaling and standardizing.
        '''

        image = cv2.resize(image, (self.image_size, self.image_size))
        image = (image / 255.0) * 2.0 - 1.0

        return image


    def compute_label_baseline(self,label):
        '''
        Compute one-hot labels given the class size
        '''

        one_hot = np.zeros(self.num_class)

        idx = self.classes.index(label)

        one_hot[idx] = 1.0

        return one_hot


    def load_set(self,set_name):

        '''
        Given a string which is either 'val' or 'train', the function should load all the
        data into an 

        '''

        data = []
        data_paths = glob.glob(set_name+'/*.png')

        count = 0


        for datum_path in data_paths:

            label_idx = datum_path.find('_')


            label = datum_path[len(set_name)+1:label_idx]

            if self.classes.count(label) > 0:

                img = cv2.imread(datum_path)
                
                label_vec = self.compute_label(label)

                features = self.compute_feature(img)
                print(img.shape, type(img),  self.image_size)
                print(features, features.shape,  self.image_size)
                exit()


                data.append({'c_img': img, 'label': label_vec, 'features': features})

        np.random.shuffle(data)
        return data


    def load_train_set(self):
        '''
        Loads the train set
        '''

        self.train_data = self.load_set('train')


    def load_validation_set(self):
        '''
        Loads the validation set
        '''

        self.val_data = self.load_set('val')

