from numpy.random import uniform
import random
import time

import numpy as np
import numpy.linalg as LA

import sys

from sklearn.linear_model import Ridge

from utils import create_one_hot_label


class Ridge_Model(): 

	def __init__(self,class_labels):

		###RIDGE HYPERPARAMETER
		self.lmda = 1.0
		self.class_labels = class_labels
		self.ridge_model = Ridge(self.lmda)


	def train_model(self,X,Y): 
		''''
		FILL IN CODE TO TRAIN MODEL
		MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

		'''
		
		X = np.array(X)
		y_one_hot = create_one_hot_label(Y, len(self.class_labels))
		self.ridge_model.fit(X, y_one_hot)		
		

	def eval(self,x):
		''''
		Fill in code to evaluate model and return a prediction
		Prediction should be an integer specifying a class
		'''
		x = x.reshape(1, -1)
		y = self.ridge_model.predict(x)
		return np.argmax(y)
	