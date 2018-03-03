from numpy.random import uniform
import random
import time

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys

from sklearn.svm import LinearSVC
from projection import Project2D, Projections

from utils import create_one_hot_label


class SVM_Model(): 

	def __init__(self,class_labels,projection=None):

		###SLACK HYPERPARAMETER
		self.C = 1.0
		self.class_labels = class_labels
		self.svc_model = LinearSVC(C=self.C)


	def train_model(self,X,Y): 
		''''
		FILL IN CODE TO TRAIN MODEL
		MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

		'''
		X = np.array(X)
		self.svc_model.fit(X, Y)
		

	def eval(self,x):
		''''
		Fill in code to evaluate model and return a prediction
		Prediction should be an integer specifying a class
		'''
		x = x.reshape(1, -1)
		y = self.svc_model.predict(x)
		return y[0]
