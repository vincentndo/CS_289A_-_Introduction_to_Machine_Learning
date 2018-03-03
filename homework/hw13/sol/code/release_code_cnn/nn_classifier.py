import IPython
from numpy.random import uniform
import random
import time

import numpy as np
import glob
import os

import matplotlib.pyplot as plt


import sys

from  sklearn.neighbors import KNeighborsClassifier



class NN(): 


	def __init__(self,train_data,val_data,n_neighbors=5):

		self.train_data = train_data
		self.val_data = val_data

		self.sample_size = 400

		self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

		
	def train_model(self): 

		'''
		Train Nearest Neighbors model
		'''
		X_train = np.array( [ np.copy(self.train_data[i]["features"]).flatten() for i in range(len(self.train_data)) ] )
		y_train = np.array( [ self.train_data[i]["label"] for i in range(len(self.train_data)) ], dtype="uint8" )
		zero = np.zeros( [1, 25], dtype="uint8")

		for i in range(len(y_train)):
			if np.array_equal(y_train[i], zero):
				print("eureka")
				
		self.model.fit(X_train, y_train)



	def get_validation_error(self):

		'''
		Compute validation error. Please only compute the error on the sample_size number 
		over randomly selected data points. To save computation. 

		'''
		X_val_sampled = []
		y_val_sampled = []
		for i in range(self.sample_size):
			index = random.randint(0, len(self.val_data) - 1)
			X_val_sampled.append( np.copy(self.val_data[index]["features"]).flatten() )
			y_val_sampled.append( self.val_data[index]["label"] )
		
		X_val_sampled = np.array(X_val_sampled)
		y_val_sampled = np.array(y_val_sampled, dtype="uint8")

		y_predicted = self.model.predict(X_val_sampled)
		count = 0
		for i in range(self.sample_size):
			if not np.array_equal(y_predicted[i], y_val_sampled[i]):
				count += 1

		print("Val error: " + str(count / self.sample_size))
		return count / self.sample_size




	def get_train_error(self):

		'''
		Compute train error. Please only compute the error on the sample_size number 
		over randomly selected data points. To save computation. 
		'''

		X_train_sampled = []
		y_train_sampled = []
		for i in range(self.sample_size):
			index = random.randint(0, len(self.train_data) - 1)
			X_train_sampled.append( np.copy(self.train_data[index]["features"]).flatten() )
			y_train_sampled.append( self.train_data[index]["label"] )
		
		X_train_sampled = np.array(X_train_sampled)
		y_train_sampled = np.array(y_train_sampled, dtype="uint8")

		y_predicted = self.model.predict(X_train_sampled)
		count = 0
		for i in range(self.sample_size):
			if not np.array_equal(y_predicted[i], y_train_sampled[i]):
				count += 1

		print("Train error: " + str(count / self.sample_size))
		return count / self.sample_size
