import random
import time

import numpy as np
import numpy.linalg as LA

from numpy.linalg import inv
from numpy.linalg import det

from projection import Project2D, Projections

from utils import subtract_mean_from_data
from utils import compute_covariance_matrix


class QDA_Model(): 

	def __init__(self,class_labels):

		###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
		self.reg_cov = 0.01
		self.NUM_CLASSES = len(class_labels)


	def train_model(self,X,Y): 
		''''
		FILL IN CODE TO TRAIN MODEL
		MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

		'''
		ps = [ [] for j in range(self.NUM_CLASSES) ] 
		for i, y in enumerate(Y):
			ps[y].append(X[i])

		self.mean_list = []
		self.Sigma_inv_list = []
		for lst in ps:
			self.mean_list.append( np.mean(np.array(lst), axis=0) )
			Sigma_XX = compute_covariance_matrix(lst, lst)
			Sigma_XX += self.reg_cov * np.identity(Sigma_XX.shape[0])
			self.Sigma_inv_list.append(inv(Sigma_XX))	
		

	def eval(self,x):
		''''
		Fill in code to evaluate model and return a prediction
		Prediction should be an integer specifying a class
		'''
		x = x.reshape(1, -1)
		y = {}
		for i in range(self.NUM_CLASSES):
			x_demeaned = x - self.mean_list[i]
			f = - x_demeaned.dot(self.Sigma_inv_list[i]).dot(x_demeaned.T)
			y[i] = f.flatten()[0]
		return max(y, key=lambda x: y[x])
