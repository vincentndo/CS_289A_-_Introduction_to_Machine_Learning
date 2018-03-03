import os
import numpy as np
from numpy.random import random
import cv2
import copy
import glob

import sys
import cPickle as pickle

import numpy.linalg as LA
import IPython


FILE_PATH = 'rollouts'

class HW3_Sol(object):

	def __init__(self):

		self.lmbda = 100

	def load_data(self):
		self.x_train = pickle.load(open('x_train.p','rb'))
		self.y_train = pickle.load(open('y_train.p','rb'))
		self.x_test = pickle.load(open('x_test.p','rb'))
		self.y_test = pickle.load(open('y_test.p','rb'))

	def compute_standardize_data_matrix(self):
	
		Y = []
		X = []

		Y_test = []
		X_test = []
				#for rollout_p in rollouts:

		#LOAD TRANING DATA 
		for x in self.x_train:


			#CONVERT TO FLOAT 64 from unint8
			x = 1.0*x

			#STANDARDIZE IMAGE
			x = (x/ 255.0) * 2.0 - 1.0
			

			X.append(x.flatten())


		for y in self.y_train:

			Y.append(y)

		#LOAD TEST DATA 

		for x in self.x_test:


			#CONVERT TO FLOAT 64 from unit8
			x = 1.0*x

			#STANDARDIZE IMAGE
			x = (x/ 255.0) * 2.0 - 1.0
			

			X_test.append(x.flatten())


		for y in self.y_test:

			Y_test.append(y)


		#CONVERT TO MATRIX
		self.X = np.vstack(X)
		self.Y = np.vstack(Y)

		self.X_test = np.vstack(X_test)
		self.Y_test = np.vstack(Y_test)

	def compute_data_matrix(self):
	
		Y = []
		X = []

		Y_test = []
		X_test = []
				#for rollout_p in rollouts:

		#LOAD TRANING DATA 
		for x in self.x_train:

			#CONVERT TO FLOAT 64 from unit8
			x = 1.0*x
			
			X.append(x.flatten())

		for y in self.y_train:

			Y.append(y)

		#LOAD TEST DATA 

		for x in self.x_test:
			

			#CONVERT TO FLOAT 64 form unint8
			x = 1.0*x
			
			X_test.append(x.flatten())


		for y in self.y_test:

			Y_test.append(y)

		#CONVERT TO MATRIX
		self.X = np.vstack(X)
		self.Y = np.vstack(Y)

		self.X_test = np.vstack(X_test)
		self.Y_test = np.vstack(Y_test)

	def ordinary_least_squares(self):
		X_G = np.matmul(self.X.T,self.X)
		
		###WILL THROW AN ERROR DUE TO RANK of X_G###
		p_1 = LA.inv(X_G)
		
		w = []
		for i in range(3):
			
			p_2 = np.matmul(self.Y[:,i],self.X)
			ans = np.matmul(p_2,p_1)
			w_ridge.append(ans)

		self.w = np.vstack(w_ridge)

	def ridge_regresion(self):


		X_G = np.matmul(self.X.T,self.X)

		#Add weighted indentity 
		ridge_f = np.eye(X_G.shape[0])*self.lmbda
		A = X_G+ridge_f
		

		p_1 = LA.inv(A)
		
		w_ridge = []
		for i in range(3):
			
			p_2 = np.matmul(self.Y[:,i],self.X)
			ans = np.matmul(p_2,p_1)
			w_ridge.append(ans)

		self.w_ridge = np.vstack(w_ridge)
		
	


	def measure_error_on_training(self):

		prediction = np.matmul(self.w_ridge,self.X.T)
		evaluation = self.Y.T - prediction

		dim,num_data = evaluation.shape
		error = []

	
		for i in range(num_data):
			
			#COMPUTE L2 NORM for each vector than  square
			error.append(LA.norm(evaluation[:,i])**2)

		#Return average error 
		return np.mean(error)

	def compute_condition_number(self):
		'''
		Function to compute condition number of the ridge regression optimization
		'''

		X_G = np.matmul(self.X.T,self.X)
		#Add weighted indentity 
		ridge_f = np.eye(X_G.shape[0])*self.lmbda
		A = X_G+ridge_f
		

		p_1 = LA.inv(A)

		#COMPUTES SINGULAR VALUES
		u,s,d = LA.svd(A)

		#SQUARE TO GET EIGENVALUES
		L_Eigenvalue = s[0]**2
		S_eigenvlaue = s[-1]**2

		condition_number = L_Eigenvalue/S_eigenvlaue
		return condition_number

	def measure_error_on_test(self):

		prediction = np.matmul(self.w_ridge,self.X_test.T)
		evaluation = self.Y_test.T - prediction

		dim,num_data = evaluation.shape

		
		error = []
	
		for i in range(num_data):
			
			#COMPUTE L2 NORM for each vector than  square
			error.append(LA.norm(evaluation[:,i])**2)

		#Return average error 
		return np.mean(error)




if __name__ == '__main__':

	hw3_sol = HW3_Sol()

	hw3_sol.load_data()

	LAMBDA = [0.1,1.0,10.0,100.0,1000.0]
	

	print "-------PROBLEM A------"
	hw3_sol.compute_data_matrix()

	try:
		hw3_sol.ordinary_least_squares()
	except:
		print "MATRIX NOT FULL RANK COULDN'T INVERT"

	# #SHOULD THROW AN ERROR IN THE INVERSION DUE TO THE MATRIX BEING SINGULAR

	print "-------PROBLEM B------"
	hw3_sol.compute_data_matrix()

	for lmbda in LAMBDA:
		hw3_sol.lmbda = lmbda
		hw3_sol.ridge_regresion()
		print "LAMBDA ",lmbda
		print "ERROR ON TRAINING ",hw3_sol.measure_error_on_training()

	print "-------PROBLEM C------"
	hw3_sol.compute_standardize_data_matrix()

	for lmbda in LAMBDA:
		hw3_sol.lmbda = lmbda
		hw3_sol.ridge_regresion()
		print "LAMBDA ",lmbda
		print "ERROR ON TRAINING ",hw3_sol.measure_error_on_training()

	print "-------PROBLEM D------"
	hw3_sol.compute_data_matrix()

	for lmbda in LAMBDA:
		hw3_sol.lmbda = lmbda
		hw3_sol.ridge_regresion()
		print "LAMBDA ",lmbda
		print "ERROR ON TEST ",hw3_sol.measure_error_on_test()

	hw3_sol.compute_standardize_data_matrix()

	for lmbda in LAMBDA:
		hw3_sol.lmbda = lmbda
		hw3_sol.ridge_regresion()
		print "LAMBDA ",lmbda
		print "ERROR ON TEST WITH STANDARDIZATION ",hw3_sol.measure_error_on_test()

	print "-------PROBLEM E------"
	hw3_sol.lmbda = 100.0
	hw3_sol.compute_data_matrix()
	c_n = hw3_sol.compute_condition_number()
	print "CONDITION NUMBER ",c_n

	hw3_sol.compute_standardize_data_matrix()
	c_n = hw3_sol.compute_condition_number()
	print "CONDITION NUMBER  WITH STANDARDIZATION ",c_n






