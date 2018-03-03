import numpy as np
import numpy.linalg as LA

import matplotlib.pyplot as plt

import random
import os
import cv2

from projection import Project2D, Projections
from confusion_mat import getConfusionMatrixPlot

from ridge_model import Ridge_Model
from qda_model import QDA_Model
from lda_model import LDA_Model
from svm_model import SVM_Model


data_dir = "../data"
CLASS_LABELS = ['no_package','package']

class Model():
	""" Generic wrapper for specific model instance. """

	def __init__(self, model):
		""" Store specific pre-initialized model instance. """

		self.model = model


	def train_model(self,X,Y): 
		""" Train using specific model's training function. """
		
		self.model.train_model(X,Y)
	

	def test_model(self,X,Y):
		""" Test using specific model's eval function. """
		
		labels = []						# List of actual labels
		p_labels = []					# List of model's predictions
		success = 0						# Number of correct predictions
		total_count = 0					# Number of images

		for i in range(len(X)):
			
			x = X[i]					# Test input
			y = Y[i]					# Actual label
			y_ = self.model.eval(x)		# Model's prediction
			labels.append(y)
			p_labels.append(y_)

			if y == y_:
				success += 1
			total_count +=1 
			
		
		
		# Compute Confusion Matrix
		getConfusionMatrixPlot(labels,p_labels,CLASS_LABELS)



if __name__ == "__main__":

	temp_dir = os.path.join(data_dir, "tmp")
	resized_box_dir = os.path.join(temp_dir, "resized_box")
	resized_nobox_dir = os.path.join(temp_dir, "resized_nobox")

	box_list = []
	count = 0
	for filename in os.listdir(resized_box_dir):
	    label = 1
	    file_path = os.path.join(resized_box_dir, filename)
	    img = cv2.imread(file_path)
	    features = features = img.flatten()
	    box_list.append({'c_img': img, 'label': label, 'features': features})
	assert len(box_list) == 22394, "Wrong appending box filenames"
	random.shuffle(box_list)

	nobox_list = []
	for filename in os.listdir(resized_nobox_dir):
	    label = 0
	    file_path = os.path.join(resized_nobox_dir, filename)
	    img = cv2.imread(file_path)
	    features = img.flatten()
	    nobox_list.append({'c_img': img, 'label': label, 'features': features})
	assert len(nobox_list) == 6100, "Wrong appending nobox filenames"
	random.shuffle(nobox_list)

	box_list.extend(nobox_list)
	all_list = box_list
	random.shuffle(all_list)
	assert len(all_list) == 22394 + 6100, "Wrong appending all filenames"

	X_train_data_list = []
	Y_train_data_list = []
	X_eval_data_list = []
	Y_eval_data_list = []
	for i in range(len(all_list)):
	    data_point = box_list.pop(0)
	    if i < 23000:
	        X_train_data_list.append(data_point["features"])
	        Y_train_data_list.append(data_point["label"])
	    else:
	        X_eval_data_list.append(data_point["features"])
	        Y_eval_data_list.append(data_point["label"])
	assert len(X_train_data_list) == 23000, "Wrong total train count"
	assert len(X_eval_data_list) == 5494, "Wrong total eval count"


	# Load Training Data and Labels
	X = X_train_data_list[:3000]
	Y = Y_train_data_list[:3000]


	# Load Validation Data and Labels
	X_val = X_eval_data_list[3000:3400]
	Y_val = Y_eval_data_list[3000:3600]

	CLASS_LABELS = ['no_package','package']


	# Project Data to 200 Dimensions using CCA
	feat_dim = max(X[0].shape)
	projections = Projections(feat_dim,CLASS_LABELS)
	cca_proj,white_cov = projections.cca_projection(X,Y,k=2)

	X = projections.project(cca_proj,white_cov,X)
	X_val = projections.project(cca_proj,white_cov,X_val)


	####RUN RIDGE REGRESSION#####
	print("Ridge model")
	ridge_m = Ridge_Model(CLASS_LABELS)
	model = Model(ridge_m)

	model.train_model(X,Y)
	model.test_model(X,Y)
	model.test_model(X_val,Y_val)


	####RUN LDA REGRESSION#####
	print("LDA model")
	lda_m = LDA_Model(CLASS_LABELS)
	model = Model(lda_m)

	model.train_model(X,Y)
	model.test_model(X,Y)
	model.test_model(X_val,Y_val)


	###RUN QDA REGRESSION#####
	print("QDA model")
	qda_m = QDA_Model(CLASS_LABELS)
	model = Model(qda_m)

	print(len(X), len(Y))
	model.train_model(X,Y)
	model.test_model(X,Y)
	model.test_model(X_val,Y_val)
	

	####RUN SVM REGRESSION#####
	print("SVM model")
	svm_m = SVM_Model(CLASS_LABELS)
	model = Model(svm_m)

	model.train_model(X,Y)
	model.test_model(X,Y)
	model.test_model(X_val,Y_val)
