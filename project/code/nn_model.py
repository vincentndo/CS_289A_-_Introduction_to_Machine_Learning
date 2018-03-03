import numpy as np
from  sklearn.neighbors import KNeighborsClassifier

class NN():

	def __init__(self,train_data,val_data,n_neighbors=5):

		self.train_data = train_data
		self.val_data = val_data
		self.sample_size = 400
		self.model = KNeighborsClassifier(n_neighbors=n_neighbors)


	def create_data(self,data):
		X = []
		Y = []
		for datum in data:
			X.append(datum['features'].flatten())
			Y.append(np.argmax(datum['label']))

		return X,Y


	def train_model(self):
		'''
		Train Nearest Neighbors model
		'''
		X,Y = self.create_data(self.train_data)
		self.model.fit(X,Y)


	def get_validation_error(self):

		'''
		Compute validation accuracy. Only compute the error on the sample_size number
		over randomly selected data points. To save computation.
		'''
		X,Y = self.create_data(self.val_data)

		acc = 0.0

		y_b_ = self.model.predict(X[0:400])
		y_b = Y[0:400]

		for i in range(len(y_b)):

			if y_b[i] == y_b_[i]:
				acc += 1.0

		return acc/float(400)


	def get_train_error(self):

		'''
		Compute train accuracy. Only compute the error on the sample_size number
		over randomly selected data points. To save computation.
		'''
		X,Y = self.create_data(self.train_data)

		acc = 0.0

		y_b_ = self.model.predict(X[0:400])
		y_b = Y[0:400]

		for i in range(len(y_b)):

			if y_b[i] == y_b_[i]:
				acc += 1.0

		return acc/float(400)
