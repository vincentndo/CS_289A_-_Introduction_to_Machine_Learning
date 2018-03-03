import matplotlib.pyplot as plt
from data_loader import data_loader
from nn_model import NN


image_size = 32
classes = ['no_package','package']

dm = data_loader(classes, image_size)

val_data = dm.val_data
train_data = dm.train_data

K = [1, 5, 10, 20, 40, 70, 100]
test_accuracy = []
train_accuracy= []

for k in K:
	nn = NN(val_data,train_data,n_neighbors=k)

	nn.train_model()

	test_accuracy.append(nn.get_validation_error())
	train_accuracy.append(nn.get_train_error())

plt.plot(K, train_accuracy, ".-", label = 'Training')
plt.plot(K, test_accuracy, ".-", label = 'Validation')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
