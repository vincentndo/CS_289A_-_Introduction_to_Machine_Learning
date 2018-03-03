import matplotlib.pyplot as plt

DATA_PATH = "accuracy_5000.txt"

if __name__ == "__main__":
	iter_num_list = []
	train_accuracy_list = []
	val_accuracy_list = []
	with open(DATA_PATH, "r") as file:
		for line in file:
			line_list = line.strip().split(' ')
			index = int(line_list[0])
			if index % 200 == 0:
				iter_num_list.append(index)
				train_accuracy_list.append(float(line_list[1]))
				val_accuracy_list.append(float(line_list[2]))

		iter_num_list.append(int(line_list[0]))
		train_accuracy_list.append(float(line_list[1]))
		val_accuracy_list.append(float(line_list[2]))
		
	plt.plot(iter_num_list, train_accuracy_list,label = 'Validation')
	plt.plot(iter_num_list, val_accuracy_list, label = 'Training')
	plt.legend()
	plt.xlabel('Iterations (in 200s)')
	plt.ylabel('Accuracy')
	plt.show()
