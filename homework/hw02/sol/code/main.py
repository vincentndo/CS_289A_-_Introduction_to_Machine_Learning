import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import itertools


def poly_eval(p, x):
	ret = 0
	for i in range(p.size):
		ret += p[i] * x ** i
	return ret


def estimate_errors(coeffs, x, y):
	err = 0
	n = y.size
	for j in range(n):
		err += ( y[j] - poly_eval(coeffs, x[j]) ) ** 2
	err /= n
	return err


def OLS(X, y):
	return np.matmul( np.linalg.inv( np.matmul(X.T, X) ), np.matmul(X.T, y) )


def OLS_solver(n, d, x, y):
	X = np.zeros( (n, d + 1), "float" )
	for i in range(d + 1):
		X[:, i] = np.power(x, i)

	coeffs = OLS(X, y)
	err = estimate_errors(coeffs, x, y)
	return coeffs, err


def multivariate_poly_eval(term_list, coeffs, x):
	ret = 0
	for i in range(len(term_list)):
		term = term_list[i]
		for t in term:
			coeffs[i] *= x[t]
		ret += coeffs[i]
	return ret


def multivariate_estimate_errors(term_list, coeffs, x, y):
	err = 0
	n = y.size
	for j in range(n):
		err += ( y[j] - multivariate_poly_eval(term_list, coeffs, x[j]) ) ** 2
	err /= n
	return err


def RR(X, l, y):
	I = np.identity(X.shape[1])
	return np.matmul( np.linalg.inv( np.matmul(X.T, X) + l * I ), np.matmul(X.T, y) )


def multivariate_RR_solver(m, d, l, x, y):
	term_list = []
	for term in itertools.combinations_with_replacement( list(range(m + 1)), d ):
		term = list(term)
		while m in term:
			term.remove(m)
		term_list.append(term)
	print(term_list)

	n = y.size
	X = np.ones( (n, len(term_list)), "float" )
	for i in range(len(term_list)):
		term = term_list[i]
		for t in term:
			X[:, i] *= x[:, t]

	coeffs = RR(X, l, y)
	err = multivariate_estimate_errors(term_list, coeffs, x, y)
	return term_list, coeffs, err


DATA_DIR = "./hw02-data"


# Question b

data_file_b = os.path.join(DATA_DIR, "1D_poly_new.mat")
xyy = scipy.io.loadmat(data_file_b)

x_train = np.ndarray.flatten( xyy.get("x_train") )
y_train = np.ndarray.flatten( xyy.get("y_train") )
y_fresh = np.ndarray.flatten( xyy.get("y_fresh") )

n = x_train.size

D_range = range(1, n)
training_errors = np.zeros( n - 1, "float" )

for d in D_range:
	(coeffs, err) = OLS_solver(n, d, x_train, y_train)
	training_errors[d - 1] = err

plt.figure()
plt.plot(D_range, training_errors, 'o-')
plt.title("Traning Error vs Polynomial Degree")
plt.xlabel("polynomial degree D")
plt.ylabel("training error")
print()
print("Problem 5.b")
print(training_errors)


# Question c

(coeffs, err) = OLS_solver(n, 20, x_train, y_train)
plt.figure()
plt.plot(x_train, y_train, 'ro', np.sort(x_train), np.polyval( np.flip(coeffs, 0), np.sort(x_train)), 'b')
print()
print("Problem 5.c")
print(err)


# Question d

fresh_errors = np.zeros( n - 1, "float" )

for d in D_range:
	(coeffs, _) = OLS_solver(n, d, x_train, y_train)
	fresh_errors[d - 1] = estimate_errors(coeffs, x_train, y_fresh)

plt.figure()
plt.plot(D_range, fresh_errors, 'go-')
plt.title("Fresh Error vs Polynomial Degree")
plt.xlabel("polynomial degree D")
plt.ylabel("fresh error")
print()
print("Problem 5.d")
print(fresh_errors)


# Question f

data_file_f = os.path.join(DATA_DIR, "polynomial_regression_samples.mat")
xxxxy = scipy.io.loadmat(data_file_f)

x_data = xxxxy.get('x')
y_data = xxxxy.get('y')

n = y_data.size
nt = int(n / 4)

D_range = range(1, 5)
errors_train = np.zeros( (len(D_range), 4), "float" )
errors_validate = np.zeros( (len(D_range), 4), "float" )


for i in range(4):
	x_train = np.concatenate( (x_data[:i*nt, :], x_data[(i+1)*nt:, :]), axis=0 )
	y_train = np.concatenate( (y_data[:i*nt, :], y_data[(i+1)*nt:, :]), axis=0 )
	x_validate = x_data[i*nt:(i+1)*nt, :]
	y_valicate = y_data[i*nt:(i+1)*nt, :]

	for d in D_range:
		term_list, coeffs, err_train = multivariate_RR_solver(5, d, 0.1, x_train, y_train)
		err_validate = multivariate_estimate_errors(term_list, coeffs, x_validate, y_valicate)
		errors_train[d-1, i] = err_train[0] 
		errors_validate[d-1, i] = err_validate[0]

errors_train_avg_over_folds = np.mean(errors_train, axis = 1)
errors_validate_avg_over_folds = np.mean(errors_validate, axis=1)

plt.figure()
plt.plot(D_range, errors_train_avg_over_folds, 'bx-', label="training")
plt.plot(D_range, errors_validate_avg_over_folds, 'bo-', label="validating")
plt.title("Training and Validating Errors vs Polynomial Degree")
plt.xlabel("polynomial degree D")
plt.ylabel("error")
plt.legend()
print()
print("Problem 5.f")
print("Training errors averaged over folds:")
print(errors_train_avg_over_folds)
print("Validating errors averaged over folds:")
print(errors_validate_avg_over_folds)
print("Coefficients")
print(coeffs)


# Question g

ll = [0.05, 0.1, 0.15, 0.2]
errors_train = np.zeros( (len(D_range), 4, len(ll)), "float" )
errors_validate = np.zeros( (len(D_range), 4, len(ll)), "float" )

for i in range(4):
	x_train = np.concatenate( (x_data[:i*nt, :], x_data[(i+1)*nt:, :]), axis=0 )
	y_train = np.concatenate( (y_data[:i*nt, :], y_data[(i+1)*nt:, :]), axis=0 )
	x_validate = x_data[i*nt:(i+1)*nt, :]
	y_valicate = y_data[i*nt:(i+1)*nt, :]

	for d in D_range:
		for li in range(len(ll)):
			term_list, coeffs, err_train = multivariate_RR_solver(5, d, ll[i], x_train, y_train)
			err_validate = multivariate_estimate_errors(term_list, coeffs, x_validate, y_valicate)
			errors_train[d-1, i, li] = err_train[0]
			errors_validate[d-1, i, li] = err_validate[0]

errors_train_avg_over_folds = np.mean(errors_train, axis=1)
errors_validate_avg_over_folds = np.mean(errors_validate, axis=1)

plt.figure()
colors = ['g', 'b', 'y', 'm']
for li in range(len(ll)):
	plt.plot(D_range, errors_train_avg_over_folds[:, li], colors[li]+'x-', label="training l="+str(ll[li]))
	plt.plot(D_range, errors_validate_avg_over_folds[:, li], colors[li]+'o-', label="validating l="+str(ll[li]))
plt.title("Training and Validating Errors vs Polynomial Degree")
plt.xlabel("polynomial degree D")
plt.ylabel("error")
plt.legend()
print()
print("Problem 5.g")
print("Training errors averaged over folds:")
print(errors_train_avg_over_folds)
print("Validating errors averaged over folds:")
print(errors_validate_avg_over_folds)


plt.show()
