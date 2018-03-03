import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os


DATA_DIR = "./hw01-data"


# Question a

data_file_a = os.path.join(DATA_DIR, "system_identification_programming_a.mat")
xu = scipy.io.loadmat(data_file_a)

n = xu['x'].size

x = xu['x'][:, :n-1].T
u = xu['u'][:, :n-1].T
X = np.concatenate( (x, u), axis=1 )
X_t = X.T

Y = xu['x'][:, 1:].T

AB = np.matmul( np.linalg.inv( np.matmul(X_t, X) ), np.matmul(X_t, Y) )
A, B = AB[0, 0], AB[1, 0]

print(A, B)


# Question b

data_file_b = os.path.join(DATA_DIR, "system_identification_programming_b.mat")
x3u3 = scipy.io.loadmat(data_file_b)

mn = x3u3['x'].shape
m, n = mn[0], mn[1]

X = np.zeros( ((m - 1) * n, n ** 2 * 2), "float")
for i in range(m - 1):
	x = x3u3['x'][i, :, :]
	u = x3u3['u'][i, :, :]
	xu = np.concatenate( (x,u), axis=0 )
	sub_X = np.zeros( (n, n ** 2 * 2), "float")
	for j in range(n):
		sub_X[j, j*n*2:(j+1)*n*2] = xu[:, 0]
	X[i*n:(i+1)*n, :] = sub_X
X_t = X.T

Y = x3u3['x'][1:, :].reshape( (m - 1) * n, 1)

AB = np.matmul( np.linalg.inv( np.matmul(X_t, X) ), np.matmul(X_t, Y) )

A = np.zeros( (n ** 2, 1), "float" )
B = np.zeros( (n ** 2, 1), "float" )
for i in range(n):
	A[i*n:(i+1)*n, :] = AB[i*2*n:(i*2+1)*n, :]
	B[i*n:(i+1)*n, :] = AB[(i*2+1)*n:(i*2+2)*n, :]
A = A.reshape( (n, n) )
B = B.reshape( (n, n) )

print(A)
print(B)


# Question e

SPEED_LIMIT = 29.0576

data_file_e = os.path.join(DATA_DIR, "system_identification_train.mat")
xes = scipy.io.loadmat(data_file_e)

n = xes['x'].size

x = xes['x'].T
xd = xes['xd'].T + SPEED_LIMIT
xp = xes['xp'].T
xdp = xes['xdp'].T + SPEED_LIMIT
xdd = xes['xdd'].T
X = np.concatenate( (x, xd, xp, xdp), axis=1 )
X_t = X.T

Y = xdd

abcd = np.matmul( np.linalg.inv( np.matmul(X_t, X) ), np.matmul(X_t, Y) )
a, b, c, d = abcd[0, 0], abcd[1, 0], abcd[2, 0], abcd[3, 0],

print(a, b, c, d)


# Question f

data_file_f = os.path.join(DATA_DIR, "system_identification_eval.mat")
x0xt = scipy.io.loadmat(data_file_f)

x0 = x0xt['x0'][0, 0]
xd0 = x0xt['xd0'][0, 0] + SPEED_LIMIT
xp = x0xt['xp']
xdp = x0xt['xdp'] + SPEED_LIMIT

xdd0 = a * x0 + b * xd0 + c * xp[0, 0] + d * xdp[0, 0]

if os.path.isfile("submission.txt"):
	os.remove("submission.txt")

fid = open("submission.txt", 'w')
fid.write(str(x0) + '\n')

delta_t = 1 / 15
v0, accel = xd0, xdd0
for i in range(1, 150):
	x0 = x0 + v0 * delta_t + 0.5 * accel * delta_t ** 2
	fid.write(str(x0) + '\n')
	v0 = v0 + accel * delta_t
	accel = a * x0 + b * v0 + c * xp[0, i] + d * xdp[0, i]

fid.close()
