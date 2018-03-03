import numpy as np
import scipy.spatial

########################################################################
#########  Data Generating Functions ###################################
########################################################################
def generate_sensors(k = 7, d = 2):
   """
   Generate sensor locations. 
   Input:
   k: The number of sensors.
   d: The spatial dimension.
   Output:
   sensor_loc: k * d numpy array.
   """
   sensor_loc = 100*np.random.randn(k,d)
   return sensor_loc

def generate_data(sensor_loc, k = 7, d = 2, 
				 n = 1, original_dist = True, sigma_s = 100):
   """
   Generate the locations of n points and distance measurements.  
   
   Input:
   sensor_loc: k * d numpy array. Location of sensor. 
   k: The number of sensors.
   d: The spatial dimension.
   n: The number of points.
   original_dist: Whether the data are generated from the original 
   distribution. 
   sigma_s: the standard deviation of the distribution 
   that generate each object location.
   
   Output:
   obj_loc: n * d numpy array. The location of the n objects. 
   distance: n * k numpy array. The distance between object and 
   the k sensors. 
   """
   assert k, d == sensor_loc.shape
   
   obj_loc = sigma_s*np.random.randn(n, d)
   if not original_dist:
	   obj_loc += 1000
	   
   distance = scipy.spatial.distance.cdist(obj_loc, 
										   sensor_loc, 
										   metric='euclidean')
   distance += np.random.randn(n, k)  
   return obj_loc, distance

def generate_data_given_location(sensor_loc, obj_loc, k = 7, d = 2):
   """
   Generate the distance measurements given location of a single object and sensor. 
   
   Input:
   obj_loc: 1 * d numpy array. Location of object
   sensor_loc: k * d numpy array. Location of sensor. 
   k: The number of sensors.
   d: The spatial dimension. 
   
   Output: 
   distance: 1 * k numpy array. The distance between object and 
   the k sensors. 
   """
   assert k, d == sensor_loc.shape 
	   
   distance = scipy.spatial.distance.cdist(obj_loc, 
										   sensor_loc, 
										   metric='euclidean')
   distance += np.random.randn(1, k)  
   return obj_loc, distance

########################################################################
######### Part b ###################################
########################################################################

########################################################################
#########  Gradient Computing and MLE ###################################
########################################################################
def compute_gradient_of_likelihood(single_obj_loc, sensor_loc, 
								single_distance):
	"""
	Compute the gradient of the loglikelihood function for part a.   
	
	Input:
	single_obj_loc: 1 * d numpy array. 
	Location of the single object.
	
	sensor_loc: k * d numpy array. 
	Location of sensor.
	
	single_distance: k dimensional numpy array. 
	Observed distance of the object.
	
	Output:
	grad: d-dimensional numpy array.
	
	"""
	loc_difference = single_obj_loc - sensor_loc # k * d.
	phi = np.linalg.norm(loc_difference, axis = 1) # k. 
	weight = (phi - single_distance) / phi # k.
	
	grad = -np.sum(np.expand_dims(weight,1)*loc_difference, 
				   axis = 0) # d
	return grad

def find_mle_by_grad_descent_part_b(initial_obj_loc, 
		   sensor_loc, single_distance, lr=0.001, num_iters = 10000):
	"""
	Compute the gradient of the loglikelihood function for part a.   
	
	Input:
	initial_obj_loc: 1 * d numpy array. 
	Initialized Location of the single object.
	
	sensor_loc: k * d numpy array. Location of sensor.
	
	single_distance: k dimensional numpy array. 
	Observed distance of the object.
	
	Output:
	obj_loc: 1 * d numpy array. The mle for the location of the object.
	
	"""    
	obj_loc = initial_obj_loc
	for t in range(num_iters):
		obj_loc += lr * compute_gradient_of_likelihood(obj_loc, 
						  sensor_loc, single_distance) 
		
	return obj_loc
	
########################################################################
#########  MAIN ########################################################
########################################################################

np.random.seed(0)
sensor_loc = generate_sensors()
obj_loc, distance = generate_data(sensor_loc)
single_distance = distance[0]
print('The real object location is')
print(obj_loc)
# Initialized as [0,0]
initial_obj_loc = np.array([[0.,0.]]) 
estimated_obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc, 
		   sensor_loc, single_distance, lr=0.001, num_iters = 10000)
print('The estimated object location with zero initialization is')
print(estimated_obj_loc)

# Random initialization.
initial_obj_loc = np.random.randn(1,2)
estimated_obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc, 
		   sensor_loc, single_distance, lr=0.001, num_iters = 10000)
print('The estimated object location with random initialization is')
print(estimated_obj_loc)   

########################################################################
######### Part c #################################################
########################################################################
def log_likelihood(obj_loc, sensor_loc, distance): 
	"""
	This function computes the log likelihood (as expressed in Part a).
	Input: 
	obj_loc: shape [1,2]
	sensor_loc: shape [7,2]
	distance: shape [7]
	Output: 
	The log likelihood function value. 
	"""  
	diff_distance = np.sqrt(np.sum((sensor_loc - obj_loc)**2, axis = 1))- distance
	func_value = -sum((diff_distance)**2)/2
	return func_value

########################################################################
######### Compute the function value at local minimum for all experiments.###
########################################################################
np.random.seed(100)
sensor_loc = generate_sensors()

# num_data_replicates = 10
num_gd_replicates = 100

obj_locs = [[[i,i]] for i in np.arange(0,1000,100)]

func_values = np.zeros((len(obj_locs),10, num_gd_replicates))
for i, obj_loc in enumerate(obj_locs): 
    for j in range(10):
        obj_loc, distance = generate_data_given_location(sensor_loc, obj_loc, 
                                                         k = 7, d = 2)
        for gd_replicate in range(num_gd_replicates): 
            initial_obj_loc = np.random.randn(1,2)* (100 * i+1)
            obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc, 
                       sensor_loc, distance[0], lr=0.1, num_iters = 1000) 
            func_value = log_likelihood(obj_loc, sensor_loc, distance[0])
            func_values[i, j, gd_replicate] = func_value

########################################################################
######### Calculate the things to be plotted. ###
########################################################################
local_mins = [[np.unique(func_values[i,j].round(decimals=2)) for j in range(10)] for i in range(10)]
num_local_min = [[len(local_mins[i][j]) for j in range(10)] for i in range(10)]
proportion_global = [[sum(func_values[i,j].round(decimals=2) == min(local_mins[i][j]))*1.0/100 \
                       for j in range(10)] for i in range(10)]


num_local_min = np.array(num_local_min)
num_local_min = np.mean(num_local_min, axis = 1)

proportion_global = np.array(proportion_global)
proportion_global = np.mean(proportion_global, axis = 1)

########################################################################
######### Plots. #######################################################
########################################################################

plt.plot(np.arange(0,1000,100), num_local_min)
plt.title('Number of local minimum found by 100 gradient descents.')
plt.xlabel('Object Location')
plt.ylabel('Number')
plt.savefig('num_obj.png')
# Proportion of gradient descents that find the local minimum of minimum value. 
plt.plot(np.arange(0,1000,100), proportion_global)
plt.title('Proportion of GD that finds the minimum local minimum among 100 gradient descents.')
plt.xlabel('Object Location')
plt.ylabel('Proportion')
plt.savefig('prop_obj.png')

########################################################################
######### Plots of contours. ###########################################
########################################################################
import matplotlib 
import matplotlib.pyplot as plt
np.random.seed(0) 
# sensor_loc = np.random.randn(7,2) * 10
x = np.arange(-10.0, 10.0, 0.1)
y = np.arange(-10.0, 10.0, 0.1)
X, Y = np.meshgrid(x, y) 
obj_loc = [[0,0]]
obj_loc, distance = generate_data_given_location(sensor_loc, 
                                                 obj_loc, k = 7, d = 2)

Z =  np.array([[log_likelihood((X[i,j],Y[i,j]), 
                               sensor_loc, distance[0]) for j in range(len(X))] \
               for i in range(len(X))]) 


plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('With object at (0,0)')
plt.show()

np.random.seed(0) 
# sensor_loc = np.random.randn(7,2) * 10
x = np.arange(-200,200, 1)
y = np.arange(-200,200, 1)
X, Y = np.meshgrid(x, y) 
obj_loc = [[100,100]]
obj_loc, distance = generate_data_given_location(sensor_loc, 
                                                 obj_loc, k = 7, d = 2)

Z =  np.array([[log_likelihood((X[i,j],Y[i,j]), 
                               sensor_loc, distance[0]) for j in range(len(X))] \
               for i in range(len(X))]) 


# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('With object at (100,100)')
plt.show()






########################################################################
######### Part e, f, g #################################################
########################################################################

########################################################################
#########  Gradient Computing and MLE ##################################
########################################################################
def compute_grad_likelihood_part_e(sensor_loc, obj_loc, distance):
	"""
	Compute the gradient of the loglikelihood function for part d.   
	
	Input:
	sensor_loc: k * d numpy array. 
	Location of sensors.
	
	obj_loc: n * d numpy array. 
	Location of the objects.
	
	distance: n * k dimensional numpy array. 
	Observed distance of the object.
	
	Output:
	grad: k * d numpy array.
	"""
	grad = np.zeros(sensor_loc.shape)
	for i, single_sensor_loc in enumerate(sensor_loc):
		single_distance = distance[:,i] 
		grad[i] = compute_gradient_of_likelihood(single_sensor_loc, 
					 obj_loc, single_distance)
		
	return grad

def find_mle_by_grad_descent_part_e(initial_sensor_loc, 
		   obj_loc, distance, lr=0.001, num_iters = 1000):
	"""
	Compute the gradient of the loglikelihood function for part a.   
	
	Input:
	initial_sensor_loc: k * d numpy array. 
	Initialized Location of the sensors.
	
	obj_loc: n * d numpy array. Location of the n objects.
	
	distance: n * k dimensional numpy array. 
	Observed distance of the n object.
	
	Output:
	sensor_loc: k * d numpy array. The mle for the location of the object.
	
	"""    
	sensor_loc = initial_sensor_loc
	for t in range(num_iters):
		sensor_loc += lr * compute_grad_likelihood_part_e(\
			sensor_loc, obj_loc, distance) 
		
	return sensor_loc
########################################################################
#########  Gradient Computing and MLE ##################################
########################################################################

np.random.seed(0)
sensor_loc = generate_sensors()
obj_loc, distance = generate_data(sensor_loc, n = 100)
print('The real sensor locations are')
print(sensor_loc)
# Initialized as zeros.
initial_sensor_loc = np.zeros((7,2)) #np.random.randn(7,2)
estimated_sensor_loc = find_mle_by_grad_descent_part_e(initial_sensor_loc, 
		   obj_loc, distance, lr=0.001, num_iters = 1000)
print('The predicted sensor locations are')
print(estimated_sensor_loc) 

 
 ########################################################################
#########  Estimate distance given estimated sensor locations. ######### 
########################################################################

def compute_distance_with_sensor_and_obj_loc(sensor_loc, obj_loc):
	"""
	stimate distance given estimated sensor locations.  
	
	Input:
	sensor_loc: k * d numpy array. 
	Location of the sensors.
	
	obj_loc: n * d numpy array. Location of the n objects.
	
	Output:
	distance: n * k dimensional numpy array. 
	""" 
	estimated_distance = scipy.spatial.distance.cdist(obj_loc, 
											sensor_loc, 
											metric='euclidean')
	return estimated_distance 
########################################################################
#########  MAIN  #######################################################
########################################################################    
np.random.seed(100)    
########################################################################
#########  Case 1. #####################################################
########################################################################
obj_loc, distance = generate_data(sensor_loc, k = 7, d = 2, 
				  n = 1000, original_dist = True)

estimated_distance = compute_distance_with_sensor_and_obj_loc(estimated_sensor_loc, 
															  obj_loc)

mse = np.mean(np.sum(estimated_distance, axis = 1))

print('The MSE for Case 1 is {}'.format(mse))

########################################################################
#########  Case 2. #####################################################
########################################################################
obj_loc, distance = generate_data(sensor_loc, k = 7, d = 2, 
				  n = 100, original_dist = False)

estimated_distance = compute_distance_with_sensor_and_obj_loc(estimated_sensor_loc, 
															  obj_loc)

mse = np.mean(np.sum(estimated_distance, axis = 1))

print('The MSE for Case 2 is {}'.format(mse)) 

