import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Gradient descent optimization
# The learning rate is specified by eta
class GDOptimizer(object):
    def __init__(self, eta):
        self.eta = eta

    def initialize(self, layers):
        pass

    # This function performs one gradient descent step
    # layers is a list of dense layers in the network
    # g is a list of gradients going into each layer before the nonlinear activation
    # a is a list of of the activations of each node in the previous layer going 
    def update(self, layers, g, a):
        m = a[0].shape[1]
        for layer, curGrad, curA in zip(layers, g, a):
            update = np.dot(curGrad,curA.T)
            updateB = np.sum(curGrad,1).reshape(layer.b.shape)
            layer.updateWeights(-self.eta/m * np.dot(curGrad,curA.T))
            layer.updateBias(-self.eta/m * np.sum(curGrad,1).reshape(layer.b.shape))

# Cost function used to compute prediction errors
class QuadraticCost(object):

    # Compute the squared error between the prediction yp and the observation y
    # This method should compute the cost per element such that the output is the
    # same shape as y and yp
    @staticmethod
    def fx(y,yp):
        return 0.5 * np.square(yp-y)

    # Derivative of the cost function with respect to yp
    @staticmethod
    def dx(y,yp):
        return y - yp

# Sigmoid function fully implemented as an example
class SigmoidActivation(object):
    @staticmethod
    def fx(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def dx(z):
        return SigmoidActivation.fx(z) * (1 - SigmoidActivation.fx(z))
        
# Hyperbolic tangent function
class TanhActivation(object):

    # Compute tanh for each element in the input z
    @staticmethod
    def fx(z):
        return np.tanh(z)

    # Compute the derivative of the tanh function with respect to z
    @staticmethod
    def dx(z):
        return 1 - np.square(np.tanh(z))

# Rectified linear unit
class ReLUActivation(object):
    @staticmethod
    def fx(z):
        return np.maximum(0,z)

    @staticmethod
    def dx(z):
        #print('z:\n'+str(z))
        #print('relu(z):\n'+str((z>0).astype('float')))
        #stuff
        return (z>0).astype('float')

# Linear activation
class LinearActivation(object):
    @staticmethod
    def fx(z):
        return z

    @staticmethod
    def dx(z):
        return np.ones(z.shape)

# This class represents a single hidden or output layer in the neural network
class DenseLayer(object):

    # numNodes: number of hidden units in the layer
    # activation: the activation function to use in this layer
    def __init__(self, numNodes, activation):
        self.numNodes = numNodes
        self.activation = activation

    def getNumNodes(self):
        return self.numNodes

    # Initialize the weight matrix of this layer based on the size of the matrix W
    def initialize(self, fanIn, scale=1.0):
        s = scale * np.sqrt(6.0 / (self.numNodes + fanIn))
        self.W = np.random.normal(0, s,
                                   (self.numNodes,fanIn))
        #self.b = np.zeros((self.numNodes,1))
        self.b = np.random.uniform(-1,1,(self.numNodes,1))

    # Apply the activation function of the layer on the input z
    def a(self, z):
        return self.activation.fx(z)

    # Compute the linear part of the layer
    # The input a is an n x k matrix where n is the number of samples
    # and k is the dimension of the previous layer (or the input to the network)
    def z(self, a):
        #print('a:\n'+str(a))
        #print('Wa:\n'+str(self.W.dot(a)))
        return self.W.dot(a) + self.b # Note, this is implemented where we assume a is k x n

    # Compute the derivative of the layer's activation function with respect to z
    # where z is the output of the above function.
    # This derivative does not contain the derivative of the matrix multiplication
    # in the layer.  That part is computed below in the model class.
    def dx(self, z):
        return self.activation.dx(z)

    # Update the weights of the layer by adding dW to the weights
    def updateWeights(self, dW):
        self.W = self.W + dW

    # Update the bias of the layer by adding db to the bias
    def updateBias(self, db):
        self.b = self.b + db

# This class handles stacking layers together to form the completed neural network
class Model(object):

    # inputSize: the dimension of the inputs that go into the network
    def __init__(self, inputSize):
        self.layers = []
        self.inputSize = inputSize

    # Add a layer to the end of the network
    def addLayer(self, layer):
        self.layers.append(layer)

    # Get the output size of the layer at the given index
    def getLayerSize(self, index):
        if index >= len(self.layers):
            return self.layers[-1].getNumNodes()
        elif index < 0:
            return self.inputSize
        else:
            return self.layers[index].getNumNodes()

    # Initialize the weights of all of the layers in the network and set the cost
    # function to use for optimization
    def initialize(self, cost, initializeLayers=True):
        self.cost = cost
        if initializeLayers:
            for i in range(0,len(self.layers)):
                if i == len(self.layers) - 1:
                    self.layers[i].initialize(self.getLayerSize(i-1))
                else:
                    self.layers[i].initialize(self.getLayerSize(i-1))

    # Compute the output of the network given some input a
    # The matrix a has shape n x k where n is the number of samples and
    # k is the dimension
    # This function returns
    # yp - the output of the network
    # a - a list of inputs for each layer of the newtork where
    #     a[i] is the input to layer i
    # z - a list of values for each layer after evaluating layer.z(a) but
    #     before evaluating the nonlinear function for the layer
    def evaluate(self, x):
        curA = x.T
        a = [curA]
        z = []
        for layer in self.layers:
            z.append(layer.z(curA))
            curA = layer.a(z[-1])
            a.append(curA)
        yp = a.pop()
        return yp, a, z

    # Compute the output of the network given some input a
    # The matrix a has shape n x k where n is the number of samples and
    # k is the dimension
    def predict(self, a):
        a,_,_ = self.evaluate(a)
        return a.T

    # Train the network given the inputs x and the corresponding observations y
    # The network should be trained for numEpochs iterations using the supplied
    # optimizer
    def train(self, x, y, numEpochs, optimizer):

        # Initialize some stuff
        n = x.shape[0]
        hist = []
        optimizer.initialize(self.layers)
        
        # Run for the specified number of epochs
        for epoch in range(0,numEpochs):

            # Feed forward
            # Save the output of each layer in the list a
            # After the network has been evaluated, a should contain the
            # input x and the output of each layer except for the last layer
            yp, a, z = self.evaluate(x)

            # Compute the error
            C = self.cost.fx(yp,y.T)
            d = self.cost.dx(yp,y.T)
            grad = []

            # Backpropogate the error
            idx = len(self.layers)
            for layer, curZ in zip(reversed(self.layers),reversed(z)):
                idx = idx - 1
                # Here, we compute dMSE/dz_i because in the update
                # function for the optimizer, we do not give it
                # the z values we compute from evaluating the network
                grad.insert(0,np.multiply(d,layer.dx(curZ)))
                d = np.dot(layer.W.T,grad[0])

            # Update the errors
            optimizer.update(self.layers, grad, a)

            # Compute the error at the end of the epoch
            yh = self.predict(x)
            C = self.cost.fx(yh,y)
            C = np.mean(C)
            hist.append(C)
        return hist

    def trainBatch(self, x, y, batchSize, numEpochs, optimizer):

        # Copy the data so that we don't affect the original one when shuffling
        x = x.copy()
        y = y.copy()
        hist = []
        n = x.shape[0]
        
        for epoch in np.arange(0,numEpochs):
            
            # Shuffle the data
            r = np.arange(0,x.shape[0])
            x = x[r,:]
            y = y[r,:]
            e = []

            # Split the data in chunks and run SGD
            for i in range(0,n,batchSize):
                end = min(i+batchSize,n)
                batchX = x[i:end,:]
                batchY = y[i:end,:]
                e += self.train(batchX, batchY, 1, optimizer)
            hist.append(np.mean(e))

        return hist

if __name__ == '__main__':

    # Generate the training set
    np.random.seed(9001)
    x=np.random.uniform(-np.pi,np.pi,(1000,1))
    y=np.sin(x)
    xLin=np.linspace(-np.pi,np.pi,250).reshape((-1,1))
    yHats = {}

    activations = dict(ReLU=ReLUActivation,
                       tanh=TanhActivation,
                       linear=LinearActivation)
    lr = dict(ReLU=0.02,tanh=0.02,linear=0.005)
    names = ['ReLU','linear','tanh']

    for key in names:

        # Build the model
        activation = activations[key]
        model = Model(x.shape[1])
        model.addLayer(DenseLayer(100,activation()))
        model.addLayer(DenseLayer(100,activation()))
        model.addLayer(DenseLayer(1,LinearActivation()))
        model.initialize(QuadraticCost())

        # Train the model and display the results
        hist = model.train(x,y,500,GDOptimizer(eta=lr[key]))
        yHat = model.predict(x)
        yHats[key] = model.predict(xLin)
        error = np.mean(np.square(yHat - y))/2
        print(key+' MSE: '+str(error))
        plt.plot(hist)
        plt.title(key+' Learning curve')
        plt.show()

    # Plot the approximations
    font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
            'size'   : 12}
    matplotlib.rc('font', **font)
    y = np.sin(xLin)
    for key in activations:
        plt.plot(xLin,y)
        plt.plot(xLin,yHats[key])
        plt.title(key+' approximation')
        plt.savefig(key+'-approx.png')
        plt.show()

    # Train with different sized networks
    names = ['ReLU', 'tanh']
    sizes = [5,10,25,50]
    widths = [1,2,3]
    errors = {}
    y = np.sin(x)
    for key in names:
        error = []
        for width in widths:
            for size in sizes:
                activation = activations[key]
                model = Model(x.shape[1])
                for _ in range(width):
                    model.addLayer(DenseLayer(size,activation()))
                model.addLayer(DenseLayer(1,LinearActivation()))
                model.initialize(QuadraticCost())
                hist = model.train(x,y,500,GDOptimizer(eta=lr[key]))
                yHat = model.predict(x)
                yHats[key] = model.predict(xLin)
                e = np.mean(np.square(yHat - y))/2
                error.append(e)
        errors[key] = np.asarray(error).reshape((len(widths),len(sizes)))

    # Print the results
    for key in names:
        error = errors[key]
        print(key+' MSE Error')
        header = '{:^8}'
        for _ in range(len(sizes)):
            header += ' {:^8}'
        headerText = ['Layers'] + [str(s)+' nodes' for s in sizes]
        print(header.format(*headerText))
        for width,row in zip(widths,error):
            text = '{:>8}'
            for _ in range(len(row)):
                text += ' {:<8}'
            rowText = [str(width)] + ['{0:.5f}'.format(r) for r in row]
            print(text.format(*rowText))

    # Perform ridge regression on the last layer of the network
    # This is for part i
    print('\n----------------------------------------\n')
    print('Running ridge regression on last layer')
    from sklearn.linear_model import Ridge
    errors = {}
    for key in names:
        error = []
        for width in widths:
            for size in sizes:
                activation = activations[key]
                model = Model(x.shape[1])
                for _ in range(width):
                    model.addLayer(DenseLayer(size,activation()))
                model.initialize(QuadraticCost())
                ridge = Ridge(alpha=0.1)
                X = model.predict(x)
                ridge.fit(X,y)
                yHat = ridge.predict(X)
                e = np.mean(np.square(yHat - y))/2
                error.append(e)
        errors[key] = np.asarray(error).reshape((len(widths),len(sizes)))
                
    # Print the results
    for key in names:
        error = errors[key]
        print(key+' MSE Error')
        header = '{:^8}'
        for _ in range(len(sizes)):
            header += ' {:^8}'
        headerText = ['Layers'] + [str(s)+' nodes' for s in sizes]
        print(header.format(*headerText))
        for width,row in zip(widths,error):
            text = '{:>8}'
            for _ in range(len(row)):
                text += ' {:<8}'
            rowText = [str(width)] + ['{0:.5f}'.format(r) for r in row]
            print(text.format(*rowText))

    # Plot the results
    for key in names:
        for width,row in zip(widths,error):
            layer = ' layers'
            if width == 1:
                layer = ' layer'
            plt.semilogy(row,label=str(width)+layer)
        plt.title('MSE for ridge regression with '+key+' activation')
        plt.xticks(range(len(sizes)),sizes)
        plt.xlabel('Layer size')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(key+'-ridge.png')
        plt.show()

    # Test for SGD
    print('\n----------------------------------------\n')
    print('Using SGD')
    batchSizes = [1,10,100]
    for key in names:
        for batchSize in batchSizes:

            # Build the model
            activation = activations[key]
            model = Model(x.shape[1])
            model.addLayer(DenseLayer(50,activation()))
            model.addLayer(DenseLayer(50,activation()))
            model.addLayer(DenseLayer(1,LinearActivation()))
            model.initialize(QuadraticCost())
            
            # Train the model and display the results
            epochs = 25 * batchSize # Make sure that the same number of gradients steps are taken
            hist = model.trainBatch(x,y,batchSize,epochs,GDOptimizer(eta=lr[key]))
            yHat = model.predict(x)
            yHats[key] = model.predict(xLin)
            error = np.mean(np.square(yHat - y))/2
            print(key+'('+str(batchSize)+') MSE: '+str(error))
            plt.plot(hist)
            plt.title(key+' Learning curve')
            plt.show()
    
