
import numpy as np
import random
from mnist_loader import load_mnist

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1.0-sigmoid(z))

def softmax(y):
    return np.exp(y)/np.sum(np.exp(y))

DATA_FILE = 'mnist.pkl.gz'
LAYER_SIZES = [784, 100, 10]
NUM_LAYERS = len(LAYER_SIZES)-1
TRAINING_ITER = 50000
ETA = 0.05

training_data, validation_data, test_data = load_mnist(data_path=DATA_FILE)


weights = [np.random.randn(LAYER_SIZES[i+1], LAYER_SIZES[i]) for i in range(NUM_LAYERS)]
biases = [np.zeros((LAYER_SIZES[i+1], 1)) for i in range(NUM_LAYERS)]

### TRAINING ###
for iter in range(TRAINING_ITER):
    x, y = training_data[random.randint(0, len(training_data)-1)]
    x = x.reshape(x.size, 1)
    zs, activations = [], []
    # FORWARD PASS
    # Store the input activations to each layer, as well as the resulting z values
    activation = x
    for i in range(NUM_LAYERS):
        activations.append(activation)
        z = np.dot(weights[i], activation) + biases[i]
        activation = sigmoid(z)
        zs.append(z)
    p = softmax(activation)
    delta = p - y
    # BACK PROPAGATION
    for i in range(NUM_LAYERS)[::-1]:
        dL_dz = delta*sigmoid_prime(zs[i])
        weights[i] -= ETA * np.dot(dL_dz, activations[i].T)
        biases[i] -= ETA * dL_dz
        delta = np.dot(weights[i].T, dL_dz)


### VALIDATION ###
num_correct = 0.0
for x, y in validation_data:
    x = x.reshape(x.size, 1)
    activation = x
    for i in range(NUM_LAYERS):
        activation = sigmoid(np.dot(weights[i], activation) + biases[i])
    if np.argmax(activation) == y:
        num_correct += 1 

print(num_correct)
print("NUM CORRECT: " + str(num_correct / len(validation_data)))
    
        






