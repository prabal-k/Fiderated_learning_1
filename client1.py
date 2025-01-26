import flwr as fl
import tensorflow as tf # TensorFlow, which is used for building and training neural networks.
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# AUxillary methods
def getDist(y):
    pd.value_counts(y).plot(kind="bar")
    # ax = sns.barplot(y)
    # ax.set(title="Count of data classes")
    plt.show()

#: This function(purpose : To make data non IID data) selects a subset of the dataset based on the specified distribution (dist). It ensures that each class is represented according to the distribution defined in dist. x is the input data (images), and y is the corresponding labels (class labels).
def getData(dist, x, y):
    dx = []
    dy = []
    counts = [0 for i in range(10)]
    for i in range(len(x)):
        if counts[y[i]]<dist[y[i]]:
            dx.append(x[i])
            dy.append(y[i])
            counts[y[i]] += 1
        
    return np.array(dx), np.array(dy)

# Load and compile Keras model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),  # Flattens the 28x28 MNIST images into a 1D vector.
    keras.layers.Dense(128, activation='relu'),     #A fully connected layer with 128 neurons and ReLU activation.
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')    # The output layer with 10 neurons (one for each class) and softmax activation (for multi-class classification).
])

#compiles the model, specifying the optimizer (adam), loss function (sparse_categorical_crossentropy for multi-class classification), and evaluation metric (accuracy).
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#normalizes the images by scaling pixel values to the range [0, 1] and adds a new axis to make the data 4D (batch_size, height, width, channels).
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
#This defines the distribution of classes in the training data for federated learning. For example, class 0 will have 4000 samples, and class 4,5,6,7 will have only 10 samples.
dist = [4000, 4000, 4000, 3000, 10, 10, 10, 10, 4000, 10]
x_train, y_train = getData(dist, x_train, y_train)
getDist(y_train)
  

# defines the FlowerClient, which communicates with the Flower server. It implements the NumPyClient interface, which is used for federated learning with NumPy arrays.
class FlowerClient(fl.client.NumPyClient):

    #This method returns the current model's weights, which are sent to the server for aggregation.
    #The server uses this method to initialize the global model by collecting the parameters (weights) from one or more clients.
    def get_parameters(self, config):
        # Your existing logic to retrieve the model parameters
        return model.get_weights()   #The client sends its model weights to the server.
    
    #This method performs one round of local training using the model's weights provided by the server. It updates the model's weights by training on the local data (x_train, y_train) and returns the updated weights.
    # Purpose OF FIT() method: To load the global model parameters, train the local model on the client’s dataset, and send the updated parameters to the server.
    def fit(self, parameters, config):
        model.set_weights(parameters)       #The global model parameters are loaded into the client's model 
        r = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0) ## Train locally
        hist = r.history            # The training performance (loss, accuracy) is stored in r.history and printed:
        print("Fit history : ", hist)
        return model.get_weights(), len(x_train), {} # # Return  locally updated model weights to the server

    #This method evaluates the model on the test data (x_test, y_test) and returns the loss and accuracy after each round(there are 3 rounds currently).
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Eval accuracy : ", accuracy)  # Measure the global models performance on the test data (which is IID i.e contains all class equally)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), #  Specifies the server's address and port to connect (e.g., localhost:8080)
        client=FlowerClient(),  #FlowerClient instance that implements federated learning methods.
        grpc_max_message_length = 1024*1024*1024  #Sets the maximum message size for gRPC communication(messages and weights are shared between client and server using grpc).
)