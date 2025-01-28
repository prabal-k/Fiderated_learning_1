import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import numpy as np

def create_cnn():
    model = keras.Sequential([
        keras.layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = create_cnn()
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # Split data for this client
        self.x_train = x_train[:30000, ..., np.newaxis]/255.0
        self.y_train = y_train[:30000]
        self.x_test = x_test[..., np.newaxis]/255.0
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.x_train, 
            self.y_train,
            epochs=1,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
            validation_data=(self.x_test, self.y_test)

        )
        performance = history.history            # The training performance (loss, accuracy) is stored in r.history and printed:
        print("Fit history : ", performance)
        return self.model.get_weights(), len(self.x_train), {
            "accuracy": float(history.history['accuracy'][-1])
        }

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Eval accuracy : ", accuracy)

        return loss, len(self.x_test), {"accuracy": accuracy}

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address=f"127.0.0.1:{sys.argv[1]}",
        client=FlowerClient(),
    )