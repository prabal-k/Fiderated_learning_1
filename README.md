![Image](https://github.com/user-attachments/assets/012056fb-79b5-436d-8dac-1fe22915f106)


# Flow of the Project: Server Side (Using Federated Averaging)

## 1. Server Initialization

server starts using fl.server.start_server with the FedAvg strategy.

a. The FedAvg strategy ensures that client updates (model parameters) are averaged to create the global model.

b. the number of rounds for training in the server configuration is set to 3 rounds.

Code Example:


fl.server.start_server
(

    server_address="localhost:8080",
    
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg()
)

## 2. Round Workflow

For each round:

a) The server sends the global model parameters to selected clients.

b) It waits for client updates (weights after local training).

c) It aggregates these updates using the Federated Averaging (FedAvg) strategy to create a new global model.

Optionally, the server may evaluate the new global model using client-provided evaluation results.

### Key Points:

The server does not need access to client data.It coordinates the process and ensures that updates flow seamlessly between clients and itself.


# Flow of the Project: Client Side

## 1. Initialization

Each client connects to the server and implements three key methods:

a) get_parameters() → is used to return the local model's current weights whenever the server requests them.

b) fit() → First load the global model weigths, Trains the model locally on the client's data and returns the trained local model weigths to server

c) evaluate() → Evaluates the updated model on the client's test data.

Key Code in Client:


class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self):    #The server uses this method to initialize the global model by collecting the parameters (weights) from clients and perform fed-averaging
        return model.get_weights()      


    def fit(self, parameters, config):   #Purpose OF FIT() method: To load the global model parameters, train the local model on the client’s dataset, and send the updated parameters to the server.

        model.set_weights(parameters)  # Load global model weights.
        model.fit(x_train, y_train, epochs=1, batch_size=32)  # Train locally.
        return model.get_weights(), len(x_train), {}  # Return updated weights.

    def evaluate(self, parameters, config):  # This method evaluates the model on the test data (x_test, y_test) and returns the loss and accuracy after each round(there                                                                                             are 3 rounds currently).

        model.set_weights(parameters)  # Load updated global model weights.
        loss, accuracy = model.evaluate(x_test, y_test)  # Evaluate locally.
        return loss, len(x_test), {"accuracy": accuracy}
        
## 2.Sending Updates

The client sends:

Updated model weights (from fit()).

Evaluation metrics (from evaluate()).



# High-Level Summary

Server:

Starts the process and coordinates communication.

Aggregates client updates using FedAvg.

Creates a new global model after each round.

Clients:

Train the model locally on their private data.

Share only model updates (weights) with the server, ensuring data privacy.
