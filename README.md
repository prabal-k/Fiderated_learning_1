# Flow of the Project: Server Side (Using Federated Averaging)

## 1. Server Initialization

server starts using fl.server.start_server with the FedAvg strategy.

a. The FedAvg strategy ensures that client updates (model parameters) are averaged to create the global model.

b. the number of rounds for training in the server configuration is set to 3 rounds.

Code Example:

fl.server.start_server(
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

a) get_parameters() → Fetches the global model weights from the server.

b) fit() → Trains the model locally on the client's data.

c) evaluate() → Evaluates the updated model on the client's test data.

Key Code in Client:

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()  # Return the current weights of the model.

    def fit(self, parameters, config):
        model.set_weights(parameters)  # Load global model weights.
        model.fit(x_train, y_train, epochs=1, batch_size=32)  # Train locally.
        return model.get_weights(), len(x_train), {}  # Return updated weights.

    def evaluate(self, parameters, config):
        model.set_weights(parameters)  # Load updated global model weights.
        loss, accuracy = model.evaluate(x_test, y_test)  # Evaluate locally.
        return loss, len(x_test), {"accuracy": accuracy}
        
## 2. Client Workflow

During each round:

a) get_parameters(): The client receives the global model weights from the server.

b) fit(): It updates its local model with the global weights and trains it using its local dataset.

c) evaluate(): After training, it evaluates the model using its own test data (if required by the server).

## 3.Sending Updates

The client sends:

Updated model weights (from fit()).

Evaluation metrics (from evaluate()).
