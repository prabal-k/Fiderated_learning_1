import flwr as fl  # imports the Flower library, used for federated learning
import sys  #allows us to work with command-line arguments (e.g., specifying the server port).
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

# Create strategy and run server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning (The server will automatically invoke get_parameters(), fit(), and evaluate() based on the FedAvg strategy and the configured number of rounds.)
fl.server.start_server(
        server_address = 'localhost:'+str(sys.argv[1]) , 
        config=fl.server.ServerConfig(num_rounds=3) ,
        grpc_max_message_length = 1024*1024*1024,   #Grpc sends commands and messages to clients and also this message length will carry the weights if the global model
        strategy = strategy
)