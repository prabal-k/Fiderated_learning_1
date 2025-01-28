import flwr as fl
from typing import List, Tuple, Dict, Optional
import numpy as np
from flwr.common import Parameters, Scalar, FitRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# Define the custom strategy extending FedAvg
class AttentionWeightedAggregation(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        """
        Initialize the strategy with parameters controlling how many clients participate.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
        )
        self.attention_weights = {}  # Store attention weights for debugging or metrics

    def _calculate_attention_weights(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Dict[str, float]:
        """
        Calculate attention weights based on client performance metrics (e.g., accuracy).
        """
        accuracies = []  # Store accuracies reported by clients
        
        # Extract accuracy from each client's FitRes metrics
        for _, fit_res in results:
            metrics = fit_res.metrics
            accuracies.append(metrics.get("accuracy", 0.0))
            
        # Convert accuracies to a numpy array for computation
        accuracies = np.array(accuracies)
        
        # Apply softmax to calculate attention weights
        attention_scores = np.exp(accuracies) / np.sum(np.exp(accuracies))
        
        # Return attention weights as a dictionary
        return {str(i): float(score) for i, score in enumerate(attention_scores)}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model weights using the attention mechanism.
        """
        if not results:  # If no results are received, return None
            return None, {}
        
        # Calculate attention weights based on client accuracies
        self.attention_weights = self._calculate_attention_weights(results)
        
        # Extract weights and number of examples from each client
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Separate weights and example counts
        weights = [w for w, _ in weights_results]
        examples = [n for _, n in weights_results]
        
        # Initialize list to store weighted averages for each layer
        weighted_weights = []
        
        # Perform weighted aggregation layer by layer
        for layer_idx in range(len(weights[0])):
            # Collect updates for the current layer
            layer_updates = [w[layer_idx] for w in weights]
            
            # Compute the weighted sum using attention weights
            weighted_update = np.sum(
                [w * self.attention_weights[str(i)] for i, w in enumerate(layer_updates)],
                axis=0,
            )
            weighted_weights.append(weighted_update)  # Append to the aggregated weights
        
        # Aggregate custom metrics, such as accuracy
        metrics_aggregated = {}
        for _, fit_res in results:
            for key, value in fit_res.metrics.items():
                if key in metrics_aggregated:
                    metrics_aggregated[key] += value
                else:
                    metrics_aggregated[key] = value
        
        # Average the aggregated metrics
        for key in metrics_aggregated:
            metrics_aggregated[key] /= len(results)
        
        # Include attention weights in the aggregated metrics for debugging
        metrics_aggregated["attention_weights"] = str(self.attention_weights)
        
        # Convert the aggregated weights back to Parameters format
        return ndarrays_to_parameters(weighted_weights), metrics_aggregated

def main():
    # Instantiate the custom strategy
    strategy = AttentionWeightedAggregation(
        min_fit_clients=2,  # Minimum number of clients required for training
        min_available_clients=2,  # Minimum number of clients that should be connected
        min_evaluate_clients=2,  # Minimum number of clients required for evaluation
    )

    # Start the server with the custom strategy
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Server address
        config=fl.server.ServerConfig(num_rounds=5),  # Configure number of rounds
        strategy=strategy,  # Use the custom strategy
    )

if __name__ == "__main__":
    main()  # Run the server
