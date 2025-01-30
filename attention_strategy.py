import flwr as fl
from typing import List, Tuple, Dict, Optional
import numpy as np
from flwr.common import Parameters, Scalar, FitRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class AttentionWeightedAggregation(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
        )
        self.attention_weights = {}
        
        # Initialize metrics storage
        self.client_metrics = defaultdict(lambda: defaultdict(list))
        self.rounds = []
        
        # Set up plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 5]

    def _calculate_attention_weights(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Dict[str, float]:
        accuracies = []
        for _, fit_res in results:
            metrics = fit_res.metrics
            accuracies.append(metrics.get("accuracy", 0.0))
        accuracies = np.array(accuracies)
        attention_scores = np.exp(accuracies) / np.sum(np.exp(accuracies))
        return {str(i): float(score) for i, score in enumerate(attention_scores)}

    def plot_training_metrics(self, server_round: int):
        """Plot loss and accuracy trends"""
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        # Plot Loss
        for client_id in self.client_metrics.keys():
            losses = self.client_metrics[client_id]['loss']
            rounds = range(1, len(losses) + 1)
            sns.lineplot(
                x=rounds,
                y=losses,
                label=f'{client_id} Loss',
                marker='o',
                ax=ax1
            )
        ax1.set_title('Training Loss per Round')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')
        
        # Plot Accuracy
        for client_id in self.client_metrics.keys():
            accuracies = self.client_metrics[client_id]['accuracy']
            rounds = range(1, len(accuracies) + 1)
            sns.lineplot(
                x=rounds,
                y=accuracies,
                label=f'{client_id} Accuracy',
                marker='o',
                ax=ax2
            )
        ax2.set_title('Training Accuracy per Round')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(f'training_metrics_round_{server_round}.png')
        plt.close()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        # Calculate attention weights and store metrics
        self.attention_weights = self._calculate_attention_weights(results)
        
        # Store metrics for each client
        for idx, (_, fit_res) in enumerate(results):
            client_id = f'client_{idx+1}'
            self.client_metrics[client_id]['accuracy'].append(fit_res.metrics.get("accuracy", 0.0))
            # Assuming loss is also provided in metrics
            self.client_metrics[client_id]['loss'].append(fit_res.metrics.get("loss", 0.0))
        
        # Generate plots every round
        self.plot_training_metrics(server_round)
        
        # Extract and aggregate weights
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        weights = [w for w, _ in weights_results]
        
        # Initialize aggregated weights
        weighted_weights = []
        
        # Aggregate weights using attention mechanism
        for layer_idx in range(len(weights[0])):
            layer_updates = [w[layer_idx] for w in weights]
            weighted_update = np.sum(
                [w * self.attention_weights[str(i)] for i, w in enumerate(layer_updates)],
                axis=0,
            )
            weighted_weights.append(weighted_update)
        
        # Aggregate metrics
        metrics_aggregated = {}
        for _, fit_res in results:
            for key, value in fit_res.metrics.items():
                if key in metrics_aggregated:
                    metrics_aggregated[key] += value
                else:
                    metrics_aggregated[key] = value
        
        for key in metrics_aggregated:
            metrics_aggregated[key] /= len(results)
        
        metrics_aggregated["attention_weights"] = str(self.attention_weights)
        
        return ndarrays_to_parameters(weighted_weights), metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics and plot them"""
        if not results:
            return None, {}
        
        # Store evaluation metrics
        for idx, (_, eval_res) in enumerate(results):
            client_id = f'client_{idx+1}'
            if eval_res.loss is not None:
                self.client_metrics[client_id]['eval_loss'].append(eval_res.loss)
            if 'accuracy' in eval_res.metrics:
                self.client_metrics[client_id]['eval_accuracy'].append(
                    eval_res.metrics['accuracy']
                )
        
        # Calculate aggregated metrics
        loss_aggregated = sum(eval_res.loss for _, eval_res in results) / len(results)
        metrics_aggregated = {}
        
        for _, eval_res in results:
            for key, value in eval_res.metrics.items():
                if key in metrics_aggregated:
                    metrics_aggregated[key] += value
                else:
                    metrics_aggregated[key] = value
        
        for key in metrics_aggregated:
            metrics_aggregated[key] /= len(results)
        
        return loss_aggregated, metrics_aggregated

def main():
    strategy = AttentionWeightedAggregation(
        min_fit_clients=2,
        min_available_clients=2,
        min_evaluate_clients=2,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()