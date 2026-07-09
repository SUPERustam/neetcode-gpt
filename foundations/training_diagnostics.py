import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        out = x
        stats = []
        with torch.no_grad():
            for name, layer in model.named_children():
                out = layer(out)
                if isinstance(layer, nn.Linear):
                    mean = torch.mean(out).item()
                    std = torch.std(out).item()

                    mask = out.le(0)
                    dead_fraction = mask.all(dim=0).float().mean().item()

                    stats.append({"mean": round(mean, 4), "std": round(std, 4), "dead_fraction": round(dead_fraction, 4)})
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        out = x
        stats = []

        criterion = nn.MSELoss()

        prediction = model(x)
        loss = criterion(prediction, y)

        model.zero_grad()
        loss.backward()

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                if layer.weight.grad is not None:
                    grad = layer.weight.grad
                
                    stats.append({"mean": round(grad.mean().item(), 4), "std": round(grad.std().item(), 4), "norm": round(torch.norm(grad).item(), 4)})
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for i in range(len(activation_stats)):
            if activation_stats[i]['dead_fraction'] > 0.5:
                return "dead_neurons"
            if gradient_stats[i]['norm'] > 1000:
                return "exploding_gradients"
            if gradient_stats[i]['norm'] < 1e-5 or activation_stats[i]['std'] < 0.1:
                return "vanishing_gradients"
            if activation_stats[i]['std'] > 10.0:
                return "exploding_gradients"
        return 'healthy'

        
