import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        output = torch.normal(mean=0, std=math.sqrt(2 / (fan_in + fan_out)), size=(fan_out, fan_in))
        return torch.round(output, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        output = torch.normal(mean=0, std=math.sqrt(2 / (fan_in)), size=(fan_out, fan_in))
        return torch.round(output, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.      
        torch.manual_seed(0)
        
        fan_in = input_dim
        fan_out = hidden_dim

        std_list = []
        weights = []
        for i in range(num_layers):
            match init_type: 
                case "kaiming":
                    w = torch.normal(mean=0, std=math.sqrt(2 / (fan_in)), size=(fan_out, fan_in))
                case "xavier":
                    w = torch.normal(mean=0, std=math.sqrt(2 / (fan_in + fan_out)), size=(fan_out, fan_in))
                case "random":
                    w = torch.randn(fan_out, fan_in)
            weights.append(w)
            fan_in = hidden_dim
    
        x = torch.randn(input_dim)
        for i in range(num_layers):  
            x = weights[i] @ x
            x = torch.relu(x)
            std_list.append(round(x.std().item(), 2))
        return std_list