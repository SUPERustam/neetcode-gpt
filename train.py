import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            torch.manual_seed(epoch)
            starts = torch.randint(data.shape[0] - context_length, (batch_size,))
            x = torch.stack([data[start:start + context_length] for start in starts])
            y = torch.stack([data[start + 1:1 + start + context_length] for start in starts])

            logits = model(x)
            logits_flat = logits.view(-1, logits.shape[-1])
            y_flat = y.ravel()
            loss = F.cross_entropy(logits_flat, y_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return round(loss.item(), 4)
            
            