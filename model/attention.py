import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.attention_dim = attention_dim
        # self.W_K = torch.randn(embedding_dim, attention_dim)
        # self.W_Q = torch.randn(embedding_dim, attention_dim)
        # self.W_V = torch.randn(embedding_dim, attention_dim)
        self.W_K = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_Q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_V = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        
        # self.K = embedded @ self.W_K
        # self.Q = embedded @ self.W_Q
        # self.V = embedded @ self.W_V
        self.K = self.W_K(embedded)
        self.Q = self.W_Q(embedded)
        self.V = self.W_V(embedded)
        self.atten_score = (self.Q @ self.K.transpose(1, 2)) / (self.attention_dim ** 0.5)
        
        self.causal_mask = torch.tril(torch.ones(self.atten_score.shape))
        self.masked_atten_score = torch.where(self.causal_mask == 0, float('-inf'), self.atten_score)
        
        self.softmax_out = self.softmax(self.masked_atten_score)
        # return torch.tensor(self.atten_score.shape)

        self.Z = self.softmax_out @ self.V
        return self.Z