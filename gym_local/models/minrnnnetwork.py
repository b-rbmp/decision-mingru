import torch
from torch import nn
from torch.nn import functional as F

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gym_local.models.mingru import MinGRU
from gym_local.models.minlstm import MinLSTM

class MinRNNNetwork(nn.Module):
    """
    Minimal RNN network for the selective copying task
    
    Args:
        vocab_size: Number of embeddings
        embedding_dim: Dimensionality of input features
        num_layers: Number of layers
        expansion_factor: Factor by which to expand hidden size
        dropout: Dropout ratio
        rnn_architecture: RNN architecture to use (minGRU or minLSTM)
    """
    def __init__(
        self, vocab_size: int, embedding_dim=64, num_layers=3, expansion_factor=2, dropout=0.1, rnn_architecture="minGRU"
    ):
        super(MinRNNNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout_ratio = dropout
        self.vocab_size = vocab_size
        self.expansion_factor = expansion_factor
        self.rnn_architecture = rnn_architecture

        # Define embedding layer for input tokens (vocab_size -> embedding_dim)
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )

        # Define layers. Each layer consists of a layer norm, minGRU, linear layer and dropout.
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict(
                {
                    "rnn": MinGRU(
                        input_dim=embedding_dim, expansion_factor=expansion_factor
                    ) if rnn_architecture == "minGRU" else MinLSTM(
                        input_dim=embedding_dim, expansion_factor=expansion_factor
                    ),
                    "linear": nn.Linear(embedding_dim, embedding_dim),
                    "dropout": nn.Dropout(dropout),
                }
            )
            self.layers.append(layer)

        # Output layer to project back to embedding number
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network

        Args:
            x: Input tensor of shape [batch_size, seq_len]

        Returns:
            logits: Output logits of shape [batch_size, seq_len, input_dim]
        """
        # Initial hidden state for minGRU
        h = None
        # Convert input tokens to embeddings with shape [batch_size, seq_len, embedding_dim]
        x = self.embedding(x) 

        # Forward pass through layers with residual connections
        for layer in self.layers:
            residual = x
            x, h = layer["rnn"](x, prev_hidden=h)
            x = layer["linear"](x)
            x = layer["dropout"](x)
            x = x + residual
        
        # Output layer, shape: [batch_size, seq_len, input_dim]
        logits = self.output_layer(x) 
        return logits