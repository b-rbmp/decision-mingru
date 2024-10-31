from typing import Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

class MinLSTM(nn.Module):
    """
    Minimal Parallel LSTM implementation using log-space computations
    for efficient parallel processing of sequences.

    Args:
        input_dim: Dimensionality of input features
        expansion_factor: Factor by which to expand hidden size
    """
        
    def __init__(self, input_dim: int, expansion_factor: int = 1):
        super(MinLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = expansion_factor * input_dim

        self.projector = nn.Linear(self.input_dim, self.hidden_dim*3, bias=False)

        # Output projection only needed if hidden size differs from input size
        self.output_transform = nn.Linear(self.hidden_dim, self.input_dim, bias=False) if self.input_dim != self.hidden_dim else nn.Identity()

    def forward(self, inputs: torch.Tensor, prev_hidden: Union[None, torch.Tensor] = None) -> Tuple[torch.Tensor, Union[None, torch.Tensor]]:
        """
        Forward pass of the minLSTM model

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, input_size]
            prev_hidden: Previous hidden state tensor of shape [batch_size, 1, hidden_size]

        Returns:
            Tuple of output tensor and next hidden state tensor
        """
        _, seq_len, _ = inputs.shape

        # Compute gates and transforms
        log_forget, log_input, log_hidden = self._compute_gate_parameters(inputs)

        log_candidate = log_input + log_hidden

        # Handle previous state if provided
        if prev_hidden is not None:
            log_prev_hidden = self._activation(prev_hidden, in_log_space=True)
            log_values = torch.cat([log_prev_hidden, log_candidate], dim=1)
            log_forget = F.pad(log_forget, (0, 0, 1, 0))
        else:
            log_values = log_candidate

        # Parallel scan computation
        next_hidden = self._parallel_prefix_sum(log_forget, log_values)
        next_hidden = next_hidden[:, -seq_len:]

        # Prepare outputs
        outputs = self.output_transform(next_hidden)
        new_hidden = next_hidden[:, -1:, :]

        return outputs, new_hidden
    
    def _compute_gate_parameters(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process inputs through gate projections and compute log-space parameters

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, input_size]

        Returns:
            Tuple of log-space forget, update, and hidden gate parameters
        """
        # Obtain gate parameters and split into forget, input, and hidden gates
        hidden, input_gate, forget_gate = self.projector(inputs).chunk(3, dim=-1)

        # Calculate Differece between forget and input gate in log space
        diff = F.softplus(-forget_gate) - F.softplus(-input_gate)
        
        # Compute log space parameters
        log_forget = -F.softplus(diff)
        log_input = -F.softplus(-diff)
        log_hidden = self._activation(hidden, in_log_space=True)

        return log_forget, log_input, log_hidden


    @staticmethod
    def _parallel_prefix_sum(log_coeffs: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
        """
        Compute parallel prefix sum in log space using Heinsen's method

        Args:
            log_coeffs: Log-space coefficients tensor of shape [batch_size, seq_len, input_size]
            log_values: Log-space values tensor of shape [batch_size, seq_len + 1, input_size]

        Returns:
            Log-space hidden state tensor of shape [batch_size, seq_len, input_size]
        """
        cumulative_sum = torch.cumsum(log_coeffs, dim=1)
        compensated_values = log_values - cumulative_sum
        accumulated_values = torch.logcumsumexp(compensated_values, dim=1)
        log_h = cumulative_sum + accumulated_values
        return torch.exp(log_h)

    @staticmethod
    def _activation(x: torch.Tensor, in_log_space: bool = False) -> torch.Tensor:
        """
        Compute activation function, optionally in log space

        Args:
            x: Input tensor
            in_log_space: Whether to compute in log space

        Returns:
            Output tensor with activation applied
        """
        if in_log_space:
            return torch.where(x >= 0, (F.relu(x)+0.5).log(),-F.softplus(-x))
        else:
            return torch.where(x >= 0, x+0.5, torch.sigmoid(x))
        




    

    