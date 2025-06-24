import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DeepONet2DGeometry(nn.Module):
    """
    DeepONet for 2D geometry problems.
    
    Args:
        num_branch_inputs (int): Number of inputs for the branch network.
        num_basis_functions (int): Number of basis functions for the trunk network.
        num_trunk_outputs (int): Number of outputs for the trunk network.
        num_hidden_layers (int): Number of hidden layers in the networks.
        hidden_dim (int): Dimension of the hidden layers.
    """
    
    def __init__(self, num_branch_inputs:int, num_basis_functions:int, num_trunk_inputs:int, num_hidden_layers=3, hidden_dim=64):
        super(DeepONet2DGeometry, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.num_branch_inputs = num_branch_inputs
        self.num_trunk_inputs = num_trunk_inputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        # Branch network
        self.branch_net = nn.Sequential(
            nn.Linear(num_branch_inputs, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_hidden_layers - 1)],
            nn.Linear(hidden_dim, num_basis_functions)
        )
        
        # Trunk network
        self.trunk_net = nn.Sequential(
            nn.Linear(num_trunk_inputs, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_hidden_layers - 1)],
            nn.Linear(hidden_dim, num_basis_functions)
        )

    def forward(self, branch_input, trunk_input):
        """
        Forward pass for the DeepONet.

        Args:
            branch_input (torch.Tensor): Input tensor for the branch network.
            trunk_input (torch.Tensor): Input tensor for the trunk network.

        Returns:
            torch.Tensor: Output of the DeepONet.
        """
        # Pass through branch network
        branch_output = self.branch_net(branch_input)

        # Pass through trunk network
        trunk_output = self.trunk_net(trunk_input)

        out = torch.sum(branch_output * trunk_output, 1) / math.sqrt(self.num_basis_functions) # peut être ajouter ,1 dans la somme pour que la somme se fasse sur les colonnes
        out = out.unsqueeze(1)

        return out
