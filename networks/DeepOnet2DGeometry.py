import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DeepONet(nn.Module):
    """
    DeepONet for 2D geometry problems.
    
    Args:
        num_branch_inputs (int): Number of inputs for the branch network.
        num_basis_functions (int): Number of basis functions for the trunk network.
        num_trunk_outputs (int): Number of outputs for the trunk network.
        num_hidden_layers (int): Number of hidden layers in the networks.
        hidden_dim (int): Dimension of the hidden layers.
    """
    
    def __init__(self, num_branch_inputs:int, num_basis_functions:int, num_trunk_inputs:int, branch_dims:list, trunk_dims:list):
        super(DeepONet, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.num_branch_inputs = num_branch_inputs
        self.num_trunk_inputs = num_trunk_inputs
        self.branch_dims = branch_dims
        self.trunk_dims = trunk_dims
        # Branch network
        self.branch_net = nn.Sequential(
            nn.Linear(num_branch_inputs, branch_dims[0]),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(branch_dims[i], branch_dims[i + 1]), nn.ReLU()) for i in range(len(branch_dims) - 1)],
            nn.Linear(branch_dims[-1], num_basis_functions)
        )
        
        # Trunk network
        self.trunk_net = nn.Sequential(
            nn.Linear(num_trunk_inputs, trunk_dims[0]),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(trunk_dims[i], trunk_dims[i + 1]), nn.ReLU()) for i in range(len(trunk_dims) - 1)],
            nn.Linear(trunk_dims[-1], num_basis_functions)
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
        #print("Branch output shape:", branch_output.shape)
        # Pass through trunk network
        trunk_output = self.trunk_net(trunk_input)
        #print("Trunk output shape:", trunk_output.shape)

        out = torch.sum(branch_output * trunk_output, -1) / math.sqrt(self.num_basis_functions) # peut être ajouter ,1 dans la somme pour que la somme se fasse sur les colonnes
        out = out.unsqueeze(1)

        return out
