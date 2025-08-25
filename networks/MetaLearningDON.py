import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DeepONet(nn.Module):
    def __init__(
            self,
            num_branch_inputs:int,
            num_basis_functions:int, 
            num_trunk_inputs:int, 
            branch_dims:list, 
            trunk_dims:list, 
            latent_dim:int,
            dropout:list=None, 
            dropout_prob:float=0.0, 
            norm_layers:tuple=(), 
            latent_in:tuple=(),
            weight_norm:bool=False,
            bias:bool=False):

        super(DeepONet, self).__init__()

        self.num_branch_inputs = num_branch_inputs
        self.num_basis_functions = num_basis_functions
        self.num_trunk_inputs = num_trunk_inputs
        self.branch_dims = branch_dims
        self.trunk_dims = trunk_dims
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.latent_in = latent_in
        self.weight_norm = weight_norm
        self.latent_dim = latent_dim

        self.branch_norm_layers, self.trunk_norm_layers = norm_layers

        branch_dims = [num_branch_inputs] + branch_dims + [num_basis_functions]
        trunk_dims = [num_trunk_inputs] + trunk_dims + [num_basis_functions]

        self.num_branch_layers = len(branch_dims) - 1
        self.num_trunk_layers = len(trunk_dims) - 1

        # Modulator for the trunk network _________________________________________________________________________________________________________________

        self.Modulator = nn.Linear(
            latent_dim,
            sum(trunk_dims[1:])
        )

        if dropout != ():
            if len(dropout) != 2:
                raise ValueError("Dropout must be a tuple of two lists: (branch_dropout, trunk_dropout)")
            self.branch_dropout, self.trunk_dropout = dropout

        # Branch network ________________________________________________________________________________________________________________
        branch_layers = []

        for layer in range(self.num_branch_layers):

            out_dim = branch_dims[layer + 1]
            """
            if layer != self.num_branch_layers - 1:
                out_dim -= num_basis_functions
            """
            if layer in self.branch_dropout:
                branch_layers.append(nn.Dropout(dropout_prob))

            if weight_norm and layer in self.branch_norm_layers:
                branch_layers.append(nn.utils.weight_norm(nn.Linear(branch_dims[layer], out_dim)))
            else:
                branch_layers.append(nn.Linear(branch_dims[layer], out_dim))

            if layer in self.branch_norm_layers:
                branch_layers.append(nn.LayerNorm(out_dim))
            branch_layers.append(nn.ReLU())

        self.branch_net = nn.Sequential(*branch_layers)

        # Trunk network ________________________________________________________________________________________________________________
        trunk_layers = []

        for layer in range(self.num_trunk_layers):
            if layer in self.trunk_dropout:
                trunk_layers.append(nn.Dropout(dropout_prob))
            if layer + 1 in self.latent_in:
                out_dim = trunk_dims[layer + 1] - trunk_dims[0]
            else:
                out_dim = trunk_dims[layer + 1]
                """
                if layer != self.num_trunk_layers - 1:
                    out_dim -= num_basis_functions
                """
            if weight_norm and layer in self.trunk_norm_layers:
                trunk_layers.append(nn.utils.weight_norm(nn.Linear(trunk_dims[layer], out_dim)))
            else:
                trunk_layers.append(nn.Linear(trunk_dims[layer], out_dim))

            if layer in self.trunk_norm_layers:
                trunk_layers.append(nn.LayerNorm(out_dim))
            trunk_layers.append(nn.ReLU())

        self.trunk_net = nn.Sequential(*trunk_layers)
        self.trunk_lin_idx = [i for i, layer in enumerate(self.trunk_net) if isinstance(layer, nn.Linear)]
        self.latent_in = [self.trunk_lin_idx[i] for i in self.latent_in]

        # ________________________________________________________________________________________________________________

        if bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.bias = 0

    def forward(self, branch_input, trunk_input):
        """
        Forward pass for the DeepONet.

        Args:
            branch_input (torch.Tensor): Input tensor for the branch network.
            trunk_input (torch.Tensor): Input tensor for the trunk network.

        Returns:
            torch.Tensor: Output of the DeepONet.
        """

        trunk_input, latent_vect = trunk_input[:, :,-2:], trunk_input[:, :, :-2] # two last columns are coordinates, the rest is the latent vector
        #print("Trunk input shape:", trunk_input.shape)
        #print("Latent vector shape:", latent_vect.shape)

        original_trunk_input = trunk_input.clone()  # Save the original trunk input for later use

        modulation = self.Modulator(latent_vect)  

        # Pass through branch network
        branch_output = self.branch_net(branch_input)
        #print("Branch output shape:", branch_output.shape)
        # Pass through trunk network
        for layer in range(len(self.trunk_net)):
            if layer in self.latent_in:
                #print(trunk_input.shape, original_trunk_input.shape)
                trunk_input = self.trunk_net[layer](torch.cat((trunk_input, original_trunk_input), dim=-1))
            else:
                trunk_input = self.trunk_net[layer](trunk_input)
            
            if layer in self.trunk_lin_idx:
                trunk_input = trunk_input + modulation[:, :, sum(self.trunk_dims[1:self.trunk_lin_idx.index(layer) + 1]): sum(self.trunk_dims[1:self.trunk_lin_idx.index(layer) + 1]) + trunk_input.shape[-1]]
            

        #trunk_output = self.trunk_net(trunk_input)
        #print("Trunk output shape:", trunk_output.shape)

        out = torch.sum(branch_output * trunk_input, -1) / math.sqrt(self.num_basis_functions) # peut être ajouter ,1 dans la somme pour que la somme se fasse sur les colonnes
        out = out.unsqueeze(1)
        out = out + self.bias

        return out

    def trunk_forward(self, trunk_input, modulation=None):
        """
        Forward pass for the trunk network.

        Args:
            trunk_input (torch.Tensor): Input tensor for the trunk network.

        Returns:
            torch.Tensor: Output of the trunk network.
        """
        original_trunk_input = trunk_input.clone()  # Save the original trunk input for later use

        for layer in range(len(self.trunk_net)):
            if layer in self.latent_in:
                #print(trunk_input.shape, original_trunk_input.shape)
                trunk_input = self.trunk_net[layer](torch.cat((trunk_input, original_trunk_input), dim=-1))
            else:
                trunk_input = self.trunk_net[layer](trunk_input)
            
            if layer in self.trunk_lin_idx and modulation is not None:
                trunk_input = trunk_input + modulation[:, :, sum(self.trunk_dims[1:self.trunk_lin_idx.index(layer) + 1]): sum(self.trunk_dims[1:self.trunk_lin_idx.index(layer) + 1]) + trunk_input.shape[-1]]
        
        return trunk_input
            

