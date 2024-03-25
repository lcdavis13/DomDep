# Architecture for domain-dependent mixture of linear layers, partially generalized for the specific case of residual stream a la Elhage et al 2021:
# Input and output have same dimensionality, but the module mixture occurs with different dimension, presumably a subspace (module_dim << embed_dim).
# Linear layers are used to encode and decode the residual stream space into and out of the module space.
# Future work: investigate if the decoding matrix can be the inverse of the encoding matrix instead of separate parameters.

import torch
import torch.nn as nn
import torch.nn.functional as F

class DomDepResidualLayer(nn.Module):
    def __init__(self, embed_dim, module_dim, module_count):
        super(DomDepResidualLayer, self).__init__()
        self.embed_dim = embed_dim
        self.module_dim = module_dim
        self.module_count = module_count

        # Weight matrix for encoding/compressing into module space
        self.we0 = nn.Linear(embed_dim, module_dim)

        # Weight matrix for computing soft weight of each module (domain dependence)
        self.ws = nn.Linear(module_dim, module_count)
        
        # 3D weight matrix for computing each module's activations
        self.wm = nn.ModuleList([nn.Linear(module_dim, module_dim) for _ in range(module_count)])

        # Weight matrix for decoding into the space of the residual stream
        self.we1 = nn.Linear(module_dim, embed_dim)

    def forward(self, h0):
        # Encode into module space
        h0_encoded = self.we0(h0)
        
        # Compute module activations
        module_activations = torch.stack([m(h0_encoded) for m in self.wm])
        module_activations = F.relu(module_activations)

        # Compute module soft weights
        module_weights = self.ws(h0_encoded)
        module_weights = F.softmax(module_weights, dim=0)

        # Mix modules based on domain-dependent soft weights
        mixed_activations = torch.matmul(module_activations.transpose(0, 1), module_weights.unsqueeze(-1)).squeeze(-1)
        
        # Decode and skip connection
        decoded_out = self.we1(mixed_activations)
        h1 = F.relu(decoded_out + h0)
        return h1

if __name__ == "__main__":
    embed_dim = 4
    module_dim = 3
    module_count = 2
    layer = DomDepResidualLayer(embed_dim, module_dim, module_count)

    # Test input
    i = torch.randn(embed_dim)
    print("Input:", i)

    # Forward pass
    o = layer(i)
    print("Output:", o)
