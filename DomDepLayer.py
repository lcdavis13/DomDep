# Generalized architecture for domain-dependent mixture of linear layers
# Input, output, and module space have different dimensionality.
# Linear layers are used to encode and decode the residual stream space into and out of the module space, as well as project the additive skip connection into the output space.
# Future work: investigate if the encoding and decoding matrices can somehow be a decomposition of the skip-projection matrix (though I think this is unlikely without data loss since the module space is presumably smaller than the in and out spaces).

import torch
import torch.nn as nn
import torch.nn.functional as F

class DomDepResidualLayer(nn.Module):
    def __init__(self, in_dim, out_dim, module_dim, module_count):
        super(DomDepResidualLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.module_dim = module_dim
        self.module_count = module_count

        # Weight matrix for encoding/compressing into module space
        self.we0 = nn.Linear(in_dim, module_dim)

        # Weight matrix for computing soft weight of each module (domain dependence)
        self.ws = nn.Linear(module_dim, module_count)
        
        # 3D weight matrix for computing each module's activations
        self.wm = nn.ModuleList([nn.Linear(module_dim, module_dim) for _ in range(module_count)])

        # Weight matrix for decoding into the space of the residual stream
        self.we1 = nn.Linear(module_dim, out_dim)
        
        # Weight matrix for reshaped skip connection
        self.we0e1 = nn.Linear(in_dim, out_dim)

    def forward(self, h0):
        # Encode into module space
        h0_encoded = self.we0(h0)
        
        # Compute module activations
        module_activations = torch.stack([m(h0_encoded) for m in self.wm])
        module_activations = F.relu(module_activations) # Do we want ReLU here? TO DO: Check

        # Compute module soft weights
        module_weights = self.ws(h0_encoded)
        module_weights = F.softmax(module_weights, dim=0)

        # Mix modules based on domain-dependent soft weights
        mixed_activations = torch.matmul(module_activations.transpose(0, 1), module_weights.unsqueeze(-1)).squeeze(-1)
        
        # Reshape outputs
        layer_outputs = self.we1(mixed_activations)
        h0_skip = self.we0e1(h0)
        
        # Decode and skip connection
        h1 = F.relu(layer_outputs + h0_skip)
        return h1

if __name__ == "__main__":
    in_dim = 5
    module_dim = 3
    module_count = 2
    out_dim = 4
    layer = DomDepResidualLayer(in_dim, out_dim, module_dim, module_count)

    # Test input
    i = torch.randn(in_dim)
    print("Input:", i)

    # Forward pass
    o = layer(i)
    print("Output:", o)
