import torch
import torch.nn as nn
import torch.nn.functional as F

class DomDepSimpleLayer(nn.Module):
    def __init__(self, embed_dim, module_count):
        super(DomDepSimpleLayer, self).__init__()
        self.embed_dim = embed_dim
        self.module_count = module_count

        # Weight matrix for computing soft weight of each module (domain dependence)
        self.ws = nn.Linear(embed_dim, module_count)
        
        # 3D weight matrix for computing module activations
        self.wm = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(module_count)])

    def forward(self, h0):
        # Compute module activations
        module_activations = torch.stack([m(h0) for m in self.wm])
        module_activations = F.relu(module_activations)

        # Compute module soft weights
        module_weights = self.ws(h0)
        module_weights = F.softmax(module_weights, dim=0)

        # Mix modules based on domain-dependent soft weights
        h1 = torch.matmul(module_activations.transpose(0, 1), module_weights.unsqueeze(-1)).squeeze(-1) + h0
        return h1

if __name__ == "__main__":
    embed_dim = 3
    module_count = 2
    layer = DomDepSimpleLayer(embed_dim, module_count)

    # Test input
    h0 = torch.randn(embed_dim)
    print("Input:", h0)

    # Forward pass
    h1 = layer(h0)
    print("Output:", h1)
