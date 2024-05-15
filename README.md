(experimental) repo testing a simple linear model of domain-dependent (data dependent) operator.

Two parallel layers. One small linear softmax layer D is used to generate soft weights S which modulate regions of a larger linear layer L. L can be thought of as segmented into modules M (subset linear layers) each of which is modulated by one of the softweights in S. M vectors weighted by S are summed to produce the output.
Generalized versions included involve linear projections from the dimensionality of the residual stream into the dimensionality of the module space, and back to residual stream space.
