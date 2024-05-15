(experimental) repo testing a simple linear model of domain-dependent (data dependent) operator.

Two parallel layers. One small linear softmax layer D is used to generate soft weights S which modulate regions of a larger linear layer M. M can be segmented into Modules (subset linear layers) each of which is modulated by one of the softweights in S.
