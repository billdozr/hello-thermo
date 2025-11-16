# THRML-Powered Restricted Boltzmann Machine

A Restricted Boltzmann Machine (RBM) implementation using the THRML library's hardware-efficient Gibbs sampling for the bars-and-stripes dataset.

## Overview

This project implements **Idea 13** from the THRML project ideas: a Boltzmann machine trained using THRML's efficient sampling infrastructure. The implementation demonstrates:

-  RBM architecture with SpinNodes for binary visible/hidden units
-  Energy-based model formulation using SpinEBMFactor
-  Block Gibbs sampling for conditional distributions
-  Contrastive Divergence (CD-k) training algorithm
-  Visualization of learned filters and reconstructions
-  Bars-and-stripes toy dataset

## Architecture

### Model Structure
- **Visible units**: 16 (4×4 images)
- **Hidden units**: 8
- **Energy function**: E(v,h) = -v^T W h - a^T v - b^T h
- **Node type**: SpinNode (binary {0,1} values)

### Key Components

1. **Dataset Generation** ([rbm_thrml.py:38](rbm_thrml.py#L38))
   - Generates horizontal bars and vertical stripes
   - 4×4 binary images
   - Converts to {-1, +1} for THRML compatibility

2. **RBM Class** ([rbm_thrml.py:94](rbm_thrml.py#L94))
   - Visible and hidden SpinNode blocks
   - Weight matrix W, biases a and b
   - Energy factors using SpinEBMFactor

3. **Sampling Methods** ([rbm_thrml.py:201](rbm_thrml.py#L201))
   - `sample_given_visible()`: Sample hidden units given visible
   - `sample_given_hidden()`: Sample visible units given hidden
   - Uses FactorSamplingProgram with BlockGibbsSpec

4. **Training** ([rbm_thrml.py:313](rbm_thrml.py#L313))
   - Contrastive Divergence (CD-1) algorithm
   - Mini-batch gradient updates
   - Positive phase: data-clamped sampling
   - Negative phase: free-running Gibbs sampling

## Usage

```bash
# Run the demo
uv run python rbm_thrml.py
```

This will:
1. Generate the bars-and-stripes dataset
2. Initialize an RBM with 16 visible and 8 hidden units
3. Train using CD-1 for 100 epochs
4. Visualize learned filters and reconstructions
5. Compute reconstruction error

## Results

### Generated Files
- `bars_and_stripes_examples.png` - Sample dataset images
- `rbm_filters.png` - Learned weight filters (8 hidden units)
- `rbm_reconstructions.png` - Original vs reconstructed images

### Performance
- **Reconstruction error**: ~0.48 (48% pixel mismatch)
- **Training time**: Fast thanks to THRML's JAX-based implementation

## Technical Highlights

### THRML Integration

The implementation showcases THRML's key features:

1. **Block Structure**: Separate blocks for visible and hidden units enable efficient bipartite Gibbs sampling

2. **Energy Factors**:
   - Unary factors for biases (visible and hidden)
   - Pairwise factors for interactions (creating edges between all visible-hidden pairs)

3. **Custom Observer**: FinalStateObserver to handle heterogeneous block sizes ([rbm_thrml.py:27](rbm_thrml.py#L27))

4. **Conditional Sampling**: Uses SpinGibbsConditional for efficient spin-valued Gibbs updates

### Key Implementation Details

**Factor Construction** ([rbm_thrml.py:133](rbm_thrml.py#L133)):
```python
# Create edges for all visible-hidden pairs
for i, v_node in enumerate(self.visible_nodes):
    for j, h_node in enumerate(self.hidden_nodes):
        visible_edges.append(v_node)
        hidden_edges.append(h_node)
        edge_weights.append(-self.W[i, j])

interaction_factor = SpinEBMFactor(
    node_groups=[Block(visible_edges), Block(hidden_edges)],
    weights=jnp.array(edge_weights)
)
```

This creates a bipartite graph where each visible unit connects to each hidden unit, as required by the RBM architecture.

**CD-k Training** ([rbm_thrml.py:313](rbm_thrml.py#L313)):
```python
# Positive phase: clamp visible to data
h_pos = self.sample_given_visible(key_pos, data, n_steps=1)

# Negative phase: alternate k times
for i in range(k):
    v_neg = self.sample_given_hidden(key_v, h_neg, n_steps=1)
    h_neg = self.sample_given_visible(key_h, v_neg, n_steps=1)

# Compute gradients from positive and negative statistics
grad_W = pos_weights - neg_weights
```

## Limitations & Future Work

### Current Limitations

1. **Poor Reconstruction Quality**: The current implementation shows ~48% reconstruction error, suggesting:
   - Insufficient model capacity (only 8 hidden units)
   - Suboptimal hyperparameters
   - Need for longer training or better initialization

2. **Sampling Architecture**: The bipartite edge construction creates n_visible × n_hidden edges, which may not be the most efficient THRML representation

### Potential Improvements

1. **Architecture**:
   - Increase hidden units (try 16-32)
   - Add momentum to parameter updates
   - Implement weight decay regularization

2. **Training**:
   - Use CD-k with k>1 for better gradient estimates
   - Implement persistent CD (PCD) for faster convergence
   - Add learning rate scheduling

3. **THRML Optimization**:
   - Investigate alternative factor structures
   - Explore using categorical nodes instead of spin nodes
   - Benchmark against naive Python Gibbs sampling

4. **Evaluation**:
   - Add pseudo-likelihood metric
   - Compute partition function estimates
   - Generate samples from the model

## Dependencies

- JAX >= 0.4
- THRML >= 0.1.3
- NumPy >= 2.3
- Matplotlib >= 3.10

## References

- Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence
- Fischer, A., & Igel, C. (2012). An introduction to restricted Boltzmann machines
- THRML Documentation: Hardware-efficient probabilistic inference

## License

MIT
