# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a JAX-based probabilistic programming project using the THRML library (Thermal) for Gibbs sampling on graphical models. The main example implements a Gaussian Markov Random Field on a grid graph with block Gibbs sampling.

## Dependencies & Package Management

- **Package manager**: `uv` (fast Python package installer and resolver)
- **Python version**: >=3.11
- **Core dependencies**: JAX, Equinox, THRML, NetworkX, scikit-learn

Install dependencies:
```bash
uv sync
```

Run the main script:
```bash
uv run python main.py
```

## Architecture

### THRML Block Sampling Framework

The codebase demonstrates THRML's block Gibbs sampling using a Gaussian MRF example:

1. **Nodes**: `ContinuousNode` - graph nodes representing random variables
2. **Blocks**: Groups of nodes of the same type for parallel sampling
3. **Factors**: Energy terms that define the probabilistic model
   - `QuadraticFactor`: diagonal precision terms (A_{ii})
   - `LinearFactor`: bias terms (b_i)
   - `CouplingFactor`: pairwise coupling terms (A_{ij})
4. **InteractionGroups**: Factors decompose into interactions with head/tail node structure
5. **ConditionalSamplers**: Implement conditional distributions (e.g., `GaussianSampler`)
6. **SamplingProgram**: Combines spec, samplers, and factors into executable program

### Key Flow

1. Generate grid graph → bipartite coloring for efficient block updates
2. Define inverse covariance matrix (precision matrix) structure
3. Create Blocks from bipartite node sets
4. Initialize `BlockGibbsSpec` with free/clamped blocks and node shapes
5. Define Factors and their InteractionGroup decompositions
6. Implement conditional sampler that processes interactions
7. Build `FactorSamplingProgram` or `BlockSamplingProgram`
8. Run parallel chains with `sample_with_observation` using vmap

### Conditional Sampling Implementation

The `GaussianSampler.sample()` method receives:
- `interactions`: List of PyTrees with shape `[n, k, ...]` where n=nodes being updated, k=max occurrences as head node
- `active_flags`: Boolean arrays `[n, k]` for handling heterogeneous graph structure
- `states`: Tail node states for each interaction

The sampler loops through interactions, accumulating bias and variance terms, then samples from the resulting conditional Gaussian.

## Testing & Validation

The main script validates sampling accuracy by:
- Running 1000 parallel chains for 10k samples each
- Using `MomentAccumulatorObserver` to track first/second moments
- Computing empirical covariances and comparing to theoretical values
- Asserting maximum relative error < 1%

Run validation:
```bash
uv run python main.py
```

Expected output: A small error value (< 0.01) printed to stdout.
