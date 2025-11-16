import random
from collections import defaultdict
from typing import Hashable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from jaxtyping import Array, Key, PyTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from thrml.block_management import Block
from thrml.block_sampling import (
    BlockGibbsSpec,
    BlockSamplingProgram,
    sample_states,
    sample_with_observation,
    SamplingSchedule,
)
from thrml.conditional_samplers import (
    _SamplerState,
    _State,
    AbstractConditionalSampler,
)
from thrml.factor import AbstractFactor, FactorSamplingProgram
from thrml.interaction import InteractionGroup
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from thrml.observers import MomentAccumulatorObserver
from thrml.pgm import AbstractNode


class ContinuousNode(AbstractNode):
    pass


def generate_grid_graph(
    *side_lengths: int,
) -> tuple[
    tuple[list[ContinuousNode], list[ContinuousNode]],
    tuple[list[ContinuousNode], list[ContinuousNode]],
    nx.Graph,
]:
    G = nx.grid_graph(dim=side_lengths, periodic=False)

    coord_to_node = {coord: ContinuousNode() for coord in G.nodes}
    nx.relabel_nodes(G, coord_to_node, copy=False)

    for coord, node in coord_to_node.items():
        G.nodes[node]["coords"] = coord

    # an aperiodic grid is always 2-colorable
    bicol = nx.bipartite.color(G)
    color0 = [n for n, c in bicol.items() if c == 0]
    color1 = [n for n, c in bicol.items() if c == 1]

    u, v = map(list, zip(*G.edges()))

    return (bicol, color0, color1), (u, v), G


def plot_grid_graph(
    G: nx.Graph,
    bicol: Mapping[Hashable, int],
    ax: plt.Axes,
    *,
    node_size: int = 300,
    colors: tuple[str, str] = ("black", "orange"),
    **draw_kwargs,
):
    pos = {n: G.nodes[n]["coords"][:2] for n in G.nodes}

    node_colors = [colors[bicol[n]] for n in G.nodes]

    nx.draw(
        G,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="k",
        linewidths=0.8,
        width=1.0,
        with_labels=False,
        **draw_kwargs,
    )


colors, edges, g = generate_grid_graph(5, 5)

all_nodes = colors[1] + colors[2]

node_map = dict(zip(all_nodes, list(range(len(all_nodes)))))

fig, axs = plt.subplots()

plot_grid_graph(g, colors[0], axs)

# Fixed RNG seed for reproducibility
seed = 4242
key = jax.random.key(seed)

# diagonal elements of the inverse covariance matrix
key, subkey = jax.random.split(key, 2)
cov_inv_diag = jax.random.uniform(subkey, (len(all_nodes),), minval=1, maxval=2)

# add an off-diagonal element to the inverse covariance matrix for each edge in the graph
key, subkey = jax.random.split(key, 2)
# make sure the covaraince matrix is PSD
cov_inv_off_diag = jax.random.uniform(
    subkey, (len(edges[0]),), minval=-0.25, maxval=0.25
)


def construct_inv_cov(
    diag: Array,
    all_edges: tuple[list[ContinuousNode], list[ContinuousNode]],
    off_diag: Array,
):
    inv_cov = np.diag(diag)

    for n1, n2, cov in zip(*all_edges, off_diag):
        inv_cov[node_map[n1], node_map[n2]] = cov
        inv_cov[node_map[n2], node_map[n1]] = cov

    return inv_cov


# construct a matrix representation of the inverse covariance matrix for convenience
inv_cov_mat = construct_inv_cov(cov_inv_diag, edges, cov_inv_off_diag)

inv_cov_mat_jax = jnp.array(inv_cov_mat)

# mean vector
key, subkey = jax.random.split(key, 2)
mean_vec = jax.random.normal(subkey, (len(all_nodes),))

# bias vector
b_vec = -1 * jnp.einsum("ij, i -> j", inv_cov_mat, mean_vec)

# a Block is just a list of nodes that are all the same type
# forcing the nodes in a Block to be of the same type is important for parallelization
free_blocks = [Block(colors[1]), Block(colors[2])]

# we won't be clamping anything here, but in principle this could be a list of Blocks just like above
clamped_blocks = []

# every node in the program has to be assigned a shape and datatype (or PyTree thereof).
# this is so THRML can build an internal "global" representation of the state of the sampling program using a small number of jax arrays
node_shape_dtypes = {ContinuousNode: jax.ShapeDtypeStruct((), jnp.float32)}

# our block specification
spec = BlockGibbsSpec(free_blocks, clamped_blocks, node_shape_dtypes)

# these are just arrays that we can identify by type, will be useful later


class LinearInteraction(eqx.Module):
    """An interaction of the form $c_i x_i$."""

    weights: Array


class QuadraticInteraction(eqx.Module):
    """An interaction of the form $d_i x_i^2$."""

    inverse_weights: Array


# now we can set up our three different types of factors


class QuadraticFactor(AbstractFactor):
    r"""A factor of the form $w \: x^2$"""

    # 1/A_{ii}
    inverse_weights: Array

    def __init__(self, inverse_weights: Array, block: Block):
        # in general, a factor is initialized via a list of blocks
        # these blocks should all have the same number of nodes, and represent groupings of nodes involved in the factor
        # for example, if a Factor involved 3 nodes, we would initialize it with 3 parallel blocks of equal length
        super().__init__([block])

        # this array has shape [n], where n is the number of nodes in block
        self.inverse_weights = inverse_weights

    def to_interaction_groups(self) -> list[InteractionGroup]:
        # based on our conditional update rule, we can see that we need this to generate a Quadratic interaction with no tail nodes (i.e this interaction has no dependence on the neighbours of x_i)

        # we create an InteractionGroup that implements this interaction

        interaction = InteractionGroup(
            interaction=QuadraticInteraction(self.inverse_weights),
            head_nodes=self.node_groups[0],
            # no tail nodes in this case
            tail_nodes=[],
        )

        return [interaction]


class LinearFactor(AbstractFactor):
    r"""A factor of the form $w \: x$"""

    # b_i
    weights: Array

    def __init__(self, weights: Array, block: Block):
        super().__init__([block])
        self.weights = weights

    def to_interaction_groups(self) -> list[InteractionGroup]:
        # follows the same pattern as previous, still no tail nodes

        return [
            InteractionGroup(
                interaction=LinearInteraction(self.weights),
                head_nodes=self.node_groups[0],
                tail_nodes=[],
            )
        ]


class CouplingFactor(AbstractFactor):
    # A_{ij}
    weights: Array

    def __init__(self, weights: Array, blocks: tuple[Block, Block]):
        # in this case our factor involves two nodes, so it is initialized with two blocks
        super().__init__(list(blocks))
        self.weights = weights

    def to_interaction_groups(self) -> list[InteractionGroup]:
        # this factor produces interactions that impact both sets of nodes that it touches
        # i.e if this factor involves a term like w x_1 x_2, it should produce one interaction with weight w that has x_1 as a head node and x_2 as a tail node,
        # and another interaction with weight w that has x_2 as a head node and x_1 as a tail node

        # if we were sure that x_1 and x_2 were always the same type of node, the two interactions could be part of the same InteractionGroup
        # we won't worry about that here though
        return [
            InteractionGroup(
                LinearInteraction(self.weights),
                self.node_groups[0],
                [self.node_groups[1]],
            ),
            InteractionGroup(
                LinearInteraction(self.weights),
                self.node_groups[1],
                [self.node_groups[0]],
            ),
        ]


class GaussianSampler(AbstractConditionalSampler):
    def sample(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: _SamplerState,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> tuple[Array, _SamplerState]:
        # this is where the rubber meets the road in THRML

        # this function gets called during block sampling, and must take in information about interactions and neighbour states and produce a state update

        # interactions, active_flags, and states are three parallel lists.

        # each item in interactions is a pytree, for which each array will have shape [n, k, ...].
        # this is generated by THRML from the set of InteractionGroups that are used to create a sampling program
        # n is the number of nodes that we are updating in parallel during this call to sample
        # k is the maximum number of times any node in the block that is being updated shows up as a head node for this interaction

        # each item in active_flags is a boolean array with shape [n, k].
        # this is padding that is generated internally by THRML based on the graphical structure of the model,
        # and serves to allow for heterogeneous graph sampling to be vectorized on accelerators that rely on homogeneous data structures

        # each item in states is a list of Pytrees that represents the state of the tail nodes that are relevant to this interaction.
        # for example, for an interaction with a single tail node that has a scalar dtype, states would be:
        # [[n, k],]

        bias = jnp.zeros(shape=output_sd.shape, dtype=output_sd.dtype)
        var = jnp.zeros(shape=output_sd.shape, dtype=output_sd.dtype)

        # loop through all of the available interactions and process them appropriately

        # here we are simply implementing the math of our conditional update rule

        for active, interaction, state in zip(active_flags, interactions, states):
            if isinstance(interaction, LinearInteraction):
                # if there are tail nodes, contribute w * x_1 * x_2 * ..., otherwise contribute w
                state_prod = jnp.array(1.0)
                if len(state) > 0:
                    state_prod = jnp.prod(jnp.stack(state, -1), -1)
                bias -= jnp.sum(interaction.weights * active * state_prod, axis=-1)

            if isinstance(interaction, QuadraticInteraction):
                # this just sets the variance of the output distribution
                # there should never be any tail nodes

                var = active * interaction.inverse_weights
                var = var[..., 0]  # there should only ever be one

        return (jnp.sqrt(var) * jax.random.normal(key, output_sd.shape)) + (
            bias * var
        ), sampler_state

    def init(self) -> _SamplerState:
        return None


# our three types of factor
lin_fac = LinearFactor(b_vec, Block(all_nodes))
quad_fac = QuadraticFactor(1 / cov_inv_diag, Block(all_nodes))
pair_quad_fac = CouplingFactor(cov_inv_off_diag, (Block(edges[0]), Block(edges[1])))

# an instance of our conditional sampler
sampler = GaussianSampler()

# the sampling program itself. Combines the three main components we just built
prog = FactorSamplingProgram(
    gibbs_spec=spec,
    # one sampler for every free block in gibbs_spec
    samplers=[sampler, sampler],
    factors=[lin_fac, quad_fac, pair_quad_fac],
    other_interaction_groups=[],
)


groups = []
for fac in [lin_fac, quad_fac, pair_quad_fac]:
    groups += fac.to_interaction_groups()

prog_2 = BlockSamplingProgram(
    gibbs_spec=spec, samplers=[sampler, sampler], interaction_groups=groups
)


# we will estimate the covariances for each pair of nodes connected by an edge and compare against theory
# to do this we will need to estimate first moments and second moments
second_moments = [(e1, e2) for e1, e2 in zip(*edges)]
first_moments = [[(x,) for x in y] for y in edges]

# this will accumulate products of the node state specified by first_moments and second_moments
observer = MomentAccumulatorObserver(first_moments + [second_moments])


# how many parallel sampling chains will we run?
n_batches = 1000


schedule = SamplingSchedule(
    # how many iterations to do before drawing the first sample
    n_warmup=0,
    # how many samples to draw in total
    n_samples=10000,
    # how many steps to take between samples
    steps_per_sample=5,
)

# construct the initial state of the iterative sampling algorithm
init_state = []
for block in spec.free_blocks:
    key, subkey = jax.random.split(key, 2)
    init_state.append(
        0.1
        * jax.random.normal(
            subkey,
            (
                n_batches,
                len(block.nodes),
            ),
        )
    )

# RNG keys to use for each chain in the batch
keys = jax.random.split(key, n_batches)

# memory to hold our moment values
init_mem = observer.init()


# we use vmap to run a bunch of parallel sampling chains
moments, _ = jax.vmap(
    lambda k, s: sample_with_observation(k, prog, schedule, s, [], init_mem, observer)
)(keys, init_state)

# Take a mean over the batch axis and divide by the total number of samples
moments = jax.tree.map(lambda x: jnp.mean(x, axis=0) / schedule.n_samples, moments)

# compute the covariance values from the moment data
covariances = moments[-1] - (moments[0] * moments[1])

cov = np.linalg.inv(inv_cov_mat)

node_map = dict(zip(all_nodes, list(range(len(all_nodes)))))

real_covs = []

for edge in zip(*edges):
    real_covs.append(cov[node_map[edge[0]], node_map[edge[1]]])

real_covs = np.array(real_covs)

error = np.max(np.abs(real_covs - covariances)) / np.abs(np.max(real_covs))

print(error)
assert error < 0.01


class SpinNode(AbstractNode):
    pass


# now, build a random grid out of spin and continuous nodes


def make_random_typed_grid(
    rows: int,
    cols: int,
    seed: int,
    p_cont: float = 0.5,
):
    rng = random.Random(seed)

    # every time we make a node, flip a coin to decide what type it should be
    grid = [
        [ContinuousNode() if rng.random() < p_cont else SpinNode() for _ in range(cols)]
        for _ in range(rows)
    ]

    # Parity-based 2-coloring
    bicol = {grid[r][c]: ((r + c) & 1) for r in range(rows) for c in range(cols)}

    # Separate by color and type
    colors_by_type = {
        0: {SpinNode: [], ContinuousNode: []},
        1: {SpinNode: [], ContinuousNode: []},
    }
    for r in range(rows):
        for c in range(cols):
            n = grid[r][c]
            color = bicol[n]
            colors_by_type[color][type(n)].append(n)

    return grid, colors_by_type


grid, coloring = make_random_typed_grid(30, 30, seed)


# now generate the edges to implement our desired skip-connected grid
# we will use only odd-length edges (1, 3, 5, ...) so that our 2-coloring remains valid
def build_skip_graph_from_grid(
    grid: list[list[AbstractNode]],
    skips: list[int],
):
    rows, cols = len(grid), len(grid[0])

    # Build graph & annotate nodes with coords and type
    G = nx.Graph()
    for r in range(rows):
        for c in range(cols):
            n = grid[r][c]
            G.add_node(n, coords=(r, c))

    # Edges sorted by edge length
    u_all = []
    v_all = []
    for k in skips:
        # vertical: (r, c) -> (r+k, c)
        for r in range(rows - k):
            r2 = r + k
            for c in range(cols):
                n1 = grid[r][c]
                n2 = grid[r2][c]
                u_all.append(n1)
                v_all.append(n2)
                G.add_edge(n1, n2)

        # horizontal: (r, c) -> (r, c+k)
        for r in range(rows):
            for c in range(cols - k):
                c2 = c + k
                n1 = grid[r][c]
                n2 = grid[r][c2]
                u_all.append(n1)
                v_all.append(n2)
                G.add_edge(n1, n2, skip=k)

    return (u_all, v_all), G


edge_lengths = [1, 3, 5]
edges, graph = build_skip_graph_from_grid(grid, edge_lengths)


def plot_node_neighbourhood(
    grid,
    G: nx.Graph,
    center: Hashable,
    hops: int,
    ax: plt.Axes,
) -> None:
    rows, cols = len(grid), len(grid[0])
    r, c = G.nodes[center]["coords"]

    # make a rectangular subgrid
    r0, r1 = max(0, r - hops), min(rows - 1, r + hops)
    c0, c1 = max(0, c - hops), min(cols - 1, c + hops)
    rect_nodes = {grid[i][j] for i in range(r0, r1 + 1) for j in range(c0, c1 + 1)}

    # collect the relevant edges by length
    edges_by_k = defaultdict(list)
    for v, ed in G[center].items():
        k = int(ed.get("skip", 1))
        edges_by_k[k].append((center, v))

    # draw edges as arcs
    max_k = max(edges_by_k.keys(), default=1)
    curve_scale = 0.8
    edge_width = 1.0
    alpha = 1.0

    def rad_for_edge(u, v, k):
        r1, c1 = G.nodes[u]["coords"]
        r2, c2 = G.nodes[v]["coords"]
        base = curve_scale * (k / max_k)
        # choose bend direction based on quadrant:
        if c1 == c2:
            sign = +1.0 if r2 < r1 else -1.0  # up vs down
        else:  # horizontal edge
            sign = +1.0 if c2 > c1 else -1.0  # right vs left
        return sign * base

    # positions for plotting
    pos = {
        n: (G.nodes[n]["coords"][1], G.nodes[n]["coords"][0])
        for n in rect_nodes | {center}
    }

    for i, k in enumerate(sorted(edges_by_k)):
        for u, v in edges_by_k[k]:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                ax=ax,
                edge_color="gray",
                width=edge_width,
                alpha=alpha,
                arrows=True,
                arrowstyle="-",
                connectionstyle=f"arc3,rad={rad_for_edge(u, v, k)}",
            )

    # draw nodes
    cont_nodes = [n for n in rect_nodes if n.__class__ == ContinuousNode]
    spin_nodes = [n for n in rect_nodes if n.__class__ == SpinNode]

    node_size = 20.0

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=cont_nodes,
        node_color="black",
        node_shape="s",
        node_size=node_size,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=spin_nodes,
        node_color="orange",
        node_shape="o",
        node_size=node_size,
        ax=ax,
    )


# pick a few nodes in the grid to inspect
centers = [grid[0][7], grid[10][10], grid[-1][-1]]

fig, axs = plt.subplots(nrows=1, ncols=len(centers), figsize=(len(centers) * 5, 5))


for ax, center in zip(axs, centers):
    plot_node_neighbourhood(grid, graph, center, max(edge_lengths) + 1, ax)
# collect the different types of nodes
spin_nodes = []
cont_nodes = []
for node in graph.nodes:
    if isinstance(node, SpinNode):
        spin_nodes.append(node)
    else:
        cont_nodes.append(node)


# spin-spin interactions
ss_edges = [[], []]

# continuous-continuous interactions
cc_edges = [[], []]

# spin-continuous interactions
sc_edges = [[], []]

for edge in zip(*edges):
    if isinstance(edge[0], SpinNode) and isinstance(edge[1], SpinNode):
        ss_edges[0].append(edge[0])
        ss_edges[1].append(edge[1])
    elif isinstance(edge[0], ContinuousNode) and isinstance(edge[1], ContinuousNode):
        cc_edges[0].append(edge[0])
        cc_edges[1].append(edge[1])
    elif isinstance(edge[0], SpinNode):
        sc_edges[0].append(edge[0])
        sc_edges[1].append(edge[1])
    else:
        sc_edges[1].append(edge[0])
        sc_edges[0].append(edge[1])


# we will just randomize the weights

key, subkey = jax.random.split(key, 2)
cont_quad = QuadraticFactor(
    jax.random.uniform(subkey, (len(cont_nodes),), minval=2, maxval=3),
    Block(cont_nodes),
)

key, subkey = jax.random.split(key, 2)
cont_linear = LinearFactor(
    jax.random.normal(subkey, (len(cont_nodes),)), Block(cont_nodes)
)

key, subkey = jax.random.split(key, 2)
cont_coupling = CouplingFactor(
    jax.random.uniform(subkey, (len(cc_edges[0]),), minval=-1 / 10, maxval=1 / 10),
    (Block(cc_edges[0]), Block(cc_edges[1])),
)

key, subkey = jax.random.split(key, 2)
spin_con_coupling = CouplingFactor(
    jax.random.normal(subkey, (len(sc_edges[0]),)),
    (Block(sc_edges[0]), Block(sc_edges[1])),
)


key, subkey = jax.random.split(key, 2)
spin_linear = SpinEBMFactor(
    [Block(spin_nodes)], jax.random.normal(subkey, (len(spin_nodes),))
)

key, subkey = jax.random.split(key, 2)
spin_coupling = SpinEBMFactor(
    [Block(x) for x in ss_edges], jax.random.normal(subkey, (len(ss_edges[0]),))
)


class ExtendedSpinGibbsSampler(SpinGibbsConditional):
    def compute_parameters(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: _SamplerState,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> PyTree:
        field = jnp.zeros(output_sd.shape, dtype=float)

        unprocessed_interactions = []
        unprocessed_active = []
        unprocessed_states = []

        for interaction, active, state in zip(interactions, active_flags, states):
            # if its our new interaction, handle it
            if isinstance(interaction, LinearInteraction):
                state_prod = jnp.prod(jnp.stack(state, -1), -1)
                field -= jnp.sum(interaction.weights * active * state_prod, axis=-1)

            # if we haven't seen it, remember it
            else:
                unprocessed_interactions.append(interaction)
                unprocessed_active.append(active)
                unprocessed_states.append(state)

        # make the parent class deal with THRML-native interactions
        field -= super().compute_parameters(
            key,
            unprocessed_interactions,
            unprocessed_active,
            unprocessed_states,
            sampler_state,
            output_sd,
        )[0]

        return field, sampler_state


# tell THRML the shape and datatype of our new node
new_sd = {SpinNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool)}

# Our new graph is still two-colorable, however within each color there are two different types of node
# this means that we can't make a single block to represent each color because all of the nodes within a block have to be of the same type
# however, we might still want to ensure that the two blocks that represent each color group are sampled at the same "algorithmic" time
# i.e even though we can't sample these blocks directly in parallel because they use different update rules, we want to make sure that they
# receive the same state information
# we can make this happen in THRML by passing in a list of tuples of blocks to BlockGibbsSpec instead of a list of Blocks
# the blocks in each tuple will be sampled at the same algorithmic time
blocks = [
    (Block(coloring[0][SpinNode]), Block(coloring[0][ContinuousNode])),
    (Block(coloring[1][SpinNode]), Block(coloring[1][ContinuousNode])),
]

block_spec = BlockGibbsSpec(blocks, [], node_shape_dtypes | new_sd)

# now we can assemble our program

# first, choose the right update rule for each block in the spec
ber_sampler = ExtendedSpinGibbsSampler()
samplers = []
for block in block_spec.free_blocks:
    if isinstance(block.nodes[0], SpinNode):
        samplers.append(ber_sampler)
    else:
        samplers.append(sampler)

# collect all of our factors
factors = [
    cont_quad,
    cont_linear,
    cont_coupling,
    spin_con_coupling,
    spin_linear,
    spin_coupling,
]

program = FactorSamplingProgram(block_spec, samplers, factors, [])


batch_size = 50

schedule = SamplingSchedule(
    # how many iterations to do before drawing the first sample
    n_warmup=100,
    # how many samples to draw in total
    n_samples=300,
    # how many steps to take between samples
    steps_per_sample=15,
)


# construct the initial state of the iterative sampling algorithm
init_state = []
for block in block_spec.free_blocks:
    init_shape = (
        batch_size,
        len(block.nodes),
    )
    key, subkey = jax.random.split(key, 2)
    if isinstance(block.nodes[0], ContinuousNode):
        init_state.append(0.1 * jax.random.normal(subkey, init_shape))
    else:
        init_state.append(jax.random.bernoulli(subkey, 0.5, init_shape))

key, subkey = jax.random.split(key, 2)
keys = jax.random.split(subkey, batch_size)

samples = jax.vmap(
    lambda k, i: sample_states(
        k, program, schedule, i, [], [Block(spin_nodes), Block(cont_nodes)]
    )
)(keys, init_state)

all_samples = jnp.concatenate(samples, axis=-1)
pca = PCA(n_components=3)
preproc_data = StandardScaler().fit_transform(
    jnp.reshape(all_samples, (-1, all_samples.shape[-1]))
)
transformed_data = pca.fit_transform(preproc_data)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    transformed_data[:, 0],  # PC1
    transformed_data[:, 1],  # PC2
    transformed_data[:, 2],  # PC3
    s=50,
    alpha=0.8,
)
ax.view_init(elev=-50, azim=280)
plt.show()
