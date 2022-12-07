# Decision Trees in JAX

## Installation guide
Run `pip install jax_trees[cpu]` to run on CPU or `pip install jax_trees[cuda]`
to run on GPU.

# Rationale

This package is a simple implementations of Decision Trees, Random Forests and
Gradient Boosted Trees using JAX.

The initial objective of this package was to learn to use JAX. I wanted a
challenge to learn the JAX syntax by reimplementing common ML algorithms.
Decision Trees and related ensemble models seemed like good candidates for this
purpose.

## Set up

JAX is a computing library with some specificities. Pure functions written in
JAX can be compiled into efficient XLA code using the `jit` (just in time)
function. The compilation process itself can be long but once the function is
compiled it runs many time faster than the equivalent Python code, especially
if you call if many times. Compiled functions only accepts statically shaped
arrays as arguments. When you call a jitted function with different shapes
inputs, the function is recompiled, so we should try as much as possible to
have statically shaped arrays as inputs to such functions.

JAX also has some utilities like the `vmap` to automatically vectorize
computations and the `grad` to automatically compute the gradient of a scalar
function. These should prove useful when ensembling the trees as well as when
implementing gradient boosted trees.

To test the constraints and advantages of using JAX, my objective is to write
the ML models in the most JAX idiomatic way. To do that I set myself the
constraint of having a completely jitted `fit` and `predict` methods on all
models.

## Implementing decision trees

### Split the nodes

Fitting a decision tree requires recursively splitting the data at each node
into two subgroups until the specified `max_depth` is reached. In a typical
implementation, we would have a `TreeNode` class that has a `split` method. One
would split the node in 2 and instantiate child nodes with the resulting
subgroups. In particular, the subgroup sizes would be particular to each model
and each dataset.

I implemented a `split` function that can compute the information gain by
spliting a dataset based on a value and a column number. To select the best
split in a node, I generate many candidates for each feature column and use the
JAX function `vmap` two times to compute the scores, one time to parallelize
over the candidates in a column and a second time to parallelize over the
columns. We can then easily parallelize the split computations to get a matrix
of information gains for many split candidates and then select the best
splitting point for a node. So far so good.

### Static number of splits

A first issue arises however as some nodes cannot be split even though we have
not reached `max_depth`. Indeed, the node may have too few data points to be
split (less that `min_samples`). In a normal Python function, we would just add
a condition and split the node if it has more than `min_samples` data points.
However, this cannot be done in a jitted function since the number and order of
operations should be known at compile time and we don't know which node will
not splittable before actually splitting the data.

To jit our fit method, we have to perform $2^n$ splits every single time where
$n$ is the maximum depth of the tree. The result of the jitted function also
has to always have the same shape, so we should be able to represent leaf nodes
and phantom nodes (i.e. nodes below leaf nodes which hold no data).
Furthermore, we have to find a way to call split on every node, even if the
node is a leaf or is a phantom node.

For this purpose, we add a `mask` array and a boolean `is_leaf` to our
`TreeNode` class. The `mask` has a static shape `(n_samples,)` and weights the
node data. If the mask sums to zero, it means that the node holds no data. If
the `is_leaf` boolean is true, then we set masks of child nodes to zero.

```python
is_leaf = jnp.maximum(depth + 1 - self.max_depth, 0) + jnp.maximum(
    self.min_samples + 1 - jnp.sum(mask), 0
)
is_leaf = jnp.minimum(is_leaf, 1).astype(jnp.int8)

# zero-out child masks if current node is a leaf
left_mask *= 1 - is_leaf
right_mask *= 1 - is_leaf
```

### Representing the tree

The decision tree itself is a collection of `TreeNode` objects. I chose to
represent the tree as a nested structure where a node can be accessed doing
`Tree.nodes[depth][rank]` where the rank is the position of the node at a given
depth. This simple structure allows to access a node children you just do
`nodes[depth+1][2*rank]` and `nodes[depth+1][2*rank+1`.

Now to be able to jit the method itself, the `DecisionTree` object should be a
pytree object so that `self`, which is passed as the first argument to the
method, can be serialized as argument and returned from a jitted function.
To do that, we register the class using the
`jax.tree_util.register_pytree_node_class` function and implement the
`tree_flatten` and `tree_unflatten` methods on the class.

Finally, since jitted functions are purely functional, the jitted fit can have
no side effect. This means that we cannot update the model parameters inplace
using fit. Instead, we return the fitted model as output of the fit method
`fitted_model = model.fit(...)`.

### JIT performance
...

The idea would be to have a function `split_node(TreeNode, X, y) ->
[TreeNode,TreeNode]` This function can be jitted if the `TreeNode` is a Pytree.
The `TreeNode` is specified by a `split_value`, a `split_column`, a `leaf_value`
and a `mask`.

The `fit` function would do something like:
 - Create the root TreeNode and add it to the list of nodes to explore
 - tree = [root]
 - for _ in range(2^n-1):
 -    node = queue.pop()
 -    left, right = split_node(node, X, y)
 -    queue.append(left, right)
 -    tree.append

The whole DecisionTree could be a queue of nodes though a deeper tree-like
structure would be preferrable.

The conditions cannot depend at all from the input data content. This implies
that the nodes created at each split need to have the same structure. If a Node
holds a link to its children and every node need to have the same structure then
we can only build an unlimited tree. We could have a node having a reference to
it parent but then the tree exploration becomes difficult. A good solution would
be to have the tree as a nested structure without nodes holding references to
each other.

The tree structure could be represented by a list of list with a structure like
`tree[depth][rank]`. You know that `tree[level][rank]` has children
`tree[level+1][2*rank]` and `tree[level+1][2*rank+1]`.

For the `fit` and `predict` methods to be jitted, the class must be registered
as a Pytree custom type.

## Retracing of inner JIT functions

Inner functions are retraced when called several times as if the whole code was
inlined. To avoid it we could primitives such as `lax.fori_loop` or `lax.scan`.

If we only use shallow trees that we reuse multiple times like in boosted trees
or random forest then the strategy of tracing the whole iteration is not so
terrible.

However for tree with a depth greater than 4 the jitting time explodes
exponentially making the use of this strategy impractical.

# Reducing the number of tracing

To avoid retracing the functions in every loop iteration we need to use the
`jax.lax.scan` primitive that allows us to execute several times a function
traced a single time.

The loop takes a sequence as input and ouput a sequence of values. We could use
`jax.vmap` instead also to avoid calling `scan` and ignoring the carry over.

At each depth, we iterate over the nodes of the level and we produce the nodes
of the next level. We could implement the inner loop using `scan`. The number of
tracing would decrease exponentially from `2**n` to `n` where `n` is the tree
depth.

Implementing the tracing at each level using scan works but produces naturally
vectorized tree nodes. To account for this new structure we use `vmap` in the
predict step to vectorize the computation of the output.
