# Decision Trees in JAX

Simple implementations of Decision Trees, Random Forests and Gradient Boosted
Trees with JAX.

The objective is educational: learning JAX syntax by reimplementing common ML
algorithms.


## Objective

The objective is to have models where the `fit` and `predict` methods are
completly jitted.

### Decision trees

Fitting a decision tree requires splitting the data at each node into two
subgroups. To jit the split function, we need to know in advance the number of
plits, which is bounded by $\sum_{i=1}^n 2^{(i-1)}=2^n-1$. We can perform
$2^n-1$ splits every time with a way to deal when split is called on leaf nodes.

Since every function should have the same input/ouput shapes we need a structure
to pass nodes as input and output dealing both with leaf and non lead nodes. We
could use pytrees for this purpose.

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
