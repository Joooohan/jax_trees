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

The whole DecisionTree could be a queue of nodes though a deeper tree-like structure would be preferrable

