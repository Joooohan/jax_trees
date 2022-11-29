from collections import deque

import jax.numpy as jnp
import pygraphviz as pgv


def show(model) -> str:
    """Return the GraphViz representation of a DecitionTree."""
    graph = pgv.AGraph(directed=True)
    graph.node_attr.update(shape="box")
    to_visit = deque([model.root])

    while to_visit:
        node = to_visit.popleft()
        if not node.is_leaf:
            graph.add_node(hash(node), label=str(node))
            graph.add_node(hash(node.right_node), label=str(node.right_node))
            graph.add_node(hash(node.left_node), label=str(node.left_node))

            graph.add_edge(hash(node), hash(node.right_node), label="no")
            graph.add_edge(hash(node), hash(node.left_node), label="yes")

            to_visit.append(node.right_node)
            to_visit.append(node.left_node)

    return graph.string()


def show_nodes(model) -> str:
    """Return the GraphViz representation of a DecitionTree."""
    graph = pgv.AGraph(directed=True)
    graph.node_attr.update(shape="box")

    for depth in range(model.max_depth + 1):
        node = model.nodes[depth]
        for rank in range(2**depth):
            if jnp.sum(node.mask[rank]) == 0:
                continue
            node_id = f"{depth}_{rank}"
            graph.add_node(node_id, label=node.show(rank))
            if not node.is_leaf[rank]:
                left_id = f"{depth+1}_{2*rank}"
                right_id = f"{depth+1}_{2*rank+1}"
                # graph.add_node(left_id, label=str(right_node))
                # graph.add_node(right_id, label=str(left_node))

                graph.add_edge(node_id, right_id, label="no")
                graph.add_edge(node_id, left_id, label="yes")

    return graph.string()
