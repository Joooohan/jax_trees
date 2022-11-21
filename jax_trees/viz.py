from collections import deque

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
        for rank, node in enumerate(model.nodes[depth]):
            if node is None:
                continue
            graph.add_node(hash(node), label=str(node))
            if not node.is_leaf:
                left_node = model.nodes[depth + 1][2 * rank]
                right_node = model.nodes[depth + 1][2 * rank + 1]
                graph.add_node(hash(right_node), label=str(right_node))
                graph.add_node(hash(left_node), label=str(left_node))

                graph.add_edge(hash(node), hash(right_node), label="no")
                graph.add_edge(hash(node), hash(left_node), label="yes")

    return graph.string()
