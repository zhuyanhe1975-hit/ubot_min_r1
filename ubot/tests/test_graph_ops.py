from ubot.core.graph import empty_state, add_edge, remove_edge, edges_undirected
from ubot.core.types import Edge

def test_add_remove_edge():
    s = empty_state([0, 1], ["N", "S"])
    add_edge(s, Edge(0, 1, "N", "S"))
    es = edges_undirected(s)
    assert len(es) == 1
    remove_edge(s, 0, 1)
    assert len(edges_undirected(s)) == 0
