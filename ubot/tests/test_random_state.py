from ubot.core.graph import edges_undirected
from ubot.core.layout import layout_from_ports
from ubot.core.random_state import random_connected_state


def test_random_connected_state_is_tree_and_grid():
    res = random_connected_state(6, seed=0)
    s = res.state
    pos = res.grid_xy

    assert len(pos) == 6
    assert len(set(pos.values())) == 6  # no overlap

    edges = edges_undirected(s)
    assert len(edges) == 5  # tree

    # Connectivity via DFS
    adj = {m: set() for m in s.adj.keys()}
    for e in edges:
        adj[e.u].add(e.v)
        adj[e.v].add(e.u)
    seen = set()
    stack = [0]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(adj[u])
    assert len(seen) == 6

    # Ports match grid directions
    dir_map = {(1, 0): ("E", "W"), (-1, 0): ("W", "E"), (0, 1): ("N", "S"), (0, -1): ("S", "N")}
    for e in edges:
        dx = pos[e.v][0] - pos[e.u][0]
        dy = pos[e.v][1] - pos[e.u][1]
        assert abs(dx) + abs(dy) == 1
        pu, pv = dir_map[(dx, dy)]
        assert (e.pu, e.pv) == (pu, pv)

    # Layout solver should recover the same grid (up to translation)
    layout = layout_from_ports(s).grid_xy
    # align by translating layout so module 0 matches
    dx = pos[0][0] - layout[0][0]
    dy = pos[0][1] - layout[0][1]
    for m in pos:
        assert (layout[m][0] + dx, layout[m][1] + dy) == pos[m]

