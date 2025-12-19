from ubot.core.graph import empty_state, add_edge, compute_parent_map
from ubot.core.types import Edge
from ubot.adapters.sim_adapter import SimAdapter
from ubot.core.types import Action
import pytest
from ubot.core.validator import ValidationError


def test_compute_parent_map_basic():
    s = empty_state([0, 1, 2], ["N", "S", "E", "W"])
    add_edge(s, Edge(0, 1, "E", "W"))
    add_edge(s, Edge(1, 2, "E", "W"))
    parents = compute_parent_map(s, root=0)
    assert parents[0] is None
    assert parents[1] == 0
    assert parents[2] == 1


def test_parent_updates_after_detach_and_attach():
    s = empty_state([0, 1, 2], ["N", "S", "E", "W"])
    add_edge(s, Edge(0, 1, "E", "W"))
    add_edge(s, Edge(1, 2, "E", "W"))
    sim = SimAdapter(s)
    assert sim.get_state().parent[2] == 1

    # Detach would create a floating branch; should be rejected.
    with pytest.raises(ValidationError):
        sim.apply(Action(kind="DETACH", u=1, v=2))
