from ubot.core.graph import empty_state, add_edge
from ubot.core.types import Edge
from ubot.core.kinematics import propagate_poses_3d, edge_geometry_ok


def test_edge_geometry_ok_for_basic_chain():
    s = empty_state([0, 1], ["N", "S", "E", "W"])
    s.q = {0: [0.0, 0.0], 1: [0.0, 0.0]}
    add_edge(s, Edge(0, 1, "E", "W"))
    s.parent = {0: None, 1: 0}
    poses = propagate_poses_3d(s)
    assert edge_geometry_ok(s, poses)

