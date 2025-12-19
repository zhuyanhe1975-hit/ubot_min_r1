from ubot.core.graph import empty_state, add_edge
from ubot.core.types import Edge, Action
from ubot.adapters.sim_adapter import SimAdapter



def test_pose_placeholder_zeroed():
    # SimAdapter no longer propagates 3D poses; ensure it initializes pose without error
    s = empty_state([0, 1], ["N", "S", "E", "W"])
    add_edge(s, Edge(0, 1, "E", "W"))
    sim = SimAdapter(s)
    st = sim.get_state()
    assert st.pose is not None
