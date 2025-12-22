from ubot.core.io import load_state_json
from ubot.core.planner import plan_astar
from ubot.core.graph import empty_state
from ubot.core.types import Edge, Action
from ubot.core.graph import add_edge

def test_planner_example():
    # Physical semantics: ATTACH requires ports already aligned in world.
    # Construct a reachable case: modules initially disconnected but positioned for E/W attach.
    s = empty_state([0, 1], ["N", "S", "E", "W"])
    s.q = {0: [0.0, 0.0], 1: [0.0, 0.0]}
    # ax poses: 0 at origin, 1 at (0.1,0,0) so 0.E and 1.W coincide
    s.pose = {
        0: (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        1: (0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    }
    goal = empty_state([0, 1], ["N", "S", "E", "W"])
    goal.q = {0: [0.0, 0.0], 1: [0.0, 0.0]}
    goal.pose = dict(s.pose)
    # For these connector frames, the aligned identity/identity pose implies twist=180.
    add_edge(goal, Edge(0, 1, "E", "W", twist=180))
    plan = plan_astar(s, goal, max_expand=2000)
    assert plan is not None
    assert any(a.kind == "ATTACH" for a in plan)


def test_planner_with_joint_goal():
    # Reachable: keep E/W connection, change joints on root; child pose is updated by kinematics.
    start = load_state_json("examples/demo2_start.json")
    goal = load_state_json("examples/demo2_start.json")
    goal.q = {0: [90.0, 90.0], 1: [0.0, 0.0]}
    # keep goal internally consistent: recompute goal pose under new joints
    from ubot.core.kinematics import solve_from_anchor, tf_to_pose_tuple, Transform
    import numpy as np

    solved = solve_from_anchor(goal, 0, goal.pose[0])
    for mid, (p, R) in solved.items():
        goal.pose[mid] = tf_to_pose_tuple(Transform(p=np.array(p), R=np.array(R)))
    plan = plan_astar(start, goal, max_expand=2000)
    assert plan is not None
    assert any(a.kind == "SET_JOINTS" for a in plan)
