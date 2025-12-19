from ubot.core.io import load_state_json
from ubot.core.planner import plan_astar

def test_planner_example():
    start = load_state_json("examples/start.json")
    goal = load_state_json("examples/goal.json")
    plan = plan_astar(start, goal, max_expand=20000)
    assert plan is not None
