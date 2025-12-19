from ubot.core.io import load_state_json
from ubot.core.layout import layout_from_ports


def test_layout_start_example_is_linear():
    s = load_state_json("examples/start.json")
    layout = layout_from_ports(s).grid_xy
    assert layout[0] == (0, 0)
    assert layout[1] == (1, 0)
    assert layout[2] == (2, 0)
    assert layout[3] == (3, 0)


def test_layout_goal_example_turns_down():
    s = load_state_json("examples/goal.json")
    layout = layout_from_ports(s).grid_xy
    # 0 --E/W--> 1
    assert layout[1] == (layout[0][0] + 1, layout[0][1])
    # 1 --S/N--> 2 : 2 is south of 1
    assert layout[2] == (layout[1][0], layout[1][1] - 1)
    # 2 --E/W--> 3
    assert layout[3] == (layout[2][0] + 1, layout[2][1])

