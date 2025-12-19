# UBot Minimal Reconfiguration Repo (Runnable)

A **minimal, runnable** repository for UBot modular robot **topology reconfiguration planning**.

- State: module connection graph with **ports**
- Actions: `DETACH(u,v)` and `ATTACH(u,pu,v,pv)`
- Planner: simple A* (BFS-like behavior via candidate pruning)
- Adapter: `SimAdapter` that applies actions to the graph (pure-topology simulation)
- CLI: `python -m ubot.apps.run_plan --start examples/start.json --goal examples/goal.json`
- MuJoCo viewer (topology -> 2D layout): `python3 -m ubot.apps.show_mujoco --state examples/start.json`
- Random connected topology generator: `python3 -m ubot.apps.gen_random_state --n 6 --seed 0 --out random_state.json`

## Quickstart

```bash
python -m ubot.apps.run_plan --start examples/start.json --goal examples/goal.json --out ubot_run_log.json
```

## MuJoCo visualization (optional)

If you have MuJoCo Python bindings installed, you can visualize a state by laying modules out in 2D using the port directions (modules are placed on the XZ plane so N/S map to +/-Z, E/W to +/-X, default spacing 0.1 m):

```bash
python3 -m ubot.apps.show_mujoco --state examples/start.json --template models/ubot/scene.xml
```

## Generate a random connected state

```bash
python3 -m ubot.apps.gen_random_state --n 6 --seed 123 --out random_state.json
```
The generator places modules on a 0.1 m grid, keeps them connected (tree), and uses the port directions (N/S/E/W) to encode adjacency.

## JSON format

See `examples/start.json` and `examples/goal.json`.

- `modules`: list of module ids (ints)
- `ports`: list of port names (same list for all modules in this minimal version)
- `edges`: list of edges, each is `{ "u": int, "v": int, "pu": str, "pv": str }`

## Notes

This repo is intentionally minimal:
- No physics, no collision geometry, no hardware SDK.
- Add a `RealAdapter` later that implements the same `UBotAdapter` interface.
