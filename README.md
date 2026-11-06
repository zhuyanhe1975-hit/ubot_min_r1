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

The default search is A*. On this start/goal pair it stops at `--max_expand` without a plan, because an `ATTACH` is legal only when the two ports are already aligned. The hierarchical planner solves the same files on CPU in a few seconds:

```bash
python -m ubot.apps.run_plan --start examples/start.json --goal examples/goal.json --planner hier --out ubot_run_log.json
```

## GitHub Actions

[![Example plan](https://github.com/zhuyanhe1975-hit/ubot_min_r1/actions/workflows/example-plan.yml/badge.svg)](https://github.com/zhuyanhe1975-hit/ubot_min_r1/actions/workflows/example-plan.yml)

Pushes to `main`, pull requests, and manual runs execute that hierarchical command on `ubuntu-latest` (workflow: [Example plan](https://github.com/zhuyanhe1975-hit/ubot_min_r1/actions/workflows/example-plan.yml)). The MuJoCo viewer is not part of CI: it opens a GUI and is not reliable on a headless runner.

### View a run and its artifacts

1. Open the **Actions** tab and select **Example plan**.
2. Open the run for your commit. The job summary shows success or failure, runtime, action counts, and the action list.
3. On the run page, download the **ubot-example-plan** artifact (kept for 14 days). It contains:
   - `ubot_run_log.json` — plan, executed actions, and meta (`planner`, `runtime_sec`), written when a plan is found
   - `plan.log` — planner stdout and stderr
   - `summary.md` — the same brief summary posted on the job

Use **Run workflow** on that page to start a run without a new commit.

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
