# UBot Minimal Reconfiguration Repo (Runnable)

A **minimal, runnable** repository for UBot modular robot **topology reconfiguration planning**.

- State: module connection graph with **ports**
- Actions: `DETACH(u,v)` and `ATTACH(u,pu,v,pv)`
- Planner: simple A* (BFS-like behavior via candidate pruning)
- Adapter: `SimAdapter` that applies actions to the graph (pure-topology simulation)
- CLI: `python -m ubot.apps.run_plan --start examples/start.json --goal examples/goal.json`

## Quickstart

```bash
python -m ubot.apps.run_plan --start examples/start.json --goal examples/goal.json --out ubot_run_log.json
```

## JSON format

See `examples/start.json` and `examples/goal.json`.

- `modules`: list of module ids (ints)
- `ports`: list of port names (same list for all modules in this minimal version)
- `edges`: list of edges, each is `{ "u": int, "v": int, "pu": str, "pv": str }`

## Notes

This repo is intentionally minimal:
- No physics, no collision geometry, no hardware SDK.
- Add a `RealAdapter` later that implements the same `UBotAdapter` interface.
