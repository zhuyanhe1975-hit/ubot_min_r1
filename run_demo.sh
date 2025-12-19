#!/usr/bin/env bash
source ~/myWorks/UBot/modular_reconf/.venv/bin/activate
python -m ubot.apps.run_plan --start examples/start.json --goal examples/goal.json --out ubot_run_log.json
