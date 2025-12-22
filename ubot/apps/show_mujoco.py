from __future__ import annotations

import argparse
import os
import tempfile
import time
import xml.etree.ElementTree as ET
import threading
import queue

from ubot.core.io import load_state_json
from ubot.core.graph import edges_undirected
from ubot.core.graph import compute_parent_map
from ubot.core.layout import layout_from_ports
from ubot.core.types import Action
from ubot.adapters.sim_adapter import SimAdapter
from ubot.core.kinematics import pose_tuple_to_tf
import numpy as np
import math

def _set_replicate_count(scene_xml: str, count: int) -> str:
    root = ET.fromstring(scene_xml)
    rep = root.find(".//replicate")
    if rep is None:
        raise ValueError("Template scene.xml has no <replicate> tag.")
    rep.set("count", str(count))
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="Path to state JSON (e.g. examples/start.json)")
    ap.add_argument("--template", default="models/ubot/scene.xml", help="MJCF template with <replicate>")
    ap.add_argument("--spacing", type=float, default=0.10, help="Meters per grid cell (module centers)")
    ap.add_argument("--z", type=float, default=0.0, help="Vertical offset (Y axis) for all modules")
    ap.add_argument("--no_viewer", action="store_true", help="Only generate the XML, do not open the MuJoCo viewer")
    ap.add_argument(
        "--out",
        default=None,
        help="Optional output MJCF path (default: writes a *_generated.xml next to the template)",
    )
    ap.add_argument("--log", default=None, help="Optional plan log JSON to replay (needs matching start state)")
    ap.add_argument("--step_dt", type=float, default=1.0, help="Seconds to hold each step when replaying a log")
    ap.add_argument("--interp_steps", type=int, default=15, help="Frames for interpolating SET_JOINTS actions (>=1)")
    ap.add_argument("--interactive", action="store_true", help="Interactive mode: read SET_JOINTS/ATTACH/DETACH from stdin")
    ap.add_argument(
        "--watch_state",
        type=float,
        default=None,
        help="Poll interval (sec) to hot-reload --state JSON and update the viewer (no restart).",
    )
    ap.add_argument(
        "--graph_out",
        default=None,
        help="Optional path to dump graph info (modules, edges, joint angles, layout) as JSON",
    )
    args = ap.parse_args()

    try:
        import mujoco
        import mujoco.viewer  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "MuJoCo python bindings not available. Install `mujoco` and re-run.\n"
            f"Import error: {e}"
        )

    state = load_state_json(args.state)
    modules = list(state.adj.keys())
    adapter = SimAdapter(state)
    n = len(modules)
    if n == 0:
        raise SystemExit("Empty state: no modules.")

    layout = layout_from_ports(state)
    for w in layout.warnings:
        print(f"[layout] {w}")

    template_path = os.path.abspath(args.template)
    template_dir = os.path.dirname(template_path)
    with open(template_path, "r", encoding="utf-8") as f:
        scene_xml = f.read()

    scene_xml = _set_replicate_count(scene_xml, n)

    if args.out is not None:
        out_path = os.path.abspath(args.out)
    else:
        fd = tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_generated.xml",
            prefix="scene_",
            dir=template_dir,
            delete=False,
            encoding="utf-8",
        )
        out_path = fd.name
        fd.close()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(scene_xml)

    model = mujoco.MjModel.from_xml_path(out_path)
    data = mujoco.MjData(model)

    # Identify root module bodies (children of world) and map them to module ids.
    root_bodies = []
    for bid in range(model.nbody):
        if int(model.body_parentid[bid]) != 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name and name.startswith("ubot"):
            root_bodies.append((name, bid))
    root_bodies.sort(key=lambda x: x[0])

    if len(root_bodies) < n:
        raise SystemExit(
            f"Template expanded into only {len(root_bodies)} root 'ubot*' bodies, expected {n}. "
            "Check that ubot_body.xml defines a root <body name=\"ubot\"> and template uses <replicate>."
        )

    # Joint id lookup: ma=joint 0, mb=joint 1 per module
    joint_ids = {}
    for m in modules:
        j_ma = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_ax_ma_{m}")
        j_mb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_ax_mb_{m}")
        joint_ids[m] = (j_ma, j_mb)
    body_ids = {m: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ubot_{m}") for m in modules}
    geom_ma = {m: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"ma_{m}") for m in modules}
    geom_mb = {m: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"mb_{m}") for m in modules}
    site_ids = {}
    for m in modules:
        site_ids[(m, "E")] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"ma_connector_right_{m}")
        site_ids[(m, "W")] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"mb_connector_left_{m}")
        site_ids[(m, "N")] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"mb_connector_top_{m}")
        site_ids[(m, "S")] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"ma_connector_bottom_{m}")

    def _apply_pose_to_model(st) -> None:
        import math

        if st.pose is None:
            raise SystemExit("State has no `pose`; run through planner/adapter that maintains ax poses.")
        # Viewer uses planning-time ax poses directly (no teleport).
        nonlocal_current = {}
        for m in modules:
            tf = pose_tuple_to_tf(st.pose[m])
            nonlocal_current[m] = (tf.p, tf.R)

        # Set base body pose per module
        def _mat_to_quat(R: np.ndarray) -> np.ndarray:
            R = np.asarray(R, dtype=float).reshape(3, 3)
            q = np.empty(4, dtype=float)
            mujoco.mju_mat2Quat(q, R.flatten())
            return q

        for (module_id, (name, bid)) in zip(modules, root_bodies[:n]):
            pos, rot = nonlocal_current.get(module_id, (np.array([0.0, args.z, 0.0]), np.eye(3)))
            quat = _mat_to_quat(rot)
            model.body_pos[bid, 0] = pos[0]
            model.body_pos[bid, 1] = pos[1]
            model.body_pos[bid, 2] = pos[2]
            model.body_quat[bid, 0] = quat[0]
            model.body_quat[bid, 1] = quat[1]
            model.body_quat[bid, 2] = quat[2]
            model.body_quat[bid, 3] = quat[3]

        # set joint qpos from state
        for m in modules:
            j_ma, j_mb = joint_ids[m]
            if j_ma >= 0:
                data.qpos[j_ma] = math.radians(st.q.get(m, [0.0, 0.0])[0]) if st.q else 0.0
            if j_mb >= 0:
                data.qpos[j_mb] = math.radians(st.q.get(m, [0.0, 0.0])[1]) if st.q else 0.0

        # run kinematics only to update derived quantities (sites, geoms)
        mujoco.mj_fwdPosition(model, data)
        # Color highlighting based on geometric connection: port positions coincide and Z axes oppose
        connected_ports: dict[int, set[str]] = {m: set() for m in modules}
        pos_tol = 1e-4
        cos_tol = 0.95
        for u, ports in st.adj.items():
            for pu, (v, pv) in ports.items():
                if u >= v:  # check each undirected edge once
                    continue
                sid_u = site_ids.get((u, pu), -1)
                sid_v = site_ids.get((v, pv), -1)
                if sid_u < 0 or sid_v < 0:
                    continue
                pos_u = np.array(data.site_xpos[sid_u])
                pos_v = np.array(data.site_xpos[sid_v])
                mat_u = np.array(data.site_xmat[sid_u]).reshape(3, 3)
                mat_v = np.array(data.site_xmat[sid_v]).reshape(3, 3)
                z_u = mat_u[:, 2]
                z_v = mat_v[:, 2]
                if np.linalg.norm(pos_u - pos_v) < pos_tol and float(np.dot(z_u, z_v)) < -cos_tol:
                    connected_ports[u].add(pu)
                    connected_ports[v].add(pv)

        for m in modules:
            ma_id = geom_ma.get(m, -1)
            mb_id = geom_mb.get(m, -1)
            if ma_id >= 0:
                if {"E", "S"}.intersection(connected_ports[m]):
                    model.geom_rgba[ma_id] = [0.8, 0.0, 0.0, 1.0]
                else:
                    model.geom_rgba[ma_id] = [0.5, 0.0, 0.0, 1.0]
            if mb_id >= 0:
                if {"W", "N"}.intersection(connected_ports[m]):
                    model.geom_rgba[mb_id] = [0.0, 0.8, 0.0, 1.0]
                else:
                    model.geom_rgba[mb_id] = [0.0, 0.5, 0.0, 1.0]

    _apply_pose_to_model(adapter.get_state())

    print(f"[mjcf] Wrote: {out_path}")

    # Export graph representation + joint angles.
    joints = {}
    for jid in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not jname:
            continue
        joints[jname] = float(data.qpos[jid])

    graph_info = {
        "modules": modules,
        "ports": list(state.ports),
        "edges": [e.__dict__ for e in edges_undirected(state)],
        "layout_m": {m: [layout.grid_xy[m][0] * args.spacing, args.z, layout.grid_xy[m][1] * args.spacing] for m in modules},
        "joints_rad": joints,
    }

    if args.graph_out:
        import json

        with open(args.graph_out, "w", encoding="utf-8") as f:
            json.dump(graph_info, f, ensure_ascii=False, indent=2)
        print(f"[graph] wrote graph + joints to {args.graph_out}")
    else:
        print("[graph] modules:", graph_info["modules"])
        print("[graph] edges:", graph_info["edges"])
        print("[graph] joints_rad:", graph_info["joints_rad"])

    if args.no_viewer:
        return

    # Hot-reload support: re-load the state JSON and update adapter/model in-place.
    state_mtime = None
    next_watch_time = 0.0
    if args.watch_state is not None:
        if args.watch_state <= 0:
            raise SystemExit("--watch_state must be > 0")
        try:
            state_mtime = os.path.getmtime(args.state)
        except OSError:
            state_mtime = None

    def _hot_reload_state_if_changed(now: float) -> None:
        nonlocal state_mtime, next_watch_time
        if args.watch_state is None:
            return
        if now < next_watch_time:
            return
        next_watch_time = now + float(args.watch_state)
        try:
            mtime = os.path.getmtime(args.state)
        except OSError:
            return
        if state_mtime is not None and mtime == state_mtime:
            return
        state_mtime = mtime
        try:
            s_new = load_state_json(args.state)
        except Exception as e:
            print(f"[watch] reload failed: {e}")
            return
        new_modules = list(s_new.adj.keys())
        if set(new_modules) != set(modules) or len(new_modules) != len(modules):
            print(f"[watch] module set changed; ignoring reload. old={modules}, new={new_modules}")
            return
        # Keep the original module ordering for mapping to replicated bodies.
        try:
            adapter._state = s_new  # type: ignore[attr-defined]
            adapter._state.parent = compute_parent_map(adapter._state, root=0)  # type: ignore[attr-defined]
            adapter._update_poses()  # type: ignore[attr-defined]
        except Exception as e:
            print(f"[watch] failed to apply reloaded state: {e}")
            return
        _apply_pose_to_model(adapter.get_state())
        print("[watch] reloaded state.")

    # If log is provided, replay actions sequentially
    actions = []
    if args.log:
        import json

        with open(args.log, "r", encoding="utf-8") as f:
            log = json.load(f)
        for a in log.get("plan", []):
            actions.append(
                Action(
                    kind=a["kind"],
                    u=a["u"],
                    v=a["v"],
                    pu=a.get("pu"),
                    pv=a.get("pv"),
                    twist=a.get("twist"),
                    j1=a.get("j1"),
                    j2=a.get("j2"),
                )
            )
        print(f"[replay] loaded {len(actions)} actions from log.")

    def _parse_command(line: str) -> Action | None:
        """
        Supported commands (whitespace separated, case-insensitive):
          - SET_JOINTS <module> <j1_deg> <j2_deg>
          - ATTACH <u> <pu> <v> <pv> [twist_deg]
          - DETACH <u> <v>
          - PRINT   (prints current edges+joints)
          - HELP
          - QUIT / EXIT
        """
        s = line.strip()
        if not s:
            return None
        parts = s.replace(",", " ").replace("(", " ").replace(")", " ").split()
        if not parts:
            return None
        cmd = parts[0].upper()
        if cmd in ("HELP", "?"):
            print(
                "[interactive] commands:\n"
                "  SET_JOINTS <m> <j1> <j2>\n"
                "  ATTACH <u> <pu> <v> <pv> [twist]\n"
                "  DETACH <u> <v>\n"
                "  PRINT\n"
                "  QUIT"
            )
            return None
        if cmd in ("QUIT", "EXIT"):
            return Action(kind="DETACH", u=-1, v=-1)  # sentinel
        if cmd == "PRINT":
            st = adapter.get_state()
            print("[interactive] edges:", [e.__dict__ for e in edges_undirected(st)])
            print("[interactive] joints:", st.q)
            return None
        if cmd in ("SET_JOINT", "SET_JOINTS"):
            if len(parts) != 4:
                print("[interactive] usage: SET_JOINTS <m> <j1> <j2>")
                return None
            m = int(parts[1])
            j1 = float(parts[2])
            j2 = float(parts[3])
            return Action(kind="SET_JOINTS", u=m, v=m, j1=j1, j2=j2)
        if cmd == "DETACH":
            if len(parts) != 3:
                print("[interactive] usage: DETACH <u> <v>")
                return None
            u = int(parts[1])
            v = int(parts[2])
            return Action(kind="DETACH", u=u, v=v)
        if cmd == "ATTACH":
            if len(parts) not in (5, 6):
                print("[interactive] usage: ATTACH <u> <pu> <v> <pv> [twist]")
                return None
            u = int(parts[1])
            pu = parts[2].upper()
            v = int(parts[3])
            pv = parts[4].upper()
            twist = None
            if len(parts) == 6:
                twist = int(float(parts[5])) % 360
            return Action(kind="ATTACH", u=u, v=v, pu=pu, pv=pv, twist=twist)
        print(f"[interactive] unknown command: {cmd}. Type HELP.")
        return None

    cmd_q: queue.Queue[Action] = queue.Queue()

    def _stdin_reader() -> None:
        print("[interactive] ready. Type HELP for commands.")
        while True:
            try:
                line = input("> ")
            except EOFError:
                cmd_q.put(Action(kind="DETACH", u=-1, v=-1))  # sentinel
                return
            try:
                a = _parse_command(line)
            except Exception as e:
                print(f"[interactive] parse error: {e}")
                continue
            if a is None:
                continue
            cmd_q.put(a)
            if a.kind == "DETACH" and a.u == -1 and a.v == -1:
                return

    with mujoco.viewer.launch_passive(model, data) as viewer:  # pragma: no cover
        step_idx = 0
        next_time = time.time()
        if args.interactive:
            t = threading.Thread(target=_stdin_reader, daemon=True)
            t.start()
        while viewer.is_running():
            now = time.time()
            _hot_reload_state_if_changed(now)

            # Process interactive command queue (highest priority).
            if args.interactive:
                try:
                    a = cmd_q.get_nowait()
                except queue.Empty:
                    a = None
                if a is not None:
                    if a.kind == "DETACH" and a.u == -1 and a.v == -1:
                        print("[interactive] exiting viewer loop.")
                        break
                    try:
                        adapter.apply(a)
                        _apply_pose_to_model(adapter.get_state())
                        viewer.sync()
                        hold = 0.5 if a.kind in ("ATTACH", "DETACH") else args.step_dt
                        time.sleep(max(0.01, hold))
                    except Exception as e:
                        print(f"[interactive] action rejected: {e}")
                    continue

            if step_idx < len(actions) and now >= next_time:
                a = actions[step_idx]
                # snapshot connected set before action (for attach interpolation)
                st_before = adapter.get_state()
                # If joint action, optionally interpolate for smoother animation
                if a.kind == "SET_JOINTS" and args.interp_steps > 1:
                    st_cur = adapter.get_state()
                    j_cur = st_cur.q.get(a.u, [0.0, 0.0]) if st_cur.q else [0.0, 0.0]
                    for k in range(1, args.interp_steps + 1):
                        alpha = k / args.interp_steps
                        j1 = j_cur[0] + (a.j1 - j_cur[0]) * alpha
                        j2 = j_cur[1] + (a.j2 - j_cur[1]) * alpha
                        # Directly set intermediate joints without discrete validation (visualization only)
                        adapter._state.q = adapter._state.q or {}
                        adapter._state.q[a.u] = [j1, j2]
                        adapter._update_poses()
                        _apply_pose_to_model(adapter.get_state())
                        viewer.sync()
                        time.sleep(1.0 / 60.0)
                # Apply final discrete action to keep state consistent/validated
                adapter.apply(a)
                st_after = adapter.get_state()

                # If ATTACH, interpolate pose for newly-connected modules to avoid teleport.
                if a.kind == "ATTACH":
                    newly = set(st_after.adj.keys())  # interpolate all for simplicity
                    cur_world = getattr(_apply_pose_to_model, "_current_world", {})
                    target_world = {m: pose_tuple_to_tf(st_after.pose[m]) for m in st_after.pose}

                    def mat_to_quat(R: np.ndarray) -> np.ndarray:
                        q = np.empty(4, dtype=float)
                        mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(3, 3).flatten())
                        return q

                    def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
                        q0 = q0 / np.linalg.norm(q0)
                        q1 = q1 / np.linalg.norm(q1)
                        dot = float(np.dot(q0, q1))
                        if dot < 0.0:
                            q1 = -q1
                            dot = -dot
                        if dot > 0.9995:
                            q = q0 + t * (q1 - q0)
                            return q / np.linalg.norm(q)
                        theta0 = math.acos(max(-1.0, min(1.0, dot)))
                        sin0 = math.sin(theta0)
                        theta = theta0 * t
                        s0 = math.sin(theta0 - theta) / sin0
                        s1 = math.sin(theta) / sin0
                        return s0 * q0 + s1 * q1

                    frames = max(1, int(0.5 * 60))
                    for f in range(1, frames + 1):
                        t = f / frames
                        for m in newly:
                            if m not in st_after.pose:
                                continue
                            p1, R1 = target_world[m].p, target_world[m].R
                            p0, R0 = cur_world.get(m, (p1, R1))
                            p = (1 - t) * np.array(p0) + t * np.array(p1)
                            q0 = mat_to_quat(R0)
                            q1 = mat_to_quat(R1)
                            q = slerp(q0, q1, t)
                            R = np.zeros((3, 3), dtype=float)
                            mujoco.mju_quat2Mat(R.flatten(), q)
                            cur_world[m] = (p, R)
                        _apply_pose_to_model._current_world = cur_world
                        _apply_pose_to_model(st_after)
                        viewer.sync()
                        time.sleep(1.0 / 60.0)

                _apply_pose_to_model(st_after)
                st = adapter.get_state()
                # NOTE: `st.pose` is a placeholder in SimAdapter; viewer uses kinematics-based world poses.
                poses_dbg = {m: list(st.pose[m][0:3]) for m in st.pose} if st.pose else {}
                print(f"[replay] step {step_idx}: {actions[step_idx]}, world_pos={poses_dbg}")
                step_idx += 1
                hold = 0.5 if a.kind in ("ATTACH", "DETACH") else args.step_dt
                next_time = now + max(0.05, hold)

            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
