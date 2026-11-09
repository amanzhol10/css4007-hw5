import json
import os
import subprocess
import sys
from pathlib import Path

from .app import HelpdeskApp, make_model, reset_storage
from .data import scenario_list
from .part1 import build_graph, run_traced


def log_path(out_dir, scenario_id):
    return Path(out_dir) / f"{scenario_id}.json"


def load_log(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_log(path, log):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def find_scenario(scenario_id):
    for scenario in scenario_list():
        if scenario["id"] == scenario_id.upper():
            return scenario
    raise SystemExit(f"unknown scenario: {scenario_id}")


def print_event(entry):
    print(f"  [pid {entry['pid']}] user={entry.get('user_id')} thread={entry.get('thread_id')}")
    print(f"  User: {entry['message']}")
    print(f"  Route: {' -> '.join(entry.get('route') or entry.get('nodes', []))}")
    print("  Reply: " + entry["reply"].replace("\n", "\n         "))


def run_scenario(scenario_id, model_kind, out_dir, storage_dir, start_event=0):
    scenario = find_scenario(scenario_id)
    path = log_path(out_dir, scenario["id"])
    if start_event == 0:
        model, model_label = make_model(model_kind) if scenario["part"] != 1 else (None, "no model (Part 1)")
        if scenario.get("reset_before"):
            reset_storage(storage_dir)
        log = {
            "scenario_id": scenario["id"],
            "title": scenario["title"],
            "part": scenario["part"],
            "status_profile": scenario["status_profile"],
            "graph_variant": scenario.get("graph_variant", "original") if scenario["part"] == 1 else None,
            "model": model_label,
            "command": " ".join(["python", "-m", "helpdesk"] + sys.argv[1:]),
            "events": [],
        }
        print(f"== {scenario['id']} - {scenario['title']} (profile={scenario['status_profile']}) ==")
    else:
        log = load_log(path)
        model, model_label = make_model(model_kind) if scenario["part"] != 1 else (None, "no model (Part 1)")
        print(f"== {scenario['id']} continued in a NEW process (pid {os.getpid()}) ==")

    events = scenario["events"]
    app = None
    try:
        for index in range(start_event, len(events)):
            event = events[index]
            if event["action"] == "restart_process":
                log["events"].append(
                    {"index": index, "action": "restart_process", "terminated_pid": os.getpid()}
                )
                save_log(path, log)
                if app is not None:
                    app.close()
                    app = None
                command = [
                    sys.executable, "-m", "helpdesk", "scenario", scenario["id"],
                    "--model", model_kind, "--out", str(out_dir),
                    "--storage", str(storage_dir), "--start-event", str(index + 1),
                ]
                print(f"  -- process {os.getpid()} exits; launching a new process --")
                sys.stdout.flush()
                code = subprocess.run(command).returncode
                if code != 0:
                    raise SystemExit(code)
                return log
            if scenario["part"] == 1:
                variant = scenario.get("graph_variant", "original")
                result = run_traced(build_graph(variant), event["message"])
                entry = {
                    "index": index,
                    "action": "message",
                    "pid": os.getpid(),
                    "user_id": event["user_id"],
                    "thread_id": event["thread_id"],
                    "message": event["message"],
                    "route": result["route"],
                    "trace": result["trace"],
                    "reply": result["final_state"]["reply"],
                    "final_state": result["final_state"],
                }
            else:
                if app is None:
                    app = HelpdeskApp(
                        model,
                        storage_dir=storage_dir,
                        ticket_dir=Path(out_dir) / "tickets",
                        status_profile=scenario["status_profile"],
                    )
                result = app.send(event["user_id"], event["thread_id"], event["message"])
                entry = {"index": index, "action": "message", "pid": os.getpid(), **result}
            log["events"].append(entry)
            print_event(entry)
            save_log(path, log)
    finally:
        if app is not None:
            app.close()
    save_log(path, log)
    print(f"  log saved: {path.as_posix()}")
    return log


def run_all(model_kind, out_dir, storage_dir):
    for scenario in scenario_list():
        command = [
            sys.executable, "-m", "helpdesk", "scenario", scenario["id"],
            "--model", model_kind, "--out", str(out_dir), "--storage", str(storage_dir),
        ]
        code = subprocess.run(command).returncode
        if code != 0:
            raise SystemExit(code)
