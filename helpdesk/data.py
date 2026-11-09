import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def load_json(name):
    with open(DATA_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


def helpdesk_data():
    return load_json("helpdesk.json")


def status_profiles():
    return load_json("service_status.json")["profiles"]


def scenario_list():
    return load_json("scenarios.json")["scenarios"]


def ticket_fields():
    return load_json("ticket_template.json")["ticket"]


def topic_in_text(topic, text):
    lowered = text.lower()
    return any(alias in lowered for alias in helpdesk_data()["topic_aliases"][topic])


def format_instructions(topic):
    entry = helpdesk_data()["instructions"][topic]
    lines = [f"{entry['title']} ({entry['instruction_id']}):"]
    lines += [f"{number}. {step}" for number, step in enumerate(entry["steps"], 1)]
    return "\n".join(lines)
