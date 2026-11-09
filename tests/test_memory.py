import json
import os
import subprocess
import sys
from pathlib import Path

from helpdesk.memory import UserMemory

ROOT = Path(__file__).resolve().parent.parent


def test_conversation_survives_closing_and_reopening(make_app):
    first = make_app()
    first.send("demo-user-a", "t1", "My issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails.")
    first.close()
    second = make_app()
    result = second.send("demo-user-a", "t1", "Which problem and steps did we discuss?")
    assert "Wi-Fi" in result["reply"]
    assert "Enabled Wi-Fi" in result["reply"] and "Tried reconnecting" in result["reply"]
    assert any(s.startswith("conversation_state") for s in result["sources"])


def test_preference_survives_closing_and_reopening(make_app, storage):
    first = make_app()
    first.send("demo-user-a", "t1", "Please remember that I prefer replies in Kazakh.")
    first.close()
    assert UserMemory(storage / "user_memory.sqlite").get_all("demo-user-a") == {"language": "Kazakh"}


def test_conversation_survives_a_real_process_restart(tmp_path):
    env = {**os.environ, "HELPDESK_STORAGE_DIR": str(tmp_path / "s")}
    base = [
        sys.executable, "-m", "helpdesk", "chat", "--model", "fake", "--user", "demo-user-a",
        "--thread", "restart-a", "--storage", str(tmp_path / "s"), "--tickets", str(tmp_path / "t"),
    ]
    first = subprocess.run(
        base + ["--message", "My issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails."],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert first.returncode == 0, first.stderr
    second = subprocess.run(
        base + ["--message", "Which problem and steps did we discuss?"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert second.returncode == 0, second.stderr
    assert "Wi-Fi" in second.stdout and "Tried reconnecting" in second.stdout


def test_reading_memory_does_not_depend_on_the_fixture_file(make_app):
    result = make_app().send("demo-user-a", "t1", "Which language did I ask you to use?")
    assert "No preferred language" in result["reply"]
