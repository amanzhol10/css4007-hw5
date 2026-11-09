# HW5 - University IT helpdesk with LangGraph

CSS-4007. The original assignment text is in [ASSIGNMENT.md](ASSIGNMENT.md). My answers, drawings and results are in [SUBMISSION.md](SUBMISSION.md).

Everything uses fictional data from `data/` and invented users. No real ticket is ever sent.

## Python and packages

Developed and tested with Python 3.12.3 (CI also uses 3.12). Check your own version with `python --version` and use 3.10 or newer.

Exact package versions are pinned in `requirements.txt` (langgraph 1.2.20, langgraph-checkpoint-sqlite 3.1.3, langchain-core 1.7.6, langchain-openai 1.7.4).

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Configure a model (only for the real-model runs)

Part 1, the tests and the `--model fake` runs need no key.

```powershell
copy .env.example .env
```

Open `.env` and set `OPENAI_API_KEY` and `MODEL_NAME` (any tool-capable OpenAI chat model you have access to). `.env` is ignored by git.

## Run it

All commands run from the repository root.

Part 1, no model:

```powershell
python -m helpdesk part1 --message "I cannot connect to Wi-Fi."
python -m helpdesk part1 --message "My Wi-Fi password does not work." --variant with_clarification
python -m helpdesk part1 --variant with_clarification --mermaid
```

The three tools called directly, without the agent:

```powershell
python -m helpdesk tool find_instructions --topic wifi
python -m helpdesk tool check_service_status --service wifi --profile wifi_outage
python -m helpdesk tool prepare_ticket --category wifi --summary "Wi-Fi still fails" --steps "Enabled Wi-Fi" "Forgot the saved network"
```

The agent, one message per process (this is how restarts are demonstrated). `--profile` is one of `normal`, `wifi_outage`, `recovery_outage`, `status_unavailable`:

```powershell
python -m helpdesk chat --user demo-user-a --thread t1 --profile normal --message "Show me the Wi-Fi troubleshooting steps."
python -m helpdesk chat --user demo-user-a --thread t1 --message "I already tried that."
```

Interactive mode (omit `--message`, an empty line quits):

```powershell
python -m helpdesk chat --user demo-user-a --thread t1
```

Add `--model fake` to any `chat` or `scenario` command to use the offline rule-based stand-in instead of the real model.

## Run the supplied scenarios S01-S14

```powershell
python -m helpdesk scenario S06
python -m helpdesk scenario all
```

Real model (needs `.env`), logs go to `outputs/real-model-logs/`:

```powershell
python -m helpdesk scenario all --model openai
```

Offline fake model, logs go to `outputs/fake-model-logs/`:

```powershell
python -m helpdesk scenario all --model fake --out outputs/fake-model-logs
```

The runner resets storage when a scenario says `reset_before`. On a `restart_process` event it really exits the current Python process and launches a new `python -m helpdesk scenario ... --start-event N` process on the same storage. Every log event records the process id, so you can see the change.

## Where memory is saved and how to clear it

Storage directory: `local-storage/` (override with `HELPDESK_STORAGE_DIR` or `--storage`). It is ignored by git.

| File | Contents |
| --- | --- |
| `local-storage/checkpoints.sqlite` | LangGraph conversation state, one checkpoint stream per `user/thread` |
| `local-storage/user_memory.sqlite` | saved user preferences, keyed by user id |

Ticket drafts are JSON files in `outputs/tickets/` (or `<--out>/tickets/` for scenario runs).

Clear all memory and start again:

```powershell
python -m helpdesk reset
```

or delete the `local-storage` folder.

## Tests and CI

```powershell
python -m pytest -q
```

The tests use `FakePolicyModel` (a clearly labelled rule-based stand-in, see `helpdesk/fake_model.py`) and temporary directories, so they need no API key and never touch `local-storage/`. GitHub Actions runs them on every push (`.github/workflows/tests.yml`). The real-model demonstration is separate and is never run by CI.

## Layout

```text
helpdesk/part1.py        Part 1 graphs (original and with clarify_problem)
helpdesk/tools.py        the three tools, the preference tool, schemas shown to the model
helpdesk/agent.py        Part 2/3 graph: load_context -> agent <-> tools, loop limit
helpdesk/memory.py       persistent user preferences (SQLite)
helpdesk/app.py          sessions, SQLite checkpointer, per-turn trace
helpdesk/fake_model.py   offline stand-in model for tests
helpdesk/scenarios.py    S01-S14 runner with real process restarts
tests/                   pytest suite
outputs/                 saved logs, diagrams, ticket drafts
```
