import json
import os
import shutil
import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .agent import build_agent_graph
from .fake_model import FAKE_MODEL_LABEL, FakePolicyModel
from .memory import UserMemory


def default_storage_dir():
    return Path(os.environ.get("HELPDESK_STORAGE_DIR", "local-storage"))


def reset_storage(storage_dir):
    storage_dir = Path(storage_dir)
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)


def make_model(kind):
    if kind == "fake":
        return FakePolicyModel(), FAKE_MODEL_LABEL
    if kind == "openai":
        from langchain_openai import ChatOpenAI

        name = os.environ.get("MODEL_NAME")
        if not os.environ.get("OPENAI_API_KEY") or not name:
            raise SystemExit("Set OPENAI_API_KEY and MODEL_NAME in the environment or in .env first.")
        return ChatOpenAI(model=name), f"OpenAI via langchain-openai, model {name}"
    raise SystemExit(f"unknown model kind: {kind}")


def text_of(message):
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def describe_message(message):
    entry = {"type": message.type, "content": text_of(message)}
    if isinstance(message, AIMessage) and message.tool_calls:
        entry["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in message.tool_calls]
    if isinstance(message, ToolMessage):
        entry["tool"] = message.name
        try:
            entry["result"] = json.loads(message.content)
        except ValueError:
            entry["result"] = message.content
    return entry


def describe_update(node, update):
    summary = {}
    for key, value in update.items():
        if key == "messages":
            summary["messages"] = [describe_message(m) for m in value]
        else:
            summary[key] = value
    return {"node": node, "update": summary}


class HelpdeskApp:
    def __init__(self, model, storage_dir=None, ticket_dir="outputs/tickets", status_profile="normal"):
        self.storage_dir = Path(storage_dir or default_storage_dir())
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.ticket_dir = str(ticket_dir)
        self.status_profile = status_profile
        self.connection = sqlite3.connect(self.storage_dir / "checkpoints.sqlite", check_same_thread=False)
        self.saver = SqliteSaver(self.connection)
        self.memory = UserMemory(self.storage_dir / "user_memory.sqlite")
        self.graph = build_agent_graph(model, self.memory, self.saver)

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def config(self, user_id, thread_id):
        return {
            "configurable": {
                "thread_id": f"{user_id}/{thread_id}",
                "conversation_id": thread_id,
                "user_id": user_id,
                "status_profile": self.status_profile,
                "ticket_dir": self.ticket_dir,
            },
            "recursion_limit": 30,
        }

    def send(self, user_id, thread_id, message):
        config = self.config(user_id, thread_id)
        payload = {
            "messages": [HumanMessage(content=message)],
            "tool_rounds": 0,
            "limit_reached": False,
            "sources": [],
        }
        trace = []
        for chunk in self.graph.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                trace.append(describe_update(node, update))
        state = self.graph.get_state(config).values
        messages = state["messages"]
        reply = next((text_of(m) for m in reversed(messages) if isinstance(m, AIMessage) and text_of(m)), "")
        tickets = []
        start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        for m in messages[start:]:
            if isinstance(m, ToolMessage) and m.name == "prepare_ticket":
                result = json.loads(m.content)
                if result.get("created"):
                    tickets.append(result["path"])
        return {
            "user_id": user_id,
            "thread_id": thread_id,
            "status_profile": self.status_profile,
            "message": message,
            "reply": reply,
            "nodes": [step["node"] for step in trace],
            "trace": trace,
            "sources": state.get("sources", []),
            "tickets": tickets,
            "limit_reached": bool(state.get("limit_reached")),
        }

    def conversation_state(self, user_id, thread_id):
        return self.graph.get_state(self.config(user_id, thread_id)).values
