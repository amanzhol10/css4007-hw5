import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .data import helpdesk_data, status_profiles, ticket_fields

SECRET_PATTERN = re.compile(
    r"(password|passcode|recovery code|api[ _-]?key|token)\s*(is|=|:)\s*\S+", re.IGNORECASE
)
TOPIC_ALIASES = {"wifi": "wifi", "wi-fi": "wifi", "password": "password"}


def normalize_topic(value):
    return TOPIC_ALIASES.get(str(value).strip().lower())


def find_instructions(topic):
    key = normalize_topic(topic)
    instructions = helpdesk_data()["instructions"]
    if key is None or key not in instructions:
        return {"found": False, "topic": str(topic), "available_topics": sorted(instructions)}
    entry = instructions[key]
    return {
        "found": True,
        "topic": key,
        "title": entry["title"],
        "instruction_id": entry["instruction_id"],
        "steps": list(entry["steps"]),
    }


def resolve_service(service):
    value = str(service).strip().lower()
    mapping = helpdesk_data()["topic_services"]
    if value in mapping:
        return mapping[value]
    if value == "wi-fi":
        return mapping["wifi"]
    if value in mapping.values():
        return value
    return None


def check_service_status(service, profile="normal"):
    profiles = status_profiles()
    if profile not in profiles:
        return {"found": False, "error": f"unknown status profile: {profile}"}
    key = resolve_service(service)
    if key is None or key not in profiles[profile]:
        return {
            "found": False,
            "service": str(service),
            "error": "unknown service",
            "known_services": sorted(profiles[profile]),
        }
    entry = profiles[profile][key]
    return {
        "found": True,
        "service": key,
        "status": entry["status"],
        "message": entry["message"],
        "profile": profile,
    }


def prepare_ticket(
    category,
    summary,
    steps_tried,
    user_id="demo-user-a",
    thread_id="direct",
    status_profile="normal",
    service_status=None,
    ticket_dir="outputs/tickets",
):
    data = helpdesk_data()
    key = normalize_topic(category)
    steps = [str(step).strip() for step in (steps_tried or []) if str(step).strip()]
    summary = (summary or "").strip()
    missing = []
    if key is None or key not in data["instructions"]:
        missing.append("category (wifi or password)")
    if not summary:
        missing.append("problem description")
    if not steps:
        missing.append("steps already tried")
    if missing:
        return {
            "created": False,
            "missing": missing,
            "message": "Ask the user for the missing information. Do not invent it.",
        }
    if any(SECRET_PATTERN.search(text) for text in [summary] + steps):
        return {
            "created": False,
            "missing": [],
            "message": "The text appears to contain a secret. Remove it before preparing a ticket.",
        }
    service = data["topic_services"][key]
    status = service_status or {
        "service": service,
        "status": "not_checked",
        "message": "Service status was not checked in this conversation.",
    }
    ticket = ticket_fields()
    ticket.update(
        {
            "ticket_id": "TCK-" + uuid.uuid4().hex[:8].upper(),
            "status": "draft",
            "user_id": user_id,
            "thread_id": thread_id,
            "category": key,
            "summary": summary,
            "steps_tried": steps,
            "service_status": status,
            "status_profile": status_profile,
            "reference_ids": [data["instructions"][key]["instruction_id"]],
        }
    )
    directory = Path(ticket_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{ticket['ticket_id']}.json"
    path.write_text(json.dumps(ticket, indent=2), encoding="utf-8")
    return {
        "created": True,
        "ticket_id": ticket["ticket_id"],
        "path": path.as_posix(),
        "ticket": ticket,
        "message": "Local draft saved. Nothing was sent to a real helpdesk.",
    }


def save_user_preference(language, user_id, memory):
    value = str(language).strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z \-]{1,29}", value):
        return {"saved": False, "message": "The language name is not valid."}
    memory.set(user_id, "language", value.title())
    return {"saved": True, "key": "language", "value": value.title(), "user_id": user_id}


class FindInstructionsArgs(BaseModel):
    topic: Literal["wifi", "password"] = Field(description="Which instruction set to look up.")


class CheckServiceStatusArgs(BaseModel):
    service: Literal["wifi", "account_recovery"] = Field(
        description="Service to check. Use account_recovery for password problems."
    )


class PrepareTicketArgs(BaseModel):
    category: Literal["wifi", "password"] = Field(description="Problem category.")
    summary: str = Field(description="Short problem description in the user's own words.")
    steps_tried: list[str] = Field(
        description="Steps the user said they already tried. Never invent steps."
    )


class SavePreferenceArgs(BaseModel):
    language: str = Field(description="Language the user asked to receive replies in.")


def schema_only(**kwargs):
    raise RuntimeError("Tool schemas are executed by the graph's tools node, not by the model.")


TOOL_SCHEMAS = [
    StructuredTool.from_function(
        func=schema_only,
        name="find_instructions",
        description="Look up Wi-Fi or password instructions in the local helpdesk guide.",
        args_schema=FindInstructionsArgs,
    ),
    StructuredTool.from_function(
        func=schema_only,
        name="check_service_status",
        description="Look up whether a service is operating, in outage, or unknown.",
        args_schema=CheckServiceStatusArgs,
    ),
    StructuredTool.from_function(
        func=schema_only,
        name="prepare_ticket",
        description=(
            "Prepare a local ticket draft. Only call when the problem and the steps already "
            "tried are known from the user. Nothing is sent anywhere."
        ),
        args_schema=PrepareTicketArgs,
    ),
    StructuredTool.from_function(
        func=schema_only,
        name="save_user_preference",
        description="Remember the language the user wants replies in, for future conversations.",
        args_schema=SavePreferenceArgs,
    ),
]


@dataclass
class ToolContext:
    user_id: str
    thread_id: str
    status_profile: str
    ticket_dir: str
    memory: object
    status_checks: dict = field(default_factory=dict)


def execute_tool(name, args, ctx):
    try:
        if name == "find_instructions":
            return find_instructions(args["topic"])
        if name == "check_service_status":
            result = check_service_status(args["service"], ctx.status_profile)
            if result.get("found"):
                ctx.status_checks[result["service"]] = {
                    "service": result["service"],
                    "status": result["status"],
                    "message": result["message"],
                    "profile": result["profile"],
                }
            return result
        if name == "prepare_ticket":
            key = normalize_topic(args.get("category"))
            service = helpdesk_data()["topic_services"].get(key)
            return prepare_ticket(
                category=args.get("category"),
                summary=args.get("summary"),
                steps_tried=args.get("steps_tried"),
                user_id=ctx.user_id,
                thread_id=ctx.thread_id,
                status_profile=ctx.status_profile,
                service_status=ctx.status_checks.get(service),
                ticket_dir=ctx.ticket_dir,
            )
        if name == "save_user_preference":
            return save_user_preference(args["language"], ctx.user_id, ctx.memory)
        return {"error": f"unknown tool: {name}"}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
