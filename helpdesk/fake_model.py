import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .data import helpdesk_data

FAKE_MODEL_LABEL = "FAKE rule-based model (not an LLM; used for tests and offline runs)"

STEP_PATTERNS = [
    (r"\benabled (it|wi-?fi)\b|\bturned (it|wi-?fi) on\b", "Enabled Wi-Fi"),
    (r"forgot the saved network|forgot the network", "Forgot the saved network"),
    (r"\breconnect(ed|ing)\b", "Tried reconnecting"),
    (r"(clicked|chose|used) forgot password", "Used the Forgot password option"),
]
SERVICES = {"wifi": "Wi-Fi", "password": "account recovery"}


def extract_steps(texts):
    steps = []
    for text in texts:
        lowered = text.lower()
        for pattern, label in STEP_PATTERNS:
            if re.search(pattern, lowered) and label not in steps:
                steps.append(label)
    return steps


def detect_topic(text):
    lowered = text.lower()
    if "wifi" in lowered or "wi-fi" in lowered:
        return "wifi"
    if "password" in lowered:
        return "password"
    return None


def parse_json(content):
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return {}


def tool_call(messages, name, args):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{len(messages)}_{name}", "type": "tool_call"}
        ],
    )


def decide(messages):
    human_positions = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    last = human_positions[-1]
    text = messages[last].content
    lowered = text.lower()
    earlier_humans = [m.content for m in messages[:last] if isinstance(m, HumanMessage)]
    system_text = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
    done = {}
    for message in messages[last + 1 :]:
        if isinstance(message, ToolMessage):
            done[message.name] = parse_json(message.content)
    prior_tickets = [
        m for m in messages[:last] if isinstance(m, ToolMessage) and m.name == "prepare_ticket"
        and parse_json(m.content).get("created")
    ]
    last_ai_text = next(
        (m.content for m in reversed(messages[:last]) if isinstance(m, AIMessage) and m.content),
        "",
    )
    prior_instructions = next(
        (
            parse_json(m.content)
            for m in reversed(messages[:last])
            if isinstance(m, ToolMessage) and m.name == "find_instructions"
        ),
        None,
    )

    preference = re.search(r"prefer\w*\s+(?:replies\s+)?in\s+([A-Za-z]+)", text, re.IGNORECASE)
    if preference and "save_user_preference" not in done:
        return tool_call(messages, "save_user_preference", {"language": preference.group(1)})
    prefix = ""
    if "save_user_preference" in done and done["save_user_preference"].get("saved"):
        language = done["save_user_preference"]["value"]
        prefix = (
            f"Saved: I will remember that you prefer replies in {language}. "
            "(This offline demo model cannot translate, so this reply stays in English.)\n"
        )

    def finish(body):
        return AIMessage(content=prefix + body)

    topic = detect_topic(text) or next(
        (t for t in (detect_topic(h) for h in reversed(earlier_humans)) if t), None
    )
    steps = extract_steps(earlier_humans + [text])
    wants_status = bool(
        re.search(
            r"check (whether|if|the)\b.*\b(down|status)|is (it|wi-?fi) down|known problem|service status",
            lowered,
        )
    )
    wants_ticket = "ticket" in lowered and bool(re.search(r"prepare|create|draft|make|help me", lowered))
    pending_ticket = (
        not wants_ticket
        and not prior_tickets
        and any(
            "ticket" in h.lower() and re.search(r"prepare|create|draft|make", h.lower())
            for h in earlier_humans
        )
    )
    affirmative = bool(re.match(r"\s*(yes|yeah|ok|okay|sure)\b", lowered)) and "ticket draft" in last_ai_text

    def ticket_summary():
        label = "Wi-Fi connection" if topic == "wifi" else "Account password recovery"
        return f"{label} still fails after the steps the user already tried."

    def ticket_call():
        return tool_call(
            messages,
            "prepare_ticket",
            {"category": topic, "summary": ticket_summary(), "steps_tried": steps},
        )

    def ticket_done_message(result):
        if not result.get("created"):
            return finish(
                "I need more details before preparing a ticket: " + ", ".join(result.get("missing", []))
                + ". What problem are you having, and what have you already tried?"
            )
        status = result["ticket"]["service_status"]
        if status["status"] == "not_checked":
            status_text = "The service status was not checked."
        else:
            status_text = f"Known service status: {status['status']}."
        return finish(
            f"I prepared a local ticket draft {result['ticket_id']} (file: {result['path']}). "
            f"It records the problem and the steps you told me you already tried: {', '.join(steps)}. "
            f"{status_text} Nothing has been sent to a real helpdesk."
        )

    if wants_status:
        if topic is None:
            return finish("Which service should I check: Wi-Fi or account recovery?")
        service = "wifi" if topic == "wifi" else "account_recovery"
        if "check_service_status" not in done:
            return tool_call(messages, "check_service_status", {"service": service})
        status = done["check_service_status"]
        name = SERVICES[topic]
        if not status.get("found"):
            return finish(f"I could not look up the {name} status.")
        if not wants_ticket:
            return finish(f"{status['message']} Status: {status['status']}.")
        if status["status"] == "operational":
            if "prepare_ticket" in done:
                return ticket_done_message(done["prepare_ticket"])
            if not steps:
                return finish(
                    f"{status['message']} What have you already tried, so I can prepare the ticket?"
                )
            return ticket_call()
        if status["status"] == "outage":
            return finish(
                f"{status['message']} Your request said to prepare a ticket only if the service was "
                "operating, so I have not prepared one."
            )
        return finish(
            f"I could not determine the {name} status. I cannot confirm that it is operating. "
            "Would you like a ticket draft explaining that the status could not be checked?"
        )

    if wants_ticket or pending_ticket or affirmative:
        if not topic or not steps:
            return finish("What problem are you having, and what have you already tried?")
        if "prepare_ticket" not in done:
            return ticket_call()
        return ticket_done_message(done["prepare_ticket"])

    parts = []
    if re.search(r"which language|preferred language|language did i", lowered):
        match = re.search(r"reply language: ([A-Za-z \-]+)", system_text)
        if match:
            parts.append(f"You asked me to reply in {match.group(1).strip()}.")
        else:
            parts.append("No preferred language has been recorded for you.")
    if re.search(r"which (problem|steps)|what steps|steps did i|did we discuss|previous problem", lowered):
        history_steps = extract_steps(earlier_humans)
        history_topic = next(
            (t for t in (detect_topic(h) for h in reversed(earlier_humans)) if t), None
        )
        if history_topic:
            label = "Wi-Fi" if history_topic == "wifi" else "password recovery"
            detail = (
                f" and said you already tried: {', '.join(history_steps)}"
                if history_steps
                else " but did not list any steps you tried"
            )
            parts.append(f"In this conversation you reported a {label} problem{detail}.")
        else:
            parts.append("No previous problem has been recorded for you in this conversation.")
    if re.search(r"what can you (help|do)", lowered):
        scope = "; ".join(helpdesk_data()["scope"])
        parts.append(f"I can help with: {scope}.")
    if parts:
        return finish(" ".join(parts))

    if "already tried that" in lowered:
        if prior_instructions and prior_instructions.get("found"):
            numbered = "; ".join(
                f"{n}. {s}" for n, s in enumerate(prior_instructions["steps"], 1)
            )
            return finish(
                f"By 'that' I understand the {prior_instructions['instruction_id']} steps I gave "
                f"you earlier: {numbered}. Since they did not help, I can check the service "
                "status and prepare a ticket if you tell me what happened."
            )
        return finish("I do not see earlier instructions in this conversation. What did you try?")

    if topic and not steps:
        if "find_instructions" not in done:
            return tool_call(messages, "find_instructions", {"topic": topic})
        result = done["find_instructions"]
        if result.get("found"):
            numbered = "\n".join(f"{n}. {s}" for n, s in enumerate(result["steps"], 1))
            return finish(f"{result['title']} ({result['instruction_id']}):\n{numbered}")
        return finish("I could not find instructions for that topic.")

    if topic and steps:
        label = "Wi-Fi" if topic == "wifi" else "password recovery"
        return finish(
            f"Noted: a {label} problem. Steps you already tried: {', '.join(steps)}. "
            "I can check the service status or prepare a ticket draft if you want."
        )

    return finish(
        "I can help with Wi-Fi troubleshooting, password recovery guidance, service status "
        "lookup and local ticket drafts. Is your problem about Wi-Fi or about a password?"
    )


class FakePolicyModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "fake-rule-based-policy-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=decide(messages))])


class RunawayModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "fake-runaway-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = tool_call(messages, "find_instructions", {"topic": "wifi"})
        return ChatResult(generations=[ChatGeneration(message=message)])
