import json
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .tools import TOOL_SCHEMAS, ToolContext, execute_tool

MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = """You are a university IT helpdesk assistant for a practice exercise. All data is fictional.
Scope: Wi-Fi troubleshooting, password recovery guidance, service status lookup, local ticket drafts.

Rules:
- Use find_instructions for instructions. Use check_service_status before claiming anything about an outage.
- Prepare a ticket only with prepare_ticket, and only when the user's problem and the steps they said they already tried are known. Never invent steps or details. If they are missing, ask the user.
- If the user says to prepare a ticket only when the service is operating: check the status first. If it is operating, prepare the ticket. If there is an outage, do not prepare it and say so. If the status is unknown, say you cannot confirm it is operating and ask whether they want a draft.
- Never state a repair time unless a tool result gives one. Never ask for or store passwords, API keys or recovery codes.
- Tickets are local drafts only. Say that nothing was sent to a real helpdesk.
- When the user asks you to remember a language preference, call save_user_preference.
- Use the earlier messages in this conversation to understand references like "that".
- {language_line}"""


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_prefs: dict
    status_checks: dict
    tool_rounds: int
    limit_reached: bool
    sources: list


def build_system_prompt(prefs):
    language = prefs.get("language")
    if language:
        line = f"Saved user memory - reply language: {language}. Write your replies in {language}."
    else:
        line = "Saved user memory: nothing is stored for this user yet."
    return SYSTEM_PROMPT.format(language_line=line)


def build_agent_graph(model, memory, checkpointer):
    bound_model = model.bind_tools(TOOL_SCHEMAS)

    def load_context(state, config):
        settings = config["configurable"]
        user_id = settings["user_id"]
        prefs = memory.get_all(user_id)
        sources = []
        earlier = len(state.get("messages") or []) - 1
        if earlier > 0:
            sources.append(
                f"conversation_state: {earlier} earlier messages loaded from the checkpoint of this thread"
            )
        if prefs:
            sources.append(f"user_memory[{user_id}]: {prefs}")
        else:
            sources.append(f"user_memory[{user_id}]: empty")
        return {"user_prefs": prefs, "sources": sources}

    def agent(state, config):
        prompt = build_system_prompt(state.get("user_prefs") or {})
        response = bound_model.invoke([SystemMessage(content=prompt)] + state["messages"])
        return {"messages": [response]}

    def tools(state, config):
        settings = config["configurable"]
        ctx = ToolContext(
            user_id=settings["user_id"],
            thread_id=settings["conversation_id"],
            status_profile=settings["status_profile"],
            ticket_dir=settings["ticket_dir"],
            memory=memory,
            status_checks=dict(state.get("status_checks") or {}),
        )
        sources = list(state.get("sources") or [])
        results = []
        for call in state["messages"][-1].tool_calls:
            result = execute_tool(call["name"], call["args"], ctx)
            if call["name"] == "find_instructions" and result.get("found"):
                sources.append(f"reference_file: helpdesk.json {result['instruction_id']}")
            if call["name"] == "check_service_status" and result.get("found"):
                sources.append(f"reference_file: service_status.json profile={result['profile']}")
            if call["name"] == "save_user_preference" and result.get("saved"):
                sources.append(f"user_memory[{ctx.user_id}]: wrote language={result['value']}")
            results.append(
                ToolMessage(content=json.dumps(result), tool_call_id=call["id"], name=call["name"])
            )
        return {
            "messages": results,
            "status_checks": ctx.status_checks,
            "tool_rounds": state.get("tool_rounds", 0) + 1,
            "sources": sources,
        }

    def stop_at_limit(state, config):
        pending = state["messages"][-1].tool_calls
        skipped = [
            ToolMessage(
                content='{"error": "not executed: tool-call limit reached"}',
                tool_call_id=call["id"],
                name=call["name"],
            )
            for call in pending
        ]
        final = AIMessage(
            content=(
                "I reached the limit on tool calls for one request and stopped. "
                "Please rephrase the request or ask for one thing at a time."
            )
        )
        return {"messages": skipped + [final], "limit_reached": True}

    def route_after_agent(state):
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            return "stop_at_limit"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("load_context", load_context)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("stop_at_limit", stop_at_limit)
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "agent")
    graph.add_conditional_edges("agent", route_after_agent, ["tools", "stop_at_limit", END])
    graph.add_edge("tools", "agent")
    graph.add_edge("stop_at_limit", END)
    return graph.compile(checkpointer=checkpointer)
