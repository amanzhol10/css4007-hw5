import json

from langchain_core.messages import ToolMessage

from helpdesk.agent import MAX_TOOL_ROUNDS
from helpdesk.fake_model import RunawayModel

CONDITIONAL = (
    "I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not connect. "
    "Check whether Wi-Fi is down. If it is operating, prepare a ticket about this problem."
)


def tool_names(result):
    names = []
    for step in result["trace"]:
        for message in step["update"].get("messages", []):
            if message["type"] == "ai" and message.get("tool_calls"):
                names += [call["name"] for call in message["tool_calls"]]
    return names


def test_operating_status_leads_to_a_ticket(make_app, tickets):
    result = make_app("normal").send("demo-user-a", "t1", CONDITIONAL)
    assert tool_names(result) == ["check_service_status", "prepare_ticket"]
    assert len(result["tickets"]) == 1
    assert len(list(tickets.iterdir())) == 1


def test_outage_leads_to_no_ticket(make_app, tickets):
    result = make_app("wifi_outage").send("demo-user-a", "t1", CONDITIONAL)
    assert tool_names(result) == ["check_service_status"]
    assert result["tickets"] == []
    assert not tickets.exists() or not list(tickets.iterdir())
    assert "repair time" in result["reply"] or "outage" in result["reply"].lower()


def test_unknown_status_asks_instead_of_creating_a_ticket(make_app):
    result = make_app("status_unavailable").send("demo-user-a", "t1", CONDITIONAL)
    assert tool_names(result) == ["check_service_status"]
    assert result["tickets"] == []
    assert "cannot confirm" in result["reply"]


def test_ticket_uses_status_from_tool_result_not_from_model(make_app, tickets):
    result = make_app("normal").send("demo-user-a", "t1", CONDITIONAL)
    ticket = json.loads(open(result["tickets"][0], encoding="utf-8").read())
    assert ticket["service_status"]["status"] == "operational"
    assert ticket["status_profile"] == "normal"


def test_missing_details_lead_to_a_question_then_a_ticket(make_app):
    app = make_app("normal")
    first = app.send("demo-user-a", "t1", "Prepare a support ticket.")
    assert first["tickets"] == [] and "What problem" in first["reply"]
    second = app.send(
        "demo-user-a", "t1", "The issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails."
    )
    assert len(second["tickets"]) == 1


def test_loop_limit_stops_a_model_that_never_answers(make_app):
    result = make_app("normal", model=RunawayModel()).send("demo-user-a", "t1", "hello")
    assert result["limit_reached"] is True
    assert result["nodes"].count("tools") == MAX_TOOL_ROUNDS
    assert result["nodes"][-1] == "stop_at_limit"
    assert "limit" in result["reply"]


def test_loop_limit_leaves_no_unanswered_tool_calls(make_app):
    app = make_app("normal", model=RunawayModel())
    app.send("demo-user-a", "t1", "hello")
    messages = app.conversation_state("demo-user-a", "t1")["messages"]
    called = {c["id"] for m in messages if getattr(m, "tool_calls", None) for c in m.tool_calls}
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    assert called == answered


def test_loop_counter_resets_on_the_next_turn(make_app):
    app = make_app("normal")
    app.send("demo-user-a", "t1", "Show me the Wi-Fi troubleshooting steps.")
    second = app.send("demo-user-a", "t1", "Is it down? Check whether Wi-Fi is down.")
    assert second["limit_reached"] is False
