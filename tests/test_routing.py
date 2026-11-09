import pytest

from helpdesk.part1 import build_graph, run_traced


@pytest.mark.parametrize(
    "message, expected",
    [
        ("I cannot connect to Wi-Fi.", ["START", "read_request", "wifi_help", "END"]),
        ("I forgot my PASSWORD.", ["START", "read_request", "password_help", "END"]),
        ("My laptop screen is broken.", ["START", "read_request", "other_help", "END"]),
        ("My Wi-Fi password does not work.", ["START", "read_request", "wifi_help", "END"]),
        ("wifi is slow", ["START", "read_request", "wifi_help", "END"]),
        ("WIFI IS SLOW", ["START", "read_request", "wifi_help", "END"]),
    ],
)
def test_original_graph_routes(message, expected):
    assert run_traced(build_graph("original"), message)["route"] == expected


def test_original_graph_help_nodes_return_reference_instructions():
    wifi = run_traced(build_graph("original"), "wifi")["final_state"]["reply"]
    password = run_traced(build_graph("original"), "password")["final_state"]["reply"]
    assert "GUIDE-WIFI" in wifi and "Campus-Demo" in wifi
    assert "GUIDE-PASSWORD" in password and "Forgot password" in password


def test_other_help_asks_which_topic():
    reply = run_traced(build_graph("original"), "My laptop screen is broken.")["final_state"]["reply"]
    assert "Wi-Fi" in reply and "password" in reply


def test_trace_records_state_after_every_node():
    result = run_traced(build_graph("original"), "I forgot my PASSWORD.")
    after_read, after_help = result["trace"]
    assert after_read["state_after"] == {"message": "I forgot my PASSWORD.", "category": "password"}
    assert "reply" in after_help["state_after"]
    assert after_help["state_after"]["message"] == "I forgot my PASSWORD."


def test_changed_graph_sends_two_topics_to_clarify_problem():
    result = run_traced(build_graph("with_clarification"), "My Wi-Fi password does not work.")
    assert result["route"] == ["START", "read_request", "clarify_problem", "END"]
    assert result["final_state"]["category"] == "both"


@pytest.mark.parametrize(
    "message, node",
    [
        ("I cannot connect to Wi-Fi.", "wifi_help"),
        ("I forgot my PASSWORD.", "password_help"),
        ("My laptop screen is broken.", "other_help"),
    ],
)
def test_changed_graph_keeps_single_topic_routes(message, node):
    assert run_traced(build_graph("with_clarification"), message)["route"][2] == node


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        build_graph("nonsense")
