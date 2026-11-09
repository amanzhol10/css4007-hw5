import json

import pytest

from helpdesk.tools import check_service_status, find_instructions, prepare_ticket


def test_find_instructions_wifi_and_password():
    wifi = find_instructions("wifi")
    assert wifi["found"] and wifi["instruction_id"] == "GUIDE-WIFI" and len(wifi["steps"]) == 4
    assert find_instructions("Wi-Fi")["topic"] == "wifi"
    assert find_instructions("password")["instruction_id"] == "GUIDE-PASSWORD"


def test_find_instructions_unknown_topic_is_not_invented():
    result = find_instructions("printer")
    assert result["found"] is False
    assert "steps" not in result


@pytest.mark.parametrize(
    "profile, service, status",
    [
        ("normal", "wifi", "operational"),
        ("wifi_outage", "wifi", "outage"),
        ("status_unavailable", "wifi", "unknown"),
        ("recovery_outage", "account_recovery", "outage"),
        ("recovery_outage", "password", "outage"),
        ("normal", "account_recovery", "operational"),
    ],
)
def test_check_service_status_follows_selected_snapshot(profile, service, status):
    assert check_service_status(service, profile)["status"] == status


def test_check_service_status_unknown_input():
    assert check_service_status("printer", "normal")["found"] is False
    assert check_service_status("wifi", "no-such-profile")["found"] is False


def test_unavailable_status_is_not_reported_as_operational_or_outage():
    result = check_service_status("wifi", "status_unavailable")
    assert result["status"] not in ("operational", "outage")


def test_prepare_ticket_writes_local_draft_with_template_fields(tickets):
    result = prepare_ticket(
        "wifi",
        "Wi-Fi still fails",
        ["Enabled Wi-Fi", "Forgot the saved network"],
        user_id="demo-user-a",
        thread_id="t1",
        ticket_dir=tickets,
    )
    assert result["created"]
    saved = json.loads((tickets / f"{result['ticket_id']}.json").read_text(encoding="utf-8"))
    assert set(saved) == {
        "ticket_id", "status", "user_id", "thread_id", "category", "summary",
        "steps_tried", "service_status", "status_profile", "reference_ids",
    }
    assert saved["status"] == "draft"
    assert saved["steps_tried"] == ["Enabled Wi-Fi", "Forgot the saved network"]
    assert saved["reference_ids"] == ["GUIDE-WIFI"]
    assert saved["service_status"]["status"] == "not_checked"


def test_prepare_ticket_refuses_missing_details_and_creates_no_file(tickets):
    result = prepare_ticket("wifi", "", [], ticket_dir=tickets)
    assert result["created"] is False
    assert "problem description" in result["missing"]
    assert "steps already tried" in result["missing"]
    assert not tickets.exists() or not list(tickets.iterdir())


def test_prepare_ticket_refuses_secrets(tickets):
    result = prepare_ticket("password", "my password is hunter2", ["tried"], ticket_dir=tickets)
    assert result["created"] is False
    assert not tickets.exists() or not list(tickets.iterdir())


def test_prepare_ticket_keeps_given_service_status(tickets):
    status = {"service": "wifi", "status": "operational", "message": "ok", "profile": "normal"}
    result = prepare_ticket("wifi", "fails", ["Enabled Wi-Fi"], service_status=status, ticket_dir=tickets)
    assert result["ticket"]["service_status"]["status"] == "operational"
