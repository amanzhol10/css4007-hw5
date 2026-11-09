from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .data import format_instructions, topic_in_text


class HelpdeskState(TypedDict, total=False):
    message: str
    category: str
    reply: str


def classify_original(message):
    if topic_in_text("wifi", message):
        return "wifi"
    if topic_in_text("password", message):
        return "password"
    return "other"


def classify_with_clarification(message):
    has_wifi = topic_in_text("wifi", message)
    has_password = topic_in_text("password", message)
    if has_wifi and has_password:
        return "both"
    if has_wifi:
        return "wifi"
    if has_password:
        return "password"
    return "other"


def make_read_request(classifier):
    def read_request(state):
        return {"category": classifier(state["message"])}

    return read_request


def wifi_help(state):
    return {"reply": format_instructions("wifi")}


def password_help(state):
    return {"reply": format_instructions("password")}


def other_help(state):
    return {"reply": "Is your problem about Wi-Fi or about a password?"}


def clarify_problem(state):
    return {
        "reply": "Is the problem connecting to Wi-Fi, or recovering your account password? "
        "Tell me which one you want to solve first."
    }


ROUTES = {
    "wifi": "wifi_help",
    "password": "password_help",
    "other": "other_help",
    "both": "clarify_problem",
}


def choose_next_node(state):
    return ROUTES[state["category"]]


def build_graph(variant="original"):
    graph = StateGraph(HelpdeskState)
    if variant == "with_clarification":
        graph.add_node("read_request", make_read_request(classify_with_clarification))
        graph.add_node("clarify_problem", clarify_problem)
        targets = ["wifi_help", "password_help", "other_help", "clarify_problem"]
    elif variant == "original":
        graph.add_node("read_request", make_read_request(classify_original))
        targets = ["wifi_help", "password_help", "other_help"]
    else:
        raise ValueError(f"unknown graph variant: {variant}")
    graph.add_node("wifi_help", wifi_help)
    graph.add_node("password_help", password_help)
    graph.add_node("other_help", other_help)
    graph.add_edge(START, "read_request")
    graph.add_conditional_edges("read_request", choose_next_node, targets)
    for node in targets:
        graph.add_edge(node, END)
    return graph.compile()


def run_traced(graph, message):
    current = {"message": message}
    trace = []
    for chunk in graph.stream({"message": message}, stream_mode="updates"):
        for node, update in chunk.items():
            current = {**current, **update}
            trace.append({"node": node, "update": dict(update), "state_after": dict(current)})
    return {
        "message": message,
        "route": ["START"] + [step["node"] for step in trace] + ["END"],
        "trace": trace,
        "final_state": current,
    }
