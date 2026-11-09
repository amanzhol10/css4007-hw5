import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .app import HelpdeskApp, default_storage_dir, make_model, reset_storage
from .part1 import build_graph, run_traced
from .scenarios import run_all, run_scenario
from .tools import check_service_status, find_instructions, prepare_ticket


def print_trace(result):
    print("  nodes: " + " -> ".join(result["nodes"]))
    for source in result["sources"]:
        print("  source: " + source)
    if result["limit_reached"]:
        print("  loop limit reached")


def cmd_part1(args):
    graph = build_graph(args.variant)
    if args.mermaid:
        print(graph.get_graph().draw_mermaid())
        return
    result = run_traced(graph, args.message)
    print("route: " + " -> ".join(result["route"]))
    for step in result["trace"]:
        print(f"after {step['node']}: {json.dumps(step['state_after'], ensure_ascii=False)}")
    print("reply:\n" + result["final_state"]["reply"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_chat(args):
    model, label = make_model(args.model)
    print(f"model: {label}")
    with HelpdeskApp(
        model,
        storage_dir=args.storage,
        ticket_dir=args.tickets,
        status_profile=args.profile,
    ) as app:
        print(f"user={args.user} thread={args.thread} profile={args.profile} pid={os.getpid()}")
        if args.message:
            result = app.send(args.user, args.thread, args.message)
            print("Helpdesk: " + result["reply"])
            print_trace(result)
            if args.log:
                Path(args.log).parent.mkdir(parents=True, exist_ok=True)
                Path(args.log).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            return
        print("Type a message, or an empty line to quit.")
        while True:
            try:
                message = input("You: ").strip()
            except EOFError:
                break
            if not message:
                break
            result = app.send(args.user, args.thread, message)
            print("Helpdesk: " + result["reply"])
            print_trace(result)


def cmd_tool(args):
    if args.name == "find_instructions":
        result = find_instructions(args.topic)
    elif args.name == "check_service_status":
        result = check_service_status(args.service, args.profile)
    else:
        result = prepare_ticket(
            args.category, args.summary, args.steps, ticket_dir=args.tickets, status_profile=args.profile
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_scenario(args):
    out_dir = Path(args.out)
    if args.id.lower() == "all":
        run_all(args.model, out_dir, args.storage)
    else:
        run_scenario(args.id, args.model, out_dir, args.storage, args.start_event)


def cmd_reset(args):
    reset_storage(args.storage)
    print(f"cleared {Path(args.storage).as_posix()}")


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m helpdesk")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("part1")
    p1.add_argument("--message", default="")
    p1.add_argument("--variant", choices=["original", "with_clarification"], default="original")
    p1.add_argument("--mermaid", action="store_true")
    p1.add_argument("--out")
    p1.set_defaults(func=cmd_part1)

    chat = sub.add_parser("chat")
    chat.add_argument("--user", required=True)
    chat.add_argument("--thread", required=True)
    chat.add_argument("--profile", default="normal")
    chat.add_argument("--model", choices=["fake", "openai"], default="openai")
    chat.add_argument("--message")
    chat.add_argument("--storage", default=str(default_storage_dir()))
    chat.add_argument("--tickets", default="outputs/tickets")
    chat.add_argument("--log")
    chat.set_defaults(func=cmd_chat)

    tool = sub.add_parser("tool")
    tool.add_argument("name", choices=["find_instructions", "check_service_status", "prepare_ticket"])
    tool.add_argument("--topic", default="wifi")
    tool.add_argument("--service", default="wifi")
    tool.add_argument("--profile", default="normal")
    tool.add_argument("--category", default="wifi")
    tool.add_argument("--summary", default="")
    tool.add_argument("--steps", nargs="*", default=[])
    tool.add_argument("--tickets", default="outputs/tickets")
    tool.set_defaults(func=cmd_tool)

    scenario = sub.add_parser("scenario")
    scenario.add_argument("id")
    scenario.add_argument("--model", choices=["fake", "openai"], default="openai")
    scenario.add_argument("--out", default="outputs/real-model-logs")
    scenario.add_argument("--storage", default=str(default_storage_dir()))
    scenario.add_argument("--start-event", type=int, default=0)
    scenario.set_defaults(func=cmd_scenario)

    reset = sub.add_parser("reset")
    reset.add_argument("--storage", default=str(default_storage_dir()))
    reset.set_defaults(func=cmd_reset)
    return parser


def main(argv=None):
    load_dotenv()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    args.func(args)
