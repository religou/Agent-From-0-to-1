from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TASK = "请把 inbox 里的 a.txt 移动到 archive，然后生成一份简单的 Python 代码。"
WORKSPACE = (BASE_DIR / "demo_workspace").resolve()
FILE_AGENT_NAME = "file_agent"
CODE_AGENT_NAME = "code_agent"
FINISH = "finish"


class MultiAgentState(TypedDict):
    task: str
    next_agent: str
    file_report: str
    code_report: str
    final_answer: str


def reset_workspace() -> None:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    (WORKSPACE / "inbox").mkdir(parents=True)
    (WORKSPACE / "archive").mkdir()
    (WORKSPACE / "inbox" / "a.txt").write_text(
        "Hello from MokioClaw MultiAgent demo.",
        encoding="utf-8",
    )


def workspace_path(path: str) -> Path:
    normalized = path.strip().replace("\\", "/").strip("/")
    candidate = WORKSPACE if normalized in {"", "."} else WORKSPACE / normalized
    resolved = candidate.resolve()

    if resolved != WORKSPACE and WORKSPACE not in resolved.parents:
        raise ValueError("Path must stay inside demo_workspace.")

    return resolved


def workspace_items(target: Path, *, files_only: bool = False) -> list[Path]:
    if not target.exists():
        return []
    if target.is_file():
        return [target]

    items = sorted(target.rglob("*"))
    return [item for item in items if item.is_file()] if files_only else items


def render_workspace(items: Iterable[Path]) -> str:
    lines = []
    for item in items:
        rel = item.relative_to(WORKSPACE).as_posix()
        lines.append(f"- {rel}/" if item.is_dir() else f"- {rel}")
    return "\n".join(lines) or "(empty)"


def show_workspace() -> str:
    return render_workspace(workspace_items(WORKSPACE))


@tool
def list_files(path: str = ".") -> str:
    """List files in the demo workspace."""
    target = workspace_path(path)
    return render_workspace(workspace_items(target, files_only=True))


@tool
def move_file(source: str, target: str) -> str:
    """Move one file in the demo workspace."""
    source_path = workspace_path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")

    target_path = workspace_path(target)
    if "." not in target_path.name:
        target_path = target_path / source_path.name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(target_path))
    return f"moved {source} -> {target_path.relative_to(WORKSPACE).as_posix()}"


@tool
def write_file(path: str, content: str) -> str:
    """Write a file in the demo workspace."""
    target_path = workspace_path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")
    return f"created {target_path.relative_to(WORKSPACE).as_posix()}"


def load_llm() -> ChatOpenAI:
    load_dotenv(BASE_DIR.parent / ".env")
    return ChatOpenAI(
        model=os.getenv("MODEL", "qwen3.6-flash"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0,
    )


SUPERVISOR_PROMPT = """
你是 supervisor agent。
你负责决定下一个应该执行的 agent。

可选 agent：
- file_agent：整理文件
- code_agent：生成代码
- finish：所有任务完成

只输出一个词，并且只能从当前允许的 agent 中选择。
""".strip()

SUMMARY_PROMPT = """
你是 supervisor agent。
根据 file_agent 和 code_agent 的报告，用中文做一个简短总结。
""".strip()

FILE_AGENT_PROMPT = """
你是 file_agent，只负责文件整理。
你可以使用 list_files 和 move_file。
你需要根据用户任务自行决定要查看哪些目录、移动哪些文件，并汇报目录变化。
不要写代码。
""".strip()

CODE_AGENT_PROMPT = """
你是 code_agent，只负责生成代码文件。
你可以使用 write_file。
你需要根据用户任务和 file_agent 的报告，自行决定要生成什么代码文件。
不要移动文件。
""".strip()


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("text"):
                parts.append(str(chunk["text"]))
            elif hasattr(chunk, "text") and getattr(chunk, "text"):
                parts.append(str(getattr(chunk, "text")))
            else:
                parts.append(str(chunk))
        return "\n".join(part for part in parts if part).strip() or str(content)
    return str(content)


def last_message_text(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        return ""

    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, "content") else last_message.get("content", "")
    return content_to_text(content)


def allowed_next_agents(state: MultiAgentState) -> tuple[str, ...]:
    if not state["file_report"]:
        return (FILE_AGENT_NAME,)
    if not state["code_report"]:
        return (CODE_AGENT_NAME,)
    return (FINISH,)


def normalize_decision(decision: object, allowed: tuple[str, ...]) -> str:
    cleaned = str(decision).strip()
    return cleaned if cleaned in allowed else allowed[0]


def summarize_reports(llm: ChatOpenAI, state: MultiAgentState) -> str:
    final_answer = llm.invoke(
        [
            SystemMessage(content=SUMMARY_PROMPT),
            HumanMessage(
                content=(
                    f"file_agent:\n{state['file_report']}\n\n"
                    f"code_agent:\n{state['code_report']}"
                )
            ),
        ]
    ).content
    return content_to_text(final_answer)


def invoke_specialist(agent: Any, *, label: str, instruction: str) -> str:
    print(f"\n[{label} node] 调用真正的 {label}")
    result = agent.invoke({"messages": [{"role": "user", "content": instruction}]})
    report = last_message_text(result).strip() or "(no report)"
    print(report)
    return report


def main() -> None:
    task = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    reset_workspace()

    print("=== 02. MultiAgent：每个 Agent 是一个 Node，用 conditional_edge 路由 ===")
    print("\n用户任务:")
    print(task)
    print("\n运行前 workspace:")
    print(show_workspace())

    llm = load_llm()
    file_agent = create_agent(
        llm,
        tools=[list_files, move_file],
        system_prompt=FILE_AGENT_PROMPT,
        name=FILE_AGENT_NAME,
    )
    code_agent = create_agent(
        llm,
        tools=[write_file],
        system_prompt=CODE_AGENT_PROMPT,
        name=CODE_AGENT_NAME,
    )

    def supervisor_node(state: MultiAgentState) -> MultiAgentState:
        print("\n[supervisor] 决定下一个 agent")

        allowed = allowed_next_agents(state)
        next_agent = allowed[0]
        decision = llm.invoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(
                    content=(
                        f"用户任务：{state['task']}\n\n"
                        f"file_report：{state['file_report'] or '(empty)'}\n\n"
                        f"code_report：{state['code_report'] or '(empty)'}\n\n"
                        f"当前允许的 agent：{', '.join(allowed)}\n\n"
                        f"请判断下一个 agent。建议答案：{next_agent}"
                    )
                ),
            ]
        ).content

        decision = normalize_decision(decision, allowed)
        print(decision)

        if decision == FINISH:
            return {**state, "next_agent": FINISH, "final_answer": summarize_reports(llm, state)}

        return {**state, "next_agent": decision}

    def file_agent_node(state: MultiAgentState) -> MultiAgentState:
        report = invoke_specialist(
            file_agent,
            label=FILE_AGENT_NAME,
            instruction=(
                f"用户原始任务：{state['task']}\n\n"
                "你是 file_agent，只处理其中和文件整理有关的部分。"
            ),
        )
        return {**state, "file_report": report}

    def code_agent_node(state: MultiAgentState) -> MultiAgentState:
        report = invoke_specialist(
            code_agent,
            label=CODE_AGENT_NAME,
            instruction=(
                f"用户原始任务：{state['task']}\n\n"
                f"file_agent 已完成的结果：\n{state['file_report']}\n\n"
                "你是 code_agent，只处理其中和代码生成有关的部分。"
            ),
        )
        return {**state, "code_report": report}

    def route_next(state: MultiAgentState) -> str:
        return END if state["next_agent"] == FINISH else state["next_agent"]

    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node(FILE_AGENT_NAME, file_agent_node)
    graph.add_node(CODE_AGENT_NAME, code_agent_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_next)
    graph.add_edge(FILE_AGENT_NAME, "supervisor")
    graph.add_edge(CODE_AGENT_NAME, "supervisor")

    result = graph.compile().invoke(
        {
            "task": task,
            "next_agent": "",
            "file_report": "",
            "code_report": "",
            "final_answer": "",
        }
    )

    print("\n[supervisor] 最终回答:")
    print(result["final_answer"])
    print("\n运行后 workspace:")
    print(show_workspace())


if __name__ == "__main__":
    main()