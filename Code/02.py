from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

DEFAULT_USER_PROMPT = "帮我查一下 Beijing 的天气"

SYSTEM_PROMPT = """
你是一个助手。
系统里有一个工具叫 get_weather，用于查询天气。

当用户询问天气时，不要直接回答。
你必须严格输出下面的格式，不要输出额外内容：

<Tool>get_weather</Tool>
<Args>{"city":"Beijing"}</Args>

如果用户不需要查询天气，就直接输出普通文本。
""".strip()


def get_weather(city: str) -> str:
    weather = {
        "Beijing": "晴天，25 度",
        "Shanghai": "多云，28 度",
        "Hangzhou": "小雨，22 度",
    }
    return f"{city} 的天气是：{weather.get(city, '未知天气')}"


def load_llm() -> ChatOpenAI:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return ChatOpenAI(
        model=os.getenv("MODEL", "qwen3.5:cloud"),
        base_url=os.getenv("BASE_URL", "http://localhost:11434/v1/"),
        api_key=os.getenv("API_KEY", "ollama"),
        temperature=0,
    )


def parse_tool_call(text: str) -> dict[str, object] | None:
    tool_match = re.search(r"<Tool>(.*?)</Tool>", text, re.DOTALL)
    args_match = re.search(r"<Args>(.*?)</Args>", text, re.DOTALL)

    if not tool_match:
        return None

    args = json.loads(args_match.group(1)) if args_match else {}
    return {
        "tool": tool_match.group(1).strip(),
        "args": args,
    }


def main() -> None:
    user_prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_USER_PROMPT

    print("=== 02. Prompt 协议版 ToolCall ===")
    print("\n用户请求:")
    print(user_prompt)

    response = load_llm().invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    model_text = str(response.content)
    print("\n模型原始输出:")
    print(model_text)

    call = parse_tool_call(model_text)
    if call is None:
        print("\n没有工具调用，模型返回的是普通文本。")
        return

    print("\n解析后的工具请求:")
    print(f"tool_name = {call['tool']}")
    print(f"tool_args = {call['args']}")

    if call["tool"] == "get_weather":
        result = get_weather(**call["args"])
    else:
        result = f"未知工具：{call['tool']}"

    print("\n工具执行结果:")
    print(result)


if __name__ == "__main__":
    main()