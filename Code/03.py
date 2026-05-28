from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

DEFAULT_USER_PROMPT = "帮我查一下 Beijing 的天气和时间"

SYSTEM_PROMPT = """
你是一个助手。
如果用户询问天气，请调用 get_weather 工具，不要自己编造天气。
如果用户询问时间，请调用 get_time 工具，不要自己编造时间。
""".strip()


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    weather = {
        "Beijing": "晴天，25 度",
        "Shanghai": "多云，28 度",
        "Hangzhou": "小雨，22 度",
    }
    return f"{city} 的天气是：{weather.get(city, '未知天气')}"

@tool
def get_time(city: str) -> str: 
    """Get the current time for a city."""
    time = {
        "Beijing": "14:00",
        "Shanghai": "14:00",
        "Hangzhou": "14:00",
    }
    return f"{city} 的当前时间是：{time.get(city, '未知时间')}"

def load_llm() -> ChatOpenAI:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return ChatOpenAI(
        model=os.getenv("MODEL", "qwen3.5:cloud"),
        base_url=os.getenv("BASE_URL", "http://localhost:11434/v1/"),
        api_key=os.getenv("API_KEY", "ollama"),
        temperature=0,
    )


def main() -> None:
    user_prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_USER_PROMPT

    print("=== 03. LangChain 原生 ToolCall ===")
    print("\n用户请求:")
    print(user_prompt)

    tools = [get_weather, get_time]
    llm = load_llm().bind_tools(tools)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)

    print("\n模型文本输出:")
    print(response.content)

    print("\nLangChain 解析出的 tool_calls:")
    print(response.tool_calls)

    for tool_call in response.tool_calls:
        print("\n准备执行工具:")
        print(f"tool_name = {tool_call['name']}")
        print(f"tool_args = {tool_call['args']}")

        if tool_call["name"] == "get_weather":
            result = get_weather.invoke(tool_call["args"])
        elif tool_call["name"] == "get_time":
            result = get_time.invoke(tool_call["args"])
        else:
            result = f"未知工具：{tool_call['name']}"

        print("\n工具执行结果:")
        print(result)

        messages.append(response)
        messages.append(
            ToolMessage(
                content=result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

    if response.tool_calls:
        final_response = llm.invoke(messages)
        print("\n把工具结果交回模型后的最终回答:")
        print(final_response.content)


if __name__ == "__main__":
    main()