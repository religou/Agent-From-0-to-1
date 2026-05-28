def get_weather(city: str) -> str:
    weather_data = {
        "Beijing": "多云，32度",
        "Chicago": "晴，25度",
        "New York": "雨，28度",
    }
    return f"{city}的天气是：{weather_data.get(city, '未知')}"

def parse_model_output(text: str) -> tuple[str, dict[str, str]]:
    if ":" in text:
        tool, tool_args = text.split(":", 1)
        return tool, tool_args
    return None, {"error": "Invalid format"}

def main() -> None:
    print(f"字符串协议版：")
    model_output = "get_weather: Beijing"
    print(f"假设模型输出: {model_output}")
    tool, tool_args = parse_model_output(model_output)
    print(f"解析结果: {tool}, 参数: {tool_args}")
    if tool == "get_weather":
        result = get_weather(tool_args.strip())
        print(f"工具调用结果: {result}")
    else:
        print("未知工具")

if __name__ == "__main__":
    main()