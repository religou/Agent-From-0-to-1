## 1. LangChain & LangGraph 简介

### 1.1. LangChain

LangChain：一个做 LLM 应用的组件工具箱，包括：模型、提示词、工具、消息等工具

Model：大模型本体，例如：GPT、Gemini、Claude

Prompt：给模型的指令模板

Message：LangChain 会统一管理 system、user、assistant 等消息格式

Tool：本质上是可调用函数

Tool Calling：本质上是调用工具

Agent：基于上下文决定要不要调用工具、怎么继续推进任务的执行体

![1779794449687](images/01_搭建简单的Agent/1779794449687.png)

一句话总结：

**Agent** 利用 **LangChain** 工具箱，通过 **Prompt** 和 **Message** 驱动 **Model** ，并在需要时通过 **Tool Calling** 调用外部 **Tool** **，从而完成复杂任务。**

### 1.2. LangGraph

LangGraph：专门做 Agent 流程编排的框架，适合长任务、循环、多步骤执行

State：整个图共享的状态数据，节点可以读写它，可以理解成 Agent的工作记忆和任务面板

Node：图里面的步骤，一个节点通常只干一个事情，例如：调用天气工具、生成报告

Edge：节点之间的连接关系，决定下一步要干啥。例如：if X>5 跳转到节点 A，支持错误处理和超时机制

Conditional Edge：带条件的边

Graph：多个节点&边组成的完整执行流程

Compile：在真正执行前，把图做一次检查和封装，确认结构可执行，会检查节点连接是否合法（是否会陷入死循环）

![1779794501480](images/01_搭建简单的Agent/1779794501480.png)

### 3. 差异比较

| **场景**          | **LangChain**       | **LangGraph**                     |
| ----------------- | ------------------- | --------------------------------- |
| **简单任务**      | **✅ 适合**         | **✅ 也可用**                     |
| **长任务/多步骤** | **❌ 难以维护状态** | **✅**`State`全局共享             |
| **条件分支/循环** | **❌ 逻辑易混乱**   | **✅ 可视化条件边 + 循环机制**    |
| **错误处理**      | **❌ 需手动编码**   | **✅ 内置** `if error occurs`路径 |

## 2. ToolCall

Agent 本质都是对模型输入输出的文本进行操作

Function Calling 的本质：大模型本身绝对不执行任何代码！ 它只是通过理解用户意图，生成了一段结构化的“调用请求”（通常是 JSON），然后由外部系统（如你的 Python 代码或 LangChain 框架）去真正执行，最后把结果喂回给大模型。

### 2.1. Function Calling 的运转流程

**假设用户问：“** _北京今天天气怎么样？_ **”**

#### 2.1.1. 定义工具 (Define Tools)

**在调用大模型前，开发者需要把可用的工具描述清楚（函数名、功能描述、参数结构）。**

```json
{
    "name": "get_weather",
    "description": "获取指定城市的当前天气",
    "parameters": {
        "type": "object",
        "properties": {
            "city": { "type": "string", "description": "城市名称，如北京" }
        },
        "required": ["city"]
    }
}
```

#### 2.1.2.发送请求(Send Request)

用户的提问和工具的描述一起发送给大模型。

**User Message\*\***: "北京今天天气怎么样？"\*\*

**System/Tools** **: [上述 get_weather 的 JSON 描述]**

#### 2.1.3.模型决策与生成意图（Model Reasoning）

大模型分析后发现自己不知道实时天气，但看到工具列表里有 `get_weather`。于是，它 **不直接回答用户** **，而是输出一个** **结构化的函数调用请求** **（拦截输出）。**

```
{
  "tool_calls": [
    {
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"北京\"}"
      }
    }
  ]
}
```

注：此时大模型的工作已经结束，只负责 “意图识别”和“参数提取”

#### 2.1.4.外部系统执行（Framework Execution）

你的应用代码（或 LangChain/LangGraph 框架）拦截到这段 JSON，解析出函数名 get_weather 和参数 北京，然后在本地或服务器上真正执行这个 Python 函数。

```
# 框架在后台默默执行
result = get_weather(city="北京")
# 假设返回结果: "晴，25℃，微风"
```

#### 2.1.5. 结果回传（Return Result）

**框架将函数执行的真实结果，包装成一条 **`ToolMessage`（工具消息），再次发送给大模型。

- Tool Message : "晴，25℃，微风"

#### 2.1.6. 模型总结输出（Final Generation）

大模型接收到天气结果后，结合用户的原始问题，生成最终的自然语言回复。

- Assistant Message : "北京今天的天气是晴天，气温 25℃，伴有微风，非常适合出行。"

#### 2.1.7. 流程总结

![1779795958370](images/01_搭建简单的Agent/1779795958370.png)

#### 2.1.8. 知识补充

为什么大模型能精准输出 JSON 而不是胡乱输出其他的格式？

- 模型是被训练出来的，在 SFT 阶段，模型接触了数百万个“问题 → 工具调用”的配对样本
- 系统提示，调用模型时，框架会对提出的问题重新设计，并给模型一个精心设计后的系统提示，例如：

    ```
    You are an AI assistant that can call functions. When you need to use a function,
    output ONLY in the following JSON format:
    {
      "name": "function_name",
      "arguments": {"param1": "value1", ...}
    }
    DO NOT output anything else. Only output the JSON.
    ```

- 参数提取的 “模式识别”能力，模型可以被训练成从用户问题中自动提取关键参数
- 严格的格式验证机制，首先框架会检查模型输出是否符合预期格式，如果输出的不是有效的 json，框架会重新提示模型，对于一些高级模型在生成时，就知道要返回 json 格式的数据
