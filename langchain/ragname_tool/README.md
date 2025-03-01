# LangChain Agent与工具调用深度教程 - 构建智能助手完全指南

## 基础概念

1. 本教程旨在帮助您深入了解LangChain Agent与工具调用的概念和实践。通过本教程，您将学习如何使用LangChain Agent与工具调用来构建智能助手，开发一些有趣的智能应用。
2. LangChain中Tool和Agent是两个核心概念，但它们的关系需要进一步明确：
Tool(工具): 是执行特定功能的函数封装，比如搜索、计算或API调用。工具本身是"无状态"的，只关注输入和输出。
Agent(代理): 是决策单元，负责协调多个工具的使用顺序和时机。Agent包含状态管理、决策逻辑和执行控制。
简言之，Tool是"做什么"，Agent是"怎么做"和"何时做"。
3. ReAct agent框架(Reasoning and Acting)框架是基于"思考-行动-观察"的循环：
Reasoning(推理): LLM根据当前情况分析下一步行动
Acting(行动): 执行选定的工具
Observation(观察): 获取工具执行结果并更新状态
ReAct让LLM将"思考过程"显式表达出来，通过"Thought:"前缀引导模型进行逐步推理，这大大提高了复杂任务的成功率
4. agent 工作流程
```shell
用户输入
   │
   ▼
┌───────────────┐
│ LangChain框架 │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Agent决策    │
└───────┬───────┘
        │ (解析意图)
        ▼
┌───────────────┐
│   LLM推理     │
└───────┬───────┘
        │ (工具选择)
        ▼
┌───────────────┐
│  工具路由     │
└───────┬───────┘
   ├─────┼─────┐
   ▼     ▼     ▼
┌───┐ ┌───┐ ┌───┐
│工具1│ │工具2│ ...(工具N)
└─┬─┘ └─┬─┘ └─┬─┘
  │     │     │
  └──┬──┴──┬──┘
      ▼
  ┌───────────────┐
  │  结果整合     │
  └───────┬───────┘
          ├─────────────┐
          ▼             │
  ┌───────────────┐     │
  │  最终响应     │◄────┘
  └───────┬───────┘
          ▼
      用户输出
```


## 1. 环境准备

```shell
pip install langchain langchain-openai langchain-deepseek langchain-chroma langchain-huggingface pydantic python-dotenv
```

## 2. 工具开发定义

在LangChain中，工具是Agent可以调用的功能。使用@tool装饰器可以轻松创建工具：

```python
from langchain.tools import tool
from pydantic import BaseModel, Field

# 定义工具的输入模型
class SearchInput(BaseModel):
    query: str = Field(description="搜索查询内容")
    num_results: Optional[int] = Field(default=3, description="返回的搜索结果数量")

@tool("web_search", args_schema=SearchInput)
def web_search(query: str, num_results: int = 10) -> str:
    """使用DuckDuckGo搜索互联网获取信息"""
    print(f"执行搜索: {query}，数量: {num_results}")
    search = DuckDuckGoSearchRun()
    results = search.run(f"{query}")
    # ... 处理和保存结果 ...
    return f"搜索完成，结果已保存。搜索结果摘要:\n{results[:1024]}...(已截断)"
```

  1.定义工具输入模型 SearchInput
    * 使用 `BaseModel`
    * 使用 `Field` 定义字段，包括描述和默认值, 
   
  2. 定义工具
    * `@tool`装饰器将函数转换为LangChain工具，指定工具名称和参数设置信息
    * 在函数第一行文档字符串中提供工具的描述，Agent会使用这些信息提供给大模型
    * 工具的描述是关键，LLM会使用它来理解工具功能

## 3. 创建agent
  ReAct (Reasoning and Acting) 是一种流行的Agent框架，允许LLM交替进行推理和行动：
```python
def create_name_generator_agent():
    # 初始化LLM
    callback_manager = create_callback_manager()
    llm = ChatOpenAI(
        model=os.getenv("MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
        temperature=0.7,
        callbacks=callback_manager.handlers,
    )
    
    # 创建工具列表
    tools = [web_search, store_to_rag, load_web_content, generate_names]
    
    # 创建ReAct提示模板, 用于提示LLM执行任务
    react_template = """回答以下问题：{input}

    你有这些工具可用:
    {tools}

    工具名称列表: {tool_names}

    使用以下格式：
    Thought: 你应该思考如何解决这个问题
    Action: 工具名称，例如 "web_search"
    Action Input: 工具的输入参数
    Observation: 工具的输出结果
    Final Answer: 最终答案

    {agent_scratchpad}"""

    react_prompt = PromptTemplate(
        input_variables=['input', 'tools', 'tool_names', 'agent_scratchpad'],
        template=react_template
    )

    # 创建ReAct agent
    react_agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(agent=react_agent, tools=tools,
     verbose=True, handle_parsing_errors=True, max_iterations=3)

    # handle_parsing_errors=True：自动处理LLM输出解析错误，非常好用
    # max_iterations=3：限制最大迭代次数，防止无限循环
    # tool_input_fixers：自定义工具输入修复函数，例如自动添加缺失字段等

    # 其他参数
    # agent_executor = AgentExecutor(
    #   agent=react_agent,  # Agent实例
    #   tools=tools,  # 可用工具列表
    #   verbose=True,  # 是否显示详细执行过程
    #   handle_parsing_errors=True,  # 自动处理输出解析错误
    #   max_iterations=3,  # 最大执行循环次数
    #   early_stopping_method="generate",  # 提前停止方法：force/generate
    #   tool_input_fixers=[fix_function],  # 参数修正函数
    #   return_intermediate_steps=True,  # 是否返回中间步骤
    #   max_execution_time=None,  # 最大执行时间(秒)
    #   timeout=120,  # 单步超时时间
    # )
    
    return agent_executor
```
1. 我们需要在prompt中告诉大模型，我们有哪些工具可以使用，以及如何使用这些工具。
  {input}, {tools},{tool_names}, {agent_scratchpad} langchain会替换这些输入和工具描述和agent_scratchpad 工具执行过程记录。
2. 使用 create_react_agent 创建 react agent。 传入llm, tools, react_prompt。langchain封装了多种类型的agent。
```py
AGENT_TYPES = [
    # 零样本 ReAct agent，适合简单任务，不需要示例就能执行
    AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # 结构化对话的零样本 ReAct agent，提供更好的对话控制
    AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    # 对话式 ReAct agent，适合需要上下文记忆的场景
    AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    # 基于聊天的零样本 ReAct agent，针对聊天场景优化
    AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    # 基于聊天的对话式 ReAct agent，结合了聊天和对话能力
    AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    # OpenAI 函数调用 agent，专门与 OpenAI 模型配合使用
    AgentType.OPENAI_FUNCTIONS,
    # OpenAI 多函数调用 agent，支持同时调用多个函数
    AgentType.OPENAI_MULTI_FUNCTIONS,
    # 自问自答搜索 agent，适合需要分步推理的搜索任务
    AgentType.SELF_ASK_WITH_SEARCH,
    # XML 处理 agent，专门处理 XML 格式数据
    AgentType.XML,
    # JSON 处理 agent，专门处理 JSON 格式数据
    AgentType.JSON,
    # 结构化对话 agent，提供严格的对话结构控制
    AgentType.STRUCTURED_CHAT,
    # 计划执行 agent，先规划后执行，适合复杂任务
    AgentType.PLAN_AND_EXECUTE
]

agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    ) 
# 可以用 initialize_agent 创建agent， 也可以用我代码例子中的creat_xxxx_agent 创建agent。
```
#其他常用agent
- create_openai_functions_agent
  - 基于 OpenAI 函数调用功能的 agent
  - 更适合与 OpenAI 模型配合使用
  - 支持结构化输出，结构化输出更准确。

- create_structured_chat_agent
  - 结构化对话 agent
  - 提供更好的对话流程控制
  - 适合需要严格输出格式的场景，比如订单处理，问卷调查

- create_json_chat_agent
  - 专门处理 JSON 格式的 agent
  - 输入输出都是 JSON 格式
  - 适合需要处理结构化数据的场景, 比如json API 调用。

- create_xml_agent
  - 专门处理 XML 格式的 agent
  - 适合需要处理 XML（网页抓取和解析） 数据的场景

- create_conversational_agent
  - 对话式 agent
  - 更适合自然语言交互
  - 支持上下文记忆，适合智能客服，教育辅导，个人助理的agent

3. 使用 AgentExecutor 执行agent。 处理工具调用和错误处理。
4. handle_parsing_errors = True：自动处理LLM输出解析错误，非常好用
   - LangChain中一个常见挑战是LLM输出格式不符合预期。您可以详细解释解析机制：
   - 输出解析器(OutputParser): 负责将LLM文本输出转换为结构化数据
   - 解析错误处理: handle_parsing_errors=True启用自动修复修复流程:
     - 检测到格式错误
     - 向LLM发送错误信息和修复提示
     - 接收修复后的输出并重新解析

## 4. 执行修正大模型的参数
1. llm通过prompt知道有哪些工具和需要什么参数，会自动生成参数，调用tool, AgentExecutor 提供了一个hook, 可以在agent执行过程中，修正大模型的参数。
```python
def fix_tool_inputs(action, tool_input):
    # 特别处理store_to_rag工具
    if action == "store_to_rag":
        # 检查是否缺少source字段
        if isinstance(tool_input, dict) and "content" in tool_input and "source" not in tool_input:
            # 自动添加source字段
            tool_input["source"] = "从web_search获取的内容"
    return tool_input
```

## 5. 回调监控
```python
from langchain.callbacks import StdOutCallbackHandler, FileCallbackHandler

handlers = [
    StdOutCallbackHandler(),  # 控制台输出
    FileCallbackHandler("agent_log.txt"),  # 文件日志
]

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    callbacks=handlers,
)
```
回调可捕获的事件包括：
  - on_llm_start: LLM开始生成
  - on_llm_end: LLM完成生成
  - on_tool_start: 工具开始执行
  - on_tool_end: 工具执行完成
  - on_chain_start: 链开始执行
  - on_chain_end: 链执行完成
  - on_agent_action: Agent选择行动
  - on_agent_finish: 
  - 可以自己实现这些方法来监控回调
```python
from langchain.callbacks.base import BaseCallbackHandler
class CustomCallbackHandler(BaseCallbackHandler):
  # TODO
```

## 6. 运行测试， 写好的工具，我们可以测试一下。
```python
def run_example_manually():
    """直接使用工具函数而不是通过agent调用"""
    print("=== 直接执行工具函数  ===")
   
    # 1. 执行搜索
    search_results = web_search.invoke({"query": "北欧神话中的女孩名字", "num_results": 3})
    
    # 2. 将搜索结果存储到RAG
    store_result = store_to_rag.invoke({"content": search_results})
    
    # 3. 使用RAG生成名字
    names_with_rag = generate_names.invoke({"gender": "girl", "use_rag": True})
```
1. 工具实现了Runnable 接口，我们可以直接调用invoke方法执行。模型大模型调用tool

## 7. 主流程函数
```python
if __name__ == "__main__":
    # 通过Agent进行交互
    agent = create_name_generator_agent()
    
    # 示例1: 请求搜索北欧名字信息
    result1 = agent.invoke({
        "input": "我想了解一些北欧神话中的女孩名字，能帮我搜索相关信息,存储到rag向量库，然后给出回答吗？"
    })
    print("结果:", result1["output"])
```
1. 结合之前我们学习的RAG知识库，我们可以做一个多链调用的智能agent, 从web搜索网页，构建知识库，生成名字。随着实践的深入，你可以扩展Agent的能力，添加更多工具，优化提示模板，甚至将多个Agent连接起来构建更复杂的系统。

希望这个教程能帮助你开始LangChain Agent的开发之旅！


git源码：https://github.com/lovelly/aicode.git
