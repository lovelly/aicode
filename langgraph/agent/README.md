# 使用 LangGraph 开发智能体的入门教程

## 简介

`LangGraph` 是一个用于构建智能体的框架，旨在简化复杂任务的管理和执行。通过定义状态图和任务分解，开发者可以轻松创建能够理解和执行用户请求的智能体。

## 核心概念

### 1. 状态图 (StateGraph)

状态图是 `LangGraph` 的核心组件，它定义了智能体的状态和状态之间的转换。每个状态代表智能体在执行过程中的一个阶段，状态之间的转换则基于条件和事件。

在 `agent.py` 中，状态图通过 `create_agent` 函数创建，并添加多个节点（如 `understand_request`、`execute_task`、`handle_errors` 和 `generate_report`），这些节点代表智能体的不同功能。状态图的设计使得智能体能够根据任务的进展动态地在不同状态之间切换。

### 2. 状态传递

智能体的状态通过 `AgentState` 类型的字典进行传递。这个字典包含了以下关键信息：

- `messages`: 存储用户的对话历史。
- `task_status`: 当前任务的状态（如新建、执行中、完成、失败等）。
- `task_plan`: 任务的规划，包括子任务的列表。
- `execution_context`: 当前执行上下文，包含执行过程中需要的动态信息。
- `results`: 存储各个子任务的执行结果。
- `memory`: 记忆管理器的实例，用于存储短期和长期记忆。
- `context`: 上下文管理器的实例，用于管理任务执行的上下文信息。

### 3. ContextManager

`ContextManager` 是用于管理任务执行上下文的类。它负责存储和更新与当前任务相关的动态信息，如用户的偏好、目的地、预算等。通过 `update_context` 和 `get_context` 方法，智能体可以在执行过程中随时获取和更新上下文信息。

```python
class ContextManager:
    """上下文管理器：管理任务执行上下文"""
    def __init__(self):
        self.context = {}
        self.context_history = []

    def update_context(self, key: str, value: Any) -> None:
        """更新上下文"""
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """获取上下文值"""
        return self.context.get(key, default)

    def save_context_snapshot(self) -> None:
        """保存上下文快照"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "context": self.context.copy()
        }
        self.context_history.append(snapshot)
        
        # 限制历史记录数量
        if len(self.context_history) > 20:
            self.context_history.pop(0)
```

### 4. MemoryManager

`MemoryManager` 是用于管理智能体记忆的类。它分为短期记忆和长期记忆，短期记忆用于存储当前会话中的信息，而长期记忆则用于存储重要的历史信息。通过 `add_to_short_term` 和 `promote_to_long_term` 方法，智能体可以动态地管理记忆。

```python
class MemoryManager:
    """记忆管理器：管理短期和长期记忆"""
    def __init__(self):
        self.short_term_memory = []
        self.long_term_memory = []
        self.importance_threshold = 0.7

    def add_to_short_term(self, memory_item: Dict) -> None:
        """添加到短期记忆"""
        if not isinstance(memory_item, dict):
            memory_item = {"content": str(memory_item)}
        
        if "timestamp" not in memory_item:
            memory_item["timestamp"] = datetime.now().isoformat()
        
        self.short_term_memory.append(memory_item)
        
        # 如果短期记忆超过容量，移除最旧的记忆
        if len(self.short_term_memory) > 50:
            self.short_term_memory.pop(0)

    def promote_to_long_term(self, index: int, importance: float) -> bool:
        """将短期记忆提升为长期记忆"""
        if importance < self.importance_threshold:
            return False
        
        if 0 <= index < len(self.short_term_memory):
            memory_item = self.short_term_memory[index].copy()
            memory_item["importance"] = importance
            self.long_term_memory.append(memory_item)
            return True
        
        return False

    def retrieve_relevant(self, query: str, limit: int = 5) -> List[Dict]:
        """检索相关记忆"""
        # 简单实现：基于关键词匹配
        relevant_memories = []
        
        # 搜索长期记忆
        for memory in self.long_term_memory:
            content = memory.get("content", "")
            if isinstance(content, str) and any(keyword in content for keyword in query.split()):
                relevant_memories.append(memory)
        
        # 搜索短期记忆
        for memory in reversed(self.short_term_memory):
            content = memory.get("content", "")
            if isinstance(content, str) and any(keyword in content for keyword in query.split()):
                relevant_memories.append(memory)
                
        # 去重并限制数量
        unique_memories = []
        memory_contents = set()
        
        for memory in relevant_memories:
            content = str(memory.get("content", ""))
            if content not in memory_contents:
                memory_contents.add(content)
                unique_memories.append(memory)
                if len(unique_memories) >= limit:
                    break
        
        return unique_memories
```

## 工作流程

### 1. 用户请求的理解

智能体首先通过 `understand_request` 函数理解用户的请求。该函数会分析用户的消息，并提取关键信息，如目的地、预算、旅行时间等。智能体会使用 `MemoryManager` 检索相关的记忆，以便更好地理解用户的需求。

```python
def understand_request(state: AgentState) -> AgentState:
    """理解用户请求并规划任务"""
    llm = get_llm()
    messages = state["messages"]
    # 初始化记忆管理器和上下文管理器
    memory_manager = state.get("memory", {}).get("manager", MemoryManager())
    context_manager = state.get("context", {}).get("manager", ContextManager())
    # 获取相关记忆
    relevant_memories = memory_manager.retrieve_relevant(messages[-1].content)
    # 构建提示，包含相关记忆
    memory_context = "\n".join([f"- {m.get('content', '')}" for m in relevant_memories])
    # 分析用户请求并推断信息
    # ...
```

### 2. 任务的执行

在理解用户请求后，智能体会调用 `execute_task` 函数来执行任务。该函数会根据任务计划逐步执行子任务，并调用相应的工具。执行过程中，智能体会更新上下文和记忆，以便在后续的任务中使用。

```python
def execute_task(state: AgentState) -> AgentState:
    """执行任务（增强版）"""
    context = state["execution_context"]
    plan = state["task_plan"]
    current_subtask = plan["execution_plan"][context["current_subtask"]]
    # 获取记忆和上下文管理器
    memory_manager = state.get("memory", {}).get("manager", MemoryManager())
    context_manager = state.get("context", {}).get("manager", ContextManager())
    # 执行当前子任务
    for tool_name in current_subtask["tools"]:
        # 获取工具实例并执行
        # ...
```

### 3. 错误处理与报告生成

在执行过程中，智能体可能会遇到错误。可以使用 `handle_errors` 函数处理错误，并使用 `generate_report` 函数生成执行报告，记录任务的执行历史和结果。

```python
def handle_errors(state: AgentState) -> AgentState:
    """处理执行过程中的错误"""
    # 实现代码...
```

```python
def generate_report(state: AgentState) -> AgentState:
    """生成增强版任务报告，包含执行历史和记忆信息"""
    # 实现代码...
```

## 运行智能体

在主函数中，解析命令行参数并初始化状态。然后调用智能体的 `invoke` 方法执行任务。

```python
def main():
    """主函数：初始化并执行工作流，展示执行结果"""
    # 实现代码...
```

## 总结

通过以上步骤，您可以使用 `LangGraph` 框架快速构建智能体。该框架提供了灵活的状态管理和任务分解能力，使得开发复杂的智能体变得更加简单和高效。希望本教程能帮助您快速上手 `LangGraph` 的使用。

请根据需要进一步调整内容或格式，以确保其符合您的要求。
