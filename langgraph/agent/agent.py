from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
import langchain_core.tools as tools
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import os
from dotenv import load_dotenv
from IPython.display import Image, display
import time
from datetime import datetime
import json

# 加载环境变量
load_dotenv()

class CustomCallbackHandler(BaseCallbackHandler):
    """增强版回调处理器，提供详细执行监控和错误处理"""
    
    def __init__(self):
        self.execution_stack = []
        self.timers = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._start_timer('llm')
        print(f"🕒 [{self._get_timestamp()}] LLM开始生成 | 提示数量: {len(prompts)}")

    def on_llm_end(self, response, **kwargs):
        duration = self._end_timer('llm')
        print(f"✅ [{self._get_timestamp()}] LLM生成完成 | 耗时: {duration:.2f}s | 响应长度: {len(str(response))}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._start_timer('tool')
        print(f"🔧 [{self._get_timestamp()}] 工具开始执行 | 工具: {serialized.get('name')} | 输入: {input_str[:200]}")

    def on_tool_end(self, output, **kwargs):
        duration = self._end_timer('tool')
        print(f"✅ [{self._get_timestamp()}] 工具执行完成 | 耗时: {duration:.2f}s | 输出: {str(output)[:200]}")

    def on_tool_error(self, error, **kwargs):
        print(f"❌ [{self._get_timestamp()}] 工具执行错误 | 错误类型: {type(error).__name__} | 详情: {str(error)[:500]}")

    def on_chain_start(self, serialized, inputs, **kwargs):
        self._start_timer('chain')
        chain_type = serialized.get('name', '未知链')
        print(f"⛓️ [{self._get_timestamp()}] 链开始执行 | 类型: {chain_type} | 输入参数: {self._format_inputs(inputs)}")

    def on_chain_end(self, outputs, **kwargs):
        duration = self._end_timer('chain')
        print(f"✅ [{self._get_timestamp()}] 链执行完成 | 耗时: {duration:.2f}s | 输出参数: {self._format_outputs(outputs)}")

    def on_agent_action(self, action, **kwargs):
        print(f"🤔 [{self._get_timestamp()}] Agent决策 | 选择工具: {action.tool} | 输入: {action.tool_input[:200]}")

    def _start_timer(self, event_type):
        self.timers[event_type] = time.time()

    def _end_timer(self, event_type):
        return time.time() - self.timers.pop(event_type, time.time())

    def _get_timestamp(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _format_inputs(self, inputs):
        return json.dumps(inputs, indent=2, ensure_ascii=False)[:300]

    def _format_outputs(self, outputs):
        return json.dumps(outputs, indent=2, ensure_ascii=False)[:300]

# 1. 定义状态模型
class AgentState(TypedDict):
    messages: List[Any]  # 对话历史
    task_status: str     # 任务状态
    task_plan: Dict     # 任务规划
    execution_context: Dict  # 执行上下文
    results: Dict       # 执行结果        

# 2. 工具定义
@tools.tool
def browse_web(url: str) -> Dict:
    """浏览网页并提取信息"""
    # 实际实现需要集成浏览器自动化工具如Selenium
    return {"title": "示例网页", "content": "网页内容摘要"}

@tools.tool
def edit_code(file_path: str, changes: Dict) -> Dict:
    """编辑代码文件"""
    # 实际实现需要集成代码编辑器API
    return {"status": "success", "file": file_path}

@tools.tool
def analyze_data(data: Dict, analysis_type: str) -> Dict:
    """分析数据"""
    # 实际实现需要集成数据分析库
    return {"analysis_result": "数据分析结果"}

# 3. 任务分解与规划
def decompose_task(task_description: str) -> List[Dict]:
    """将复杂任务分解为子任务"""
    # 使用LLM分析任务并生成子任务列表
    return [
        {"id": 1, "name": "信息收集", "tools": ["browse_web"]},
        {"id": 2, "name": "数据分析", "tools": ["analyze_data"]},
        {"id": 3, "name": "代码实现", "tools": ["edit_code"]}
    ]

def plan_execution(subtasks: List[Dict]) -> Dict:
    """规划任务执行顺序"""
    return {
        "execution_plan": subtasks,
        "dependencies": {"2": [1], "3": [2]}
    }

# 4. 节点函数
def understand_request(state: AgentState) -> AgentState:
    """理解用户请求并规划任务"""
    llm = ChatOpenAI(
        model=os.getenv("MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
        temperature=0.7,
        callbacks=[CustomCallbackHandler()]
    )
    messages = state["messages"]
    
    # 分析用户请求
    response = llm.invoke([
        HumanMessage(content=f"分析以下任务需求，提取关键信息和目标。用户消息: {messages[-1].content}")
    ])
    
    # 分解任务
    subtasks = decompose_task(messages[-1].content)
    execution_plan = plan_execution(subtasks)
    
    return {
        **state,
        "task_plan": execution_plan,
        "task_status": "planned",
        "execution_context": {"current_subtask": 0}
    }

def execute_task(state: AgentState) -> AgentState:
    """执行任务（简化版）"""
    context = state["execution_context"]
    plan = state["task_plan"]
    current_subtask = plan["execution_plan"][context["current_subtask"]]
    
    # 初始化工具实例
    tools_map = {
        "browse_web": browse_web,
        "analyze_data": analyze_data,
        "edit_code": edit_code
    }
    
    # 初始化执行结果和日志
    results = {}
    execution_log = context.get("logs", [])
    
    # 执行当前子任务
    for tool_name in current_subtask["tools"]:
        # 获取工具实例
        tool = tools_map.get(tool_name)
        if not tool:
            error_info = f"工具 {tool_name} 未找到"
            return {
                **state,
                "task_status": "error",
                "error_info": error_info
            }
        
        # 准备工具输入参数
        tool_input = {
            "browse_web": {"url": "https://example.com"},
            "analyze_data": {"data": results.get("web_data", {}), "analysis_type": "basic"},
            "edit_code": {"file_path": "example.py", "changes": {"line": 1, "content": "# New code"}}
        }.get(tool_name, {})
        
        try:
            # 记录工具调用开始
            execution_log.append({
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name,
                "status": "started",
                "input": tool_input
            })
            
            # 执行工具调用
            result = tool.invoke(tool_input)
            
            # 记录成功日志
            execution_log.append({
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name,
                "status": "success",
                "output": result
            })
            
            # 保存结果
            results[tool_name] = result
            
        except Exception as e:
            error_info = f"工具 {tool_name} 执行失败: {str(e)}"
            return {
                **state,
                "task_status": "error",
                "error_info": error_info
            }
    
    # 更新执行状态和日志
    context["current_subtask"] += 1
    context["logs"] = execution_log
    task_status = "completed" if context["current_subtask"] >= len(plan["execution_plan"]) else "executing"
    
    return {
        **state,
        "execution_context": context,
        "task_status": task_status,
        "results": {**state.get("results", {}), **results}
    }

def generate_report(state: AgentState) -> AgentState:
    """生成任务报告"""
    llm = ChatOpenAI(
        model=os.getenv("MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
        temperature=0.7,
        callbacks=[CustomCallbackHandler()]
    )
    
    results = state["results"]
    prompt = f"""
    基于以下执行结果生成详细报告：
    1. 网页数据: {results.get('web_data', {})}
    2. 分析结果: {results.get('analysis', {})}
    3. 代码变更: {results.get('code_changes', {})}
    
    请生成包含以下内容的报告：
    1. 任务概述
    2. 执行过程
    3. 关键发现
    4. 建议和下一步
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    messages = state["messages"] + [
        AIMessage(content=f"任务执行报告:\n\n{response.content}")
    ]
    
    return {
        **state,
        "messages": messages,
        "task_status": "completed"
    }

 
def handle_errors(state: AgentState) -> AgentState:
    """错误处理函数
    分析执行错误并尝试恢复
    """
    context = state["execution_context"]
    error_info = state.get("error_info", "未知错误")
    
    # 记录错误信息
    error_log = context.get("error_logs", [])
    error_log.append({
        "timestamp": datetime.now().isoformat(),
        "error": error_info
    })
    
    # 简单的恢复策略：重试当前失败的子任务
    if len(error_log) < 3:  # 最多重试3次
        return {
            **state,
            "task_status": "executing",
            "execution_context": {
                **context,
                "error_logs": error_log,
                "retry_count": len(error_log)
            }
        }
    
    # 超过重试次数，标记为无法恢复
    return {
        **state,
        "task_status": "failed",
        "execution_context": {
            **context,
            "error_logs": error_log
        }
    }
   

# 5. 条件分支处理
def should_end(state: AgentState) -> str:
    """决定工作流程序"""
    if state["task_status"] == "completed":
        return END
    if state["task_status"] == "planned":
        return "execute_task"
    if state["task_status"] == "executing":
        return "execute_task"
    return "understand_request"

# 6. 构建工作流图
# 定义状态转换条件
def can_execute_task(state: AgentState) -> bool:
    """判断是否可以执行任务"""
    return state["task_status"] in ["planned", "executing"]

def is_task_completed(state: AgentState) -> bool:
    """判断任务是否完成"""
    return state["task_status"] == "completed"

def is_error_recovered(state: AgentState) -> bool:
    """判断错误是否已恢复"""
    return state["task_status"] == "recovered"

# 创建工作流图实例
workflow = StateGraph(AgentState)

# 添加核心处理节点
workflow.add_node("understand_request", understand_request)  # 理解请求
workflow.add_node("execute_task", execute_task)          # 执行任务
workflow.add_node("generate_report", generate_report)    # 生成报告
workflow.add_node("error_handler", handle_errors)       # 错误处理

# 设置工作流起点
workflow.set_entry_point("understand_request")

# 配置状态转换规则
# 1. 请求理解节点的转换
workflow.add_conditional_edges(
    "understand_request",
    lambda state: "execute_task" if can_execute_task(state) else END,
    {
        "execute_task": "execute_task",  # 可以执行则转到执行节点
        END: "generate_report"          # 否则生成报告并结束
    }
)

# 2. 任务执行节点的转换
workflow.add_conditional_edges(
    "execute_task",
    lambda state: (
        "execute_task" if can_execute_task(state)
        else "error_handler" if state["task_status"] == "error"
        else "generate_report"
    ),
    {
        "execute_task": "execute_task",      # 继续执行
        "error_handler": "error_handler",    # 错误处理
        "generate_report": "generate_report"  # 生成报告
    }
)

# 3. 错误处理节点的转换
workflow.add_conditional_edges(
    "error_handler",
    lambda state: "generate_report" if is_error_recovered(state) else END,
    {
        "generate_report": "generate_report",  # 恢复后生成报告
        END: END                             # 无法恢复则结束
    }
)

# 4. 报告生成节点转换
workflow.add_edge("generate_report", END)  # 生成报告后结束

# 编译工作流
agent = workflow.compile()

def main():
    """主函数：初始化并执行工作流，展示执行结果"""
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='智能任务执行工作流')
    parser.add_argument('--task', type=str, help='要执行的任务描述', default='请帮我制定一个旅行计划')
    args = parser.parse_args()

    # 初始化状态
    initial_state = {
        "messages": [HumanMessage(content=args.task)],
        "task_status": "new",
        "task_plan": {},
        "execution_context": {},
        "results": {}
    }

    # 执行工作流
    print("\n🚀 开始执行工作流...\n")
    result = agent.invoke(initial_state)

    # 展示执行结果
    print("\n📊 执行结果汇总：")
    print("-" * 50)
    print(f"✅ 任务状态：{result['task_status']}")
    
    if result.get('task_plan'):
        print("\n📋 任务规划：")
        for task in result['task_plan'].get('execution_plan', []):
            print(f"  - {task['name']} (ID: {task['id']})")
    
    if result.get('results'):
        print("\n🎯 执行结果：")
        for key, value in result['results'].items():
            print(f"  - {key}: {value}")
    
    if result.get('messages'):
        print("\n💬 最终报告：")
        print(result['messages'][-1].content)

if __name__ == "__main__":
    main()

