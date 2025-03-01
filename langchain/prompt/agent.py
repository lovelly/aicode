from typing import List, Optional
from langchain.agents import AgentType, Tool
from langchain.agents import create_react_agent, create_tool_calling_agent, create_structured_chat_agent, create_openai_functions_agent
from langchain_community.agent_toolkits.json.base import create_json_agent
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit
from langchain_community.tools.json.tool import JsonSpec
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.callbacks import StdOutCallbackHandler, FileCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.prompts import (
    PromptTemplate,
    PipelinePromptTemplate,
    FewShotPromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.prompts.image import ImagePromptTemplate
from langchain.agents import AgentExecutor
from langchain_community.vectorstores import FAISS
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class CustomCallbackHandler(BaseCallbackHandler):
    """自定义回调处理器，用于监控Agent执行过程"""
    
    def on_llm_start(self, *args, **kwargs):
        """当LLM开始生成时触发"""
        print("🤖 LLM开始生成...")
    
    def on_llm_end(self, *args, **kwargs):
        """当LLM完成生成时触发"""
        print("✅ LLM生成完成")
    
    def on_tool_start(self, *args, **kwargs):
        """当工具开始执行时触发"""
        print("🔧 工具开始执行...")
    
    def on_tool_end(self, *args, **kwargs):
        """当工具执行完成时触发"""
        print("✅ 工具执行完成")
    
    def on_chain_start(self, *args, **kwargs):
        """当链开始执行时触发"""
        print("⛓️ 链开始执行...")
    
    def on_chain_end(self, *args, **kwargs):
        """当链执行完成时触发"""
        print("✅ 链执行完成")
    
    def on_agent_action(self, *args, **kwargs):
        """当Agent选择行动时触发"""
        print("🤔 Agent正在思考下一步行动...")

# 1. ReAct Agent与基础PromptTemplate和PipelinePromptTemplate的结合
def create_enhanced_react_agent():
    """创建增强版ReAct Agent，专注于多步推理和复杂任务分解
    
    ReAct（Reasoning + Acting）是一种结合推理和行动的Agent范式：
    1. Reasoning：通过思考分析问题，制定解决方案
    2. Acting：选择并使用合适的工具执行行动
    3. Observing：观察行动结果，进行下一步决策
    
    工作原理：
    1. Agent接收任务输入
    2. 通过LLM进行推理，生成思考过程
    3. 选择合适的工具执行行动
    4. 观察结果，继续推理或得出最终答案
    
    优势：
    - 可解释性强：每步推理过程清晰可见
    - 自我纠错：能够根据观察结果调整策略
    - 复杂任务处理：适合需要多步推理的任务
    """
    
    # 创建回调处理器，用于监控和记录执行过程
    handlers = [StdOutCallbackHandler(), FileCallbackHandler("agent_log.txt"), CustomCallbackHandler()]
    # 初始化ChatOpenAI模型，temperature控制输出的随机性
    llm = ChatOpenAI(
        model=os.getenv("MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
        temperature=0.7, callbacks=handlers)
    
    # 创建专业工具集，每个工具都有特定的功能和描述
    def search_web(query: str) -> str:
        """网络搜索工具，用于获取信息"""
        return f"搜索结果: 关于{query}的深度分析报告"
    
    def analyze_data(data: str) -> str:
        """数据分析工具，用于处理和分析信息"""
        return f"多维度分析: {data}的系统性分析，包含关键指标、趋势和建议"
    
    def decompose_task(task: str) -> str:
        """任务分解工具，用于将复杂任务分解为可管理的子任务"""
        return f"任务分解: 将{task}分解为可执行的子任务列表"
    
    # 定义工具列表，每个工具都包含名称、函数和描述
    tools = [
        Tool(name="web_search", func=search_web, description="高级网络搜索工具，提供深度分析报告"),
        Tool(name="data_analyzer", func=analyze_data, description="多维数据分析工具，提供系统性分析"),
        Tool(name="task_decomposer", func=decompose_task, description="任务分解工具，将复杂任务分解为子任务")
    ]
    
    # 创建增强版ReAct模板，提供详细的思考框架
    react_template = """执行以下复杂任务：{input}

你有这些专业工具可用:
{tools}

工具名称列表: {tool_names}

使用以下系统化思考格式：
1. Thought: 系统分析问题
   - 任务的关键目标是什么？
   - 需要收集哪些信息？
   - 如何分解为子任务？
   - 各个步骤如何衔接？

2. Action: 选择最合适的工具
   - 为什么选择这个工具？
   - 预期获得什么结果？

3. Action Input: 准备工具输入
   - 输入是否完整准确？
   - 是否需要预处理？

4. Observation: 分析工具输出
   - 结果是否符合预期？
   - 是否需要补充信息？
   - 如何用于下一步？

5. 重复上述步骤直到问题解决

6. Final Answer: 给出系统性解决方案
   - 总结关键发现
   - 提供可执行建议
   - 说明潜在影响

开始系统思考：
{agent_scratchpad}"""
    
    # 创建PromptTemplate实例，定义输入变量和模板
    react_prompt = PromptTemplate(
        input_variables=['input', 'tools', 'tool_names', 'agent_scratchpad'],
        template=react_template
    )
    
    # 打印格式化后的prompt内容
    print("\n=== ReAct Agent Prompt ===\n")
    print(react_prompt.format(
        input="示例任务",
        tools=tools,
        tool_names=[tool.name for tool in tools],
        agent_scratchpad=""
    ))
    
    # 创建ReAct Agent，将LLM、工具和提示模板组合在一起
    react_agent = create_react_agent(llm, tools, react_prompt)
    
    # 创建并返回AgentExecutor，负责实际执行Agent的操作
    return AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=True,  # 显示详细执行过程
        handle_parsing_errors=True,  # 自动处理解析错误
        max_iterations=5,  # 最大迭代次数，防止无限循环
        early_stopping_method="generate"  # 提前停止方法
    )

# 2. Structured Chat Agent与StructuredPrompt的结合
def create_enhanced_structured_chat_agent():
    """创建增强版Structured Chat Agent，专注于结构化对话和严格的流程控制
    
    Structured Chat Agent的核心特点：
    1. 结构化对话：使用预定义的对话结构和格式
    2. 严格的流程控制：按照特定步骤和规则处理任务
    3. 数据验证：使用Pydantic模型确保输入输出符合规范
    
    工作原理：
    1. 定义结构化数据模型：使用Pydantic创建强类型的数据结构
    2. 创建专业化工具：每个工具都有明确的输入输出规范
    3. 构建结构化提示：使用StructuredPrompt定义对话流程
    4. 执行严格的验证：确保所有操作符合预定义规则
    
    应用场景：
    - 项目管理：需要严格的任务跟踪和进度管理
    - 数据处理：需要保证数据格式和质量的场景
    - 规范化流程：需要标准化操作流程的场景
    """
    
    # 创建回调处理器，用于监控执行过程
    handlers = [StdOutCallbackHandler(), FileCallbackHandler("agent_log.txt"), CustomCallbackHandler()]
    # 初始化语言模型
    llm = ChatOpenAI(
        model=os.getenv("MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
        temperature=0.7, callbacks=handlers)
    
    # 使用Pydantic定义项目任务的数据模型
    # 这确保了数据的类型安全和验证
    class ProjectTask(BaseModel):
        project_name: str = Field(description="项目名称")
        objectives: List[str] = Field(description="项目目标列表")
        milestones: List[dict] = Field(description="项目里程碑，包含时间和目标")
        resources: List[str] = Field(description="所需资源列表")
        risks: List[dict] = Field(description="风险评估列表")
        progress: float = Field(description="项目进度百分比", ge=0, le=100)
        status_report: str = Field(description="项目状态报告")
    
    # 创建专业化工具，每个工具都有明确的输入输出规范
    def analyze_requirements(requirements: str) -> str:
        """需求分析工具：深入分析项目需求，生成结构化报告"""
        return f"需求分析: {requirements}的详细分析报告"
    
    def create_project_plan(objectives: List[str]) -> str:
        """项目计划工具：基于目标创建详细的实施方案"""
        return f"项目计划: 基于{objectives}的详细实施方案"
    
    def track_progress(project_name: str) -> str:
        """进度追踪工具：监控和报告项目执行状态"""
        return f"进度追踪: {project_name}的当前执行状态"
    
    # 定义工具列表，每个工具都有明确的功能描述
    tools = [
        Tool(name="requirement_analyzer", func=analyze_requirements, description="需求分析工具，提供详细的需求分析报告"),
        Tool(name="plan_creator", func=create_project_plan, description="项目计划生成工具，创建详细的实施方案"),
        Tool(name="progress_tracker", func=track_progress, description="进度追踪工具，监控项目执行状态")
    ]
    
    # 创建结构化提示模板
    # 使用StructuredPrompt确保对话遵循预定义的格式和流程
    structured_prompt = StructuredPrompt(
        messages=[
            ("system", """你是一个专业的项目管理助手，擅长结构化项目规划和执行监控。
            请严格按照以下步骤处理项目：
            1. 需求分析：深入理解项目需求
            2. 目标设定：制定清晰可衡量的目标
            3. 计划制定：创建详细的项目计划
            4. 资源分配：确定所需资源
            5. 风险评估：识别和评估潜在风险
            6. 进度监控：追踪项目执行情况
            7. 状态报告：生成规范的状态报告
            
            可用工具：
            {tools}
            
            工具名称列表：
            {tool_names}
            
            执行记录：
            {agent_scratchpad}"""),
            ("human", "{task_description}"),
            ("ai", "我将按照结构化流程处理您的项目。")
        ],
        schema_=ProjectTask  # 使用ProjectTask模型验证输出
    )
    
    # 打印格式化后的prompt内容
    print("\n=== Structured Chat Agent Prompt ===\n")
    print(structured_prompt.format(
        task_description="示例任务",
        tools=tools,
        tool_names=[tool.name for tool in tools],
        agent_scratchpad=""
    ))
    
    # 创建Structured Chat Agent
    structured_agent = create_structured_chat_agent(llm, tools, structured_prompt)
    
    # 创建并返回AgentExecutor
    return AgentExecutor(
        agent=structured_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate"
    )

# 3. OpenAI Functions Agent与ChatPromptTemplate的结合
def create_enhanced_openai_functions_agent():
    """创建增强版OpenAI Functions Agent，专注于函数调用和参数处理
    
    OpenAI Functions Agent的特点：
    1. 函数调用能力：能够理解和执行复杂的函数调用
    2. 参数验证：确保函数参数的正确性和完整性
    3. 结果处理：智能处理函数返回值
    
    工作原理：
    1. 函数注册：将可用函数注册到Agent
    2. 参数解析：分析用户需求，提取函数参数
    3. 函数执行：调用相应函数并处理结果
    4. 结果整合：将多个函数调用结果组合成完整答案
    """
    
    handlers = [StdOutCallbackHandler(), FileCallbackHandler("agent_log.txt"), CustomCallbackHandler()]
    llm = ChatOpenAI(
        model=os.getenv("MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
        temperature=0.7, callbacks=handlers)
    
    # 创建专业化函数工具
    def execute_function(func_name: str, params: dict) -> str:
        """函数执行工具：执行指定函数并返回结果
        
        参数：
        - func_name: 要执行的函数名
        - params: 函数参数字典
        
        返回：
        - 函数执行结果
        """
        return f"执行函数 {func_name} 参数: {params}"
    
    def validate_params(params: dict, schema: dict) -> str:
        """参数验证工具：验证参数是否符合schema定义
        
        参数：
        - params: 待验证的参数字典
        - schema: 参数验证模式
        
        返回：
        - 验证结果
        """
        return f"参数验证结果: 验证{params}是否符合{schema}"
    
    def format_result(result: str, output_format: str) -> str:
        """结果格式化工具：将结果转换为指定格式
        
        参数：
        - result: 原始结果
        - output_format: 目标格式
        
        返回：
        - 格式化后的结果
        """
        return f"格式化结果: 将{result}转换为{output_format}格式"
    
    # 定义工具列表
    tools = [
        Tool(name="function_executor", func=execute_function, description="高级函数执行工具，支持复杂参数处理"),
        Tool(name="param_validator", func=validate_params, description="参数验证工具，确保输入参数符合规范"),
        Tool(name="result_formatter", func=format_result, description="结果格式化工具，支持多种输出格式")
    ]
    
    # 创建专业化对话模板
    # 使用ChatPromptTemplate构建结构化的对话流程
    system_template = """你是一个专业的函数调用专家，擅长处理{domain}领域的复杂函数调用。
    请遵循以下调用流程：
    1. 参数验证：确保所有输入参数符合规范
    2. 函数执行：按照正确的顺序调用函数
    3. 结果处理：格式化和优化输出结果
    4. 错误处理：妥善处理异常情况
    5. 性能优化：注意函数调用的效率
    
    可用工具：{tools}
    工具名称：{tool_names}
    """
    
    human_template = """请处理以下函数调用任务：
    函数名称：{function_name}
    输入参数：{parameters}
    期望输出：{expected_output}
    
    执行记录：{agent_scratchpad}"""
    
    ai_template = "我将按照专业流程处理您的函数调用请求。"
    
    # 创建ChatPromptTemplate，组合系统、用户和AI消息
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
        AIMessagePromptTemplate.from_template(ai_template)
    ])
    
    # 打印格式化后的prompt内容
    print("\n=== OpenAI Functions Agent Prompt ===\n")
    print(chat_prompt.format(
        domain="通用",
        tools=tools,
        tool_names=[tool.name for tool in tools],
        function_name="示例函数",
        parameters="{}",
        expected_output="示例输出",
        agent_scratchpad=""
    ))
    
    # 创建OpenAI Functions Agent
    openai_agent = create_openai_functions_agent(llm, tools, chat_prompt)
    
    # 配置并返回AgentExecutor
    return AgentExecutor(
        agent=openai_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate"
    )

# 4. JSON Agent与StructuredPrompt和FewShotPromptTemplate的结合
def create_enhanced_json_agent():
    """创建增强版JSON Agent，专注于数据验证和格式转换
    
    JSON Agent的特点：
    1. Schema验证：确保JSON数据符合预定义的格式
    2. 格式转换：支持不同JSON格式之间的转换
    3. 结构分析：分析JSON数据的结构特征
    """
    
    handlers = [StdOutCallbackHandler(), FileCallbackHandler("agent_log.txt"), CustomCallbackHandler()]
    llm = ChatOpenAI(
        model=os.getenv("MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
        temperature=0.7, callbacks=handlers)
    
    # 定义简化的JSON验证规则模型
    class JsonValidationRules(BaseModel):
        """JSON验证规则模型"""
        schema_version: str = Field(description="JSON Schema版本")
        validation_result: str = Field(description="验证结果")
    
    # 创建专业化工具
    def validate_json_schema(json_str: str, schema: dict) -> str:
        """JSON Schema验证工具"""
        return f"Schema验证: 验证{json_str}是否符合{schema}"
    
    def transform_json_format(json_str: str, target_format: str) -> str:
        """JSON格式转换工具"""
        return f"格式转换: 将{json_str}转换为{target_format}格式"
    
    def analyze_json_structure(json_str: str) -> str:
        """JSON结构分析工具"""
        return f"结构分析: 分析{json_str}的结构特征和复杂度"
    
    tools = [
        Tool(name="schema_validator", func=validate_json_schema, description="JSON Schema验证工具，支持复杂规则验证"),
        Tool(name="format_transformer", func=transform_json_format, description="JSON格式转换工具，支持多种格式间转换"),
        Tool(name="structure_analyzer", func=analyze_json_structure, description="JSON结构分析工具，提供深度分析报告")
    ]
    
    # 创建简化的结构化提示模板
    json_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """你是一个专业的JSON处理专家，擅长数据验证和格式转换。
            请按照以下步骤处理JSON数据：
            1. 验证JSON格式的正确性
            2. 分析数据结构特征
            3. 执行必要的格式转换
            4. 提供处理建议
            
            可用工具：{tools}
            工具名称：{tool_names}"""),
        HumanMessagePromptTemplate.from_template("请处理以下JSON数据：{json_input}"),
        AIMessagePromptTemplate.from_template("我将专业地处理您的JSON数据。")
    ])
    
     # 打印格式化后的prompt内容
    print("\n=== JSON Agent Prompt ===\n")
    print(json_prompt.format(
        json_input="示例JSON数据",
        tools=tools,
        tool_names=[tool.name for tool in tools]
    ))

    # 创建JSON工具包
    json_spec = JsonSpec(dict_={
        "name": "JSON处理器",
        "version": "1.0",
        "capabilities": ["schema验证", "格式转换", "结构分析"]
    })
    json_toolkit = JsonToolkit(spec=json_spec)
    
    # 创建JSON Agent
    json_agent = create_json_agent(
        llm=llm,
        toolkit=json_toolkit,
        prompt=json_prompt
    )
    
    # 配置并返回AgentExecutor
    return AgentExecutor(
        agent=json_agent,
        tools=json_toolkit.get_tools(),
        verbose=True,
        handle_parsing_errors=True
    )

# 5. SQL Agent与StructuredPrompt的结合
def create_enhanced_sql_agent():
    """创建增强版SQL Agent，结合StructuredPrompt
    
    SQL Agent的特点：
    1. 自然语言转SQL：将用户需求转换为SQL查询
    2. 查询优化：自动优化SQL查询性能
    3. 结果解释：提供查询结果的详细解释
    
    工作原理：
    1. 需求分析：理解用户的数据查询需求
    2. SQL生成：构建符合需求的SQL查询
    3. 查询执行：安全地执行SQL语句
    4. 结果处理：格式化和解释查询结果
    """
    
    handlers = [StdOutCallbackHandler(), FileCallbackHandler("agent_log.txt"), CustomCallbackHandler()]
    llm = ChatOpenAI(
        model=os.getenv("MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
        temperature=0.7, callbacks=handlers)
    
    # 定义SQL查询结果模型
    class SQLQueryResult(BaseModel):
        query: str = Field(description="SQL查询语句")
        result: str = Field(description="查询结果")
        explanation: str = Field(description="查询说明")
    
    # 创建专业化工具
    def validate_sql(sql: str) -> str:
        return f"SQL验证: {sql}的语法和安全性检查"
    
    def optimize_query(sql: str) -> str:
        return f"查询优化: 优化{sql}的性能"
    
    def explain_result(result: str) -> str:
        return f"结果解释: {result}的详细分析"
    
    # 定义工具列表
    tools = [
        Tool(name="sql_validator", func=validate_sql, description="SQL验证工具，检查语法和安全性"),
        Tool(name="query_optimizer", func=optimize_query, description="查询优化工具，提升SQL性能"),
        Tool(name="result_explainer", func=explain_result, description="结果解释工具，提供详细分析")
    ]
    
    # 创建SQL提示模板
    sql_prompt = StructuredPrompt(
        messages=[
            ("system", """你是一个专业的SQL专家，擅长数据库查询和优化。
            请按照以下步骤处理SQL任务：
            1. 语法验证：检查SQL语句的正确性
            2. 安全检查：防止SQL注入和其他安全问题
            3. 性能优化：优化查询执行计划
            4. 结果分析：提供查询结果的详细解释
            
            可用工具：
            {tools}
            
            工具名称列表：
            {tool_names}
            
            执行记录：
            {agent_scratchpad}"""),
            ("human", "{query}"),
            ("ai", "我将按照专业标准处理您的SQL查询。")
        ],
        schema_=SQLQueryResult
    )

    db = SQLDatabase.from_uri("sqlite:///:memory:")
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    # 打印格式化后的prompt内容

    print("=== SQL Agent Prompt ===")
    print(sql_prompt.format(
        query="示例SQL查询",
        tools=tools,
        tool_names=[tool.name for tool in tools],
        agent_scratchpad=""
    ))
    
    # 创建SQL Agent
    sql_agent = create_sql_agent(
        llm=llm,
        toolkit=sql_toolkit,
        prompt=sql_prompt
    )
    
    # 配置并返回AgentExecutor
    return AgentExecutor(
        agent=sql_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

# 6. Tool Calling Agent与PipelinePromptTemplate和ImagePromptTemplate的结合
def create_enhanced_tool_calling_agent():
    """创建增强版Tool Calling Agent，专注于工具链和多模态处理
    
    Tool Calling Agent的特点：
    1. 工具链管理：协调多个工具的顺序调用
    2. 多模态处理：支持图像、音频等多种数据类型
    3. 结果整合：将多个工具的结果整合成统一输出
    
    工作原理：
    1. 任务分析：理解用户需求，规划工具调用顺序
    2. 工具选择：根据任务需求选择合适的工具
    3. 参数准备：为每个工具准备所需参数
    4. 结果整合：将多个工具的输出组合成最终结果
    """
    
    handlers = [StdOutCallbackHandler(), FileCallbackHandler("agent_log.txt"), CustomCallbackHandler()]
    llm = ChatOpenAI(
        model=os.getenv("MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
        temperature=0.7, callbacks=handlers)
    
    # 创建多模态处理工具
    def process_image(image_url: str, params: dict) -> str:
        """图像处理工具：支持多种图像处理操作"""
        return f"图片处理: 对{image_url}进行处理，参数:{params}"
    
    def process_audio(audio_url: str, params: dict) -> str:
        """音频处理工具：支持音频转换和分析"""
        return f"音频处理: 对{audio_url}进行处理，参数:{params}"
    
    def process_text(text: str, mode: str) -> str:
        """文本处理工具：支持多种文本处理模式"""
        return f"文本处理: 以{mode}模式处理文本:{text}"
    
    def integrate_results(results: List[dict]) -> str:
        """结果整合工具：将多个处理结果合并"""
        return f"结果整合: 整合多模态处理结果:{results}"
    
    # 定义工具列表
    tools = [
        Tool(name="image_processor", func=process_image, description="高级图片处理工具，支持多种处理参数"),
        Tool(name="audio_processor", func=process_audio, description="音频处理工具，支持多种音频格式"),
        Tool(name="text_processor", func=process_text, description="文本处理工具，支持多种处理模式"),
        Tool(name="result_integrator", func=integrate_results, description="结果整合工具，合并多模态处理结果")
    ]
    
    # 1. 创建图像处理模板
    image_prompt = ImagePromptTemplate(
        template={
            "url": "{image_url}",
            "detail_level": "{detail_level}",
            "processing_type": "{processing_type}",
            "output_format": "{output_format}"
        },
        template_format="f-string",
        input_variables=["image_url", "detail_level", "processing_type", "output_format"]
    )
    
    # 2. 创建交互对话模板
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """你是一个专业的多模态处理专家，擅长处理图像、音频和文本数据。
            请遵循以下处理流程：
            1. 分析输入数据类型和要求
            2. 选择合适的处理工具
            3. 设置处理参数
            4. 执行处理操作
            5. 整合处理结果
            
            可用工具：
            {tools}
            
            工具名称列表：
            {tool_names}

            执行记录：
            {agent_scratchpad}
            """
        ),
        HumanMessagePromptTemplate.from_template("{user_input}"),
        AIMessagePromptTemplate.from_template("我将按专业流程处理您的请求。")
    ])
    
    # 3. 创建分析模板
    analysis_template = PromptTemplate(
        template="""分析任务：{input}
        
        处理要求：
        1. 数据类型：{data_type}
        2. 处理参数：{params}
        3. 输出格式：{output_format}
        
        请按照以下格式回答：
        Thought: 思考当前的问题
        Action: 选择要使用的工具
        Action Input: 工具的输入参数
        Observation: 工具的输出结果
        Thought: 基于结果继续思考
        Action: 下一步使用的工具
        ... (重复上述步骤)
        Final Answer: 最终答案
        
        执行记录：
        {agent_scratchpad}""",
        input_variables=["input", "data_type", "params", "output_format", "agent_scratchpad"]
    )
    
    # 4. 使用PipelinePromptTemplate串联所有模板
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=analysis_template,
        pipeline_prompts=[
            ("data_type", image_prompt),
            ("params", chat_prompt)
        ],
        input_variables=["input", "image_url", "detail_level", "processing_type", "output_format", "user_input"]
    )
    
    # 打印格式化后的pipeline_prompt内容
    print("\n=== Tool Calling Agent Pipeline Prompt ===\n")
    
    # 创建一个包含所有必要变量的字典
    format_kwargs = {
        "input": "示例多模态处理任务",
        "image_url": "https://example.com/sample.jpg",
        "detail_level": "high",
        "processing_type": "enhancement",
        "output_format": "json",
        "user_input": "处理一张图片",
        "tools": tools,
        "tool_names": [tool.name for tool in tools],
        "agent_scratchpad": ""
    }
    
    # 使用完整的参数字典格式化提示模板
    print(pipeline_prompt.format(**format_kwargs))
    
    # 创建工具调用Agent
    tool_calling_agent = create_tool_calling_agent(llm, tools, pipeline_prompt)
    
    # 配置并返回AgentExecutor
    return AgentExecutor(
        agent=tool_calling_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="generate"
    )

# 主函数：展示不同类型Agent的使用方法
def main():
    """主函数：创建并测试不同类型的Agent
    
    本函数展示了如何：
    1. 创建不同类型的Agent
    2. 设计针对性的测试任务
    3. 执行任务并分析结果
    4. 展示中间执行步骤
    """
    # 创建不同类型的Agent
    agents = {
       # "ReAct": create_enhanced_react_agent(),
      #  "Structured Chat": create_enhanced_structured_chat_agent(),
      #  "OpenAI Functions": create_enhanced_openai_functions_agent(),
      #  "JSON": create_enhanced_json_agent(),
      #  "SQL": create_enhanced_sql_agent(),
        "Tool Calling": create_enhanced_tool_calling_agent()
    }
    
    # 针对性测试任务
    test_tasks = {
        "ReAct": "分析一篇关于'量子计算在人工智能中的应用'的文章：首先搜索相关信息，然后分析文章的技术深度和情感倾向，最后给出综合评估。这个任务需要多步推理和信息整合。",
        "Structured Chat": "请完成以下多语言翻译任务：1. 将'人工智能正在重塑未来商业模式'翻译成英文 2. 将结果格式化为JSON格式 3. 添加语言检测信息。要求严格按照步骤执行。",
        "OpenAI Functions": "执行以下数学分析任务：1. 计算(15 * 8 + 27) / 3的值 2. 分析'这是一个需要多步计算的数学问题'这句话的情感 3. 总结计算过程的复杂度",
        "JSON": {
            "json_input": '{"user": {"id": 123, "name": "张三"}}',
            "query": "请验证这个用户数据的结构"
        },
        "SQL": "针对用户数据库执行以下分析：1. 查询所有年龄在25-35岁之间的活跃用户 2. 解释查询的执行计划 3. 验证SQL语法的正确性",
        "Tool Calling": "完成以下多模态任务链：1. 搜索最新的AI研究进展 2. 分析文章的情感倾向 3. 将分析结果翻译成英文 4. 将最终结果格式化为结构化数据"
    }

    # 使用不同类型的Agent执行测试任务
    for agent_type, agent in agents.items():
        print(f"\n=== 使用 {agent_type} Agent ===")
        task = test_tasks[agent_type]
        print(f"\n执行任务: {task}")
        
        # 根据Agent类型选择不同的调用方式
        if agent_type == "JSON":
            result = agent.invoke(task)  # 直接传入包含json_input和query的字典
        elif agent_type == "Tool Calling":
            # 为Tool Calling Agent提供所有必需的变量
            result = agent.invoke({
                "input": task,
                "image_url": "https://example.com/sample.jpg",
                "detail_level": "high",
                "processing_type": "enhancement",
                "output_format": "json",
                "user_input": task,
            })
        else:
            result = agent.invoke({"input": task})
            
        print(f"执行结果: {result['output']}")
        
        # 如果具有中间步骤，打印出来以便分析
        if result.get("intermediate_steps"):
            print("\n执行步骤:")
            for step in result["intermediate_steps"]:
                print(f"- {step}")

# 程序入口
if __name__ == "__main__":
    main()