from langchain_deepseek import ChatDeepSeek  # 导入deepseek库
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder  # 导入PromptTemplate模块
from langchain.schema.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv  # 导入dotenv库，用于加载环境变量
from langchain_chroma import Chroma  # 导入向量存储
from langchain_huggingface import HuggingFaceEmbeddings  # 导入嵌入模型
from langchain_community.document_loaders import TextLoader, WebBaseLoader  # 导入文档加载器
from langchain.text_splitter import CharacterTextSplitter  # 导入文本分割器
from langchain.chains import RetrievalQA  # 导入RetrievalQA
from langchain.schema.runnable import RunnablePassthrough  # 导入RunnablePassthrough
from langchain.tools import tool  # 导入工具装饰器
from langchain.agents import AgentExecutor, create_openai_functions_agent  # 导入Agent
from langchain_community.tools import DuckDuckGoSearchRun  # 导入搜索工具
from pydantic import BaseModel, Field  # 导入Pydantic模型
from typing import List, Optional
from langchain.memory import ConversationBufferMemory  # 导入内存组件
from langchain.callbacks import FileCallbackHandler, StdOutCallbackHandler  # 导入回调处理器
from langchain.callbacks.manager import CallbackManager  # 导入回调管理器
from langchain.agents import create_react_agent

import os
import requests
from datetime import datetime

load_dotenv()  # 加载.env文件中的环境变量
os.environ["USER_AGENT"] = "myagent/1.0"
# 设置向量数据库路径
CHROMA_PATH = "chroma_db"

# 定义用于存储搜索结果的文件夹
SEARCH_RESULTS_DIR = "search_results"
if not os.path.exists(SEARCH_RESULTS_DIR):
    os.makedirs(SEARCH_RESULTS_DIR)

# 创建日志文件夹
LOGS_DIR = "logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# 创建回调管理器
def create_callback_manager():
    """创建回调管理器，用于追踪工具调用"""
    # 创建文件回调处理器
    # log_file_path = os.path.join(LOGS_DIR, f"langchain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    # file_handler = FileCallbackHandler(log_file_path)
    
    # # 创建标准输出回调处理器
    # stdout_handler = StdOutCallbackHandler()
    
    # 创建回调管理器
    return CallbackManager([StdOutCallbackHandler()])

# 创建或获取向量数据库
def get_vector_db():
    """创建或加载向量数据库"""
    # 创建嵌入
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 创建或加载向量存储
    if os.path.exists(CHROMA_PATH):
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        print("加载现有向量数据库")
    else:
        # 如果不存在，创建一个空的向量数据库
        vector_store = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PATH)
        vector_store.persist()
        print("创建新的空向量数据库")
    
    return vector_store, embeddings

# 定义搜索工具的输入模型
class SearchInput(BaseModel):
    query: str = Field(description="搜索查询内容")
    num_results: Optional[int] = Field(default=3, description="返回的搜索结果数量")

# 定义RAG工具的输入模型
class RagStoreInput(BaseModel):
    content: str = Field(description="要存储到RAG知识库的内容")

# 定义名字生成工具的输入模型
class NameGeneratorInput(BaseModel):
    gender: str = Field(description="宝宝的性别 (boy/girl)")
    use_rag: bool = Field(default=True, description="是否使用知识库进行名字生成")

# 定义web内容加载工具的输入模型
class WebContentInput(BaseModel):
    url: str = Field(description="要加载的网页URL")

# 创建工具
@tool("web_search", args_schema=SearchInput)
def web_search(query: str, num_results: int = 10) -> str:
    """使用DuckDuckGo搜索互联网获取信息"""
    print(f"执行搜索: {query}，数量: {num_results}")  # 调试输出
    search = DuckDuckGoSearchRun()
    results = search.run(f"{query}")

    if not results:
        return "没有找到任何结果。请尝试更改查询。"
    
    # 将搜索结果保存到文件中
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SEARCH_RESULTS_DIR}/search_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"搜索查询: {query}\n\n")
        f.write(f"搜索结果:\n{results}\n")
    
    return f"搜索完成，结果已保存到 {filename}。搜索结果摘要:\n{results[:1024]}...(已截断)"

@tool("store_to_rag", args_schema=RagStoreInput)
def store_to_rag(content: str) -> str:
    """将内容存储到RAG知识库中。输入是一个dict"""

    print(f"存储到RAG: 内容长度: {len(content)}")  # 调试输出
    # 获取向量数据库和嵌入模型
    vector_store, embeddings = get_vector_db()
    
    # 分割文本
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(content)
    
    # 创建文档对象并添加元数据
    from langchain.schema import Document
    docs = [Document(page_content=chunk, metadata={"source": "unknow"}) for chunk in chunks]
    
    # 将文档添加到向量数据库
    vector_store.add_documents(docs)
    
    return f"已将 {len(chunks)} 个文本块存储到RAG知识库中"

@tool("load_web_content", args_schema=WebContentInput)
def load_web_content(url: str) -> str:
    """从网页加载内容，并返回文本内容"""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        content = "\n\n".join([doc.page_content for doc in docs])
        return f"已成功加载网页内容: {url}\n{content[:300]}...(内容已截断)"
    except Exception as e:
        return f"加载网页内容时出错: {str(e)}"

@tool("generate_names", args_schema=NameGeneratorInput)
def generate_names(gender: str, use_rag: bool = True) -> str:
    """为男孩或女孩生成名字，可以选择是否使用RAG知识库"""
    # 初始化LLM
    llm = ChatOpenAI(
        model=os.getenv("MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    # 基本提示模板
    base_prompt_template = PromptTemplate(
        input_variables=['gender'],
        template="I have a {gender} baby and I want a cool name for them. Suggest me five cool names for my {gender} baby."
    )
    
    # 如果使用RAG
    if use_rag:
        # 获取向量数据库
        vector_store, _ = get_vector_db()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 创建RAG提示模板
        rag_prompt_template = PromptTemplate(
            input_variables=['context', 'gender'],
            template="""
            Based on the following information about names and their meanings:
            
            {context}
            
            I have a {gender} baby and I want a cool name for them. 
            Suggest me five cool names for my {gender} baby based on the above information.
            For each name, explain its meaning and origin.
            """
        )
        
        # 创建RAG链
        rag_chain = (
            {"context": retriever, "gender": RunnablePassthrough()}
            | rag_prompt_template
            | llm
        )
        
        # 执行RAG链
        response = rag_chain.invoke(gender)
        
    else:
        # 如果不使用RAG，执行基本链
        basic_chain = base_prompt_template | llm
        response = basic_chain.invoke({'gender': gender})
    
    return response.content

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
    
    # 创建系统消息
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

    # 创建PromptTemplate实例时传递正确的变量
    react_prompt = PromptTemplate(
        input_variables=['input', 'tools', 'tool_names', 'agent_scratchpad'],
        template=react_template
    )

    # 打印调试信息以确保变量替换正确
    print("生成的提示模板:", react_prompt.format(
        input="示例输入",
        tools="\n".join([tool.name for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools]),
        agent_scratchpad=""
    ))

     # 创建自定义的工具解析器函数
    def fix_tool_inputs(action, tool_input):
        # 特别处理store_to_rag工具
        if action == "store_to_rag":
            # 检查是否缺少source字段
            if isinstance(tool_input, dict) and "content" in tool_input and "source" not in tool_input:
                # 自动添加source字段
                tool_input["source"] = "从web_search获取的内容"
        return tool_input

    # 创建ReAct agent
    react_agent = create_react_agent(llm, tools, react_prompt)
    # 启用解析错误处理
    agent_executor = AgentExecutor(agent=react_agent, tools=tools,
     verbose=True, handle_parsing_errors=True, max_iterations=3,
      tool_input_fixers=[fix_tool_inputs])
    
    return agent_executor

# 直接使用工具函数而不是通过agent调用
def run_example_manually():
    """直接使用工具函数而不是通过agent调用"""
    print("=== 直接执行工具函数  ===")
   
    # 1. 执行搜索
    print("\n1. 执行搜索:")
    search_results = web_search.invoke({"query": "北欧神话中的女孩名字", "num_results": 3})
    print(f"搜索结果: {search_results}")
    
    # 2. 将搜索结果存储到RAG
    print("\n2. 存储到RAG:")
    store_result = store_to_rag.invoke({"content": search_results, "source": "DuckDuckGo搜索:北欧神话中的女孩名字"})
    print(f"存储结果: {store_result}")
    
    # 3. 使用RAG生成名字
    print("\n3. 使用RAG生成名字:")
    names_with_rag = generate_names.invoke({"gender": "girl", "use_rag": True})
    print(f"RAG名字生成结果: {names_with_rag}")
    
    # 4. 不使用RAG生成名字
    print("\n4. 不使用RAG生成名字:")
    names_without_rag = generate_names.invoke({"gender": "girl", "use_rag": False})
    print(f"非RAG名字生成结果: {names_without_rag}")
    
    # 5. 加载网页内容
    print("\n5. 加载网页内容:")
    web_content = load_web_content.invoke({"url": "https://www.behindthename.com/names/usage/chinese"})
    print(f"网页内容长度: {len(web_content)}")
    
    # 6. 将网页内容存储到RAG
    print("\n6. 将网页内容存储到RAG:")
    store_web_result = store_to_rag.invoke({"content": web_content, "source": "https://www.behindthename.com/names/usage/chinese"})
    print(f"网页内容存储结果: {store_web_result}")

# 示例用法
if __name__ == "__main__":
    # 方式1: 直接执行工具函数
    # run_example_manually()
    
    # 方式2: 通过Agent进行交互
    print("\n=== 通过Agent进行交互 ===")
    agent = create_name_generator_agent()
    
    # 示例1: 请求搜索北欧名字信息
    print("\n=== 示例1: 请求搜索北欧名字信息 ===")
    result1 = agent.invoke({
        "input": "我想了解一些北欧神话中的女孩名字，能帮我搜索相关信息,存储到rag向量库，然后给出回答吗？"
    })
    print("结果:", result1["output"])

