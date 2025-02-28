# 使用LangChain和RAG构建智能名字推荐系统
在这个教程中，我们将深入探讨如何使用LangChain框架和检索增强生成（RAG）技术构建一个智能的婴儿名字推荐系统。该系统不仅能够提供名字建议，还能基于自定义知识库提供每个名字的含义和文化背景。




## 目录
1. [RAG技术概述](##1.RAG技术概述)
2. [RAG调用流程](##2.RAG调用流程)
3. [代码详解](##3.代码详解)
4. [总结](##4.总结)

## 1.RAG技术概述
检索增强生成（Retrieval-Augmented Generation，RAG）是一种结合了信息检索和文本生成的混合技术。RAG的核心思想是：

首先从知识库中检索与用户查询相关的信息
然后将检索到的信息作为上下文，与用户的查询一起输入到语言模型中
最后由语言模型生成综合了检索信息和模型知识的高质量回答
这种方法有几个显著优势：

减少幻觉：通过提供外部知识，减少模型生成虚假或不准确的信息
知识更新：可以不断更新知识库，而无需重新训练模型
透明度：可以清楚地看到回答的信息来源
定制化：可以针对特定领域构建专门的知识库
在我们的名字推荐系统中，RAG使我们能够基于特定的名字数据库提供更有文化内涵和历史背景的名字建议。

环境

## 2.RAG调用流程
```bash
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│                │     │                │     │                │
│  文本语料库     │────▶│  向量数据库     │────▶│  检索系统      │
│                │     │                │     │                │
└────────────────┘     └────────────────┘     └───────┬────────┘
                                                      │
                                                      ▼
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│                │     │                │     │                │
│  用户查询       │────▶│  提示工程      │◀────│  相关上下文    │
│                │     │                │     │                │
└────────────────┘     └───────┬────────┘     └────────────────┘
                               │
                               ▼
                       ┌────────────────┐
                       │                │
                       │  DeepSeek LLM  │
                       │                │
                       └───────┬────────┘
                               │
                               ▼
                       ┌────────────────┐
                       │                │
                       │  生成回复       │
                       │                │
                       └────────────────┘
```


## 3.代码详解

### 3.1.依赖

```py
from langchain_deepseek import ChatDeepSeek  # 导入deepseek库
from langchain.prompts import PromptTemplate  # 导入PromptTemplate模块
from dotenv import load_dotenv  # 导入dotenv库，用于加载环境变量
from langchain_community.vectorstores import Chroma  # 导入向量存储
from langchain_community.embeddings import HuggingFaceEmbeddings  # 导入嵌入模型
from langchain_community.document_loaders import TextLoader  # 导入文档加载器
from langchain.text_splitter import CharacterTextSplitter  # 导入文本分割器
from langchain.chains import RetrievalQA  # 导入RetrievalQA
from langchain.schema.runnable import RunnablePassthrough  # 导入RunnablePassthrough

import os

load_dotenv()  # 加载.env文件中的环境变量
```
- ChatDeepSeek：提供大语言模型的接口，负责生成文本回复
- PromptTemplate：用于构建结构化提示，帮助模型理解任务需求
- dotenv & load_dotenv：管理环境变量，特别是API密钥等敏感信息
- Chroma：向量数据库，用于存储和检索文本嵌入
- HuggingFaceEmbeddings：将文本转换为向量嵌入的模型
- TextLoader：加载文本文件
- CharacterTextSplitter：将长文本拆分成小块，便于处理
- RetrievalQA：构建检索问答链
- RunnablePassthrough：在LangChain链中传递变量

### 3.2 向量数据库的创建与管理
```py
# 设置向量数据库路径
CHROMA_PATH = "chroma_db"

def create_vector_db(documents_path):
    """创建或加载向量数据库"""
    # 加载文档
    loader = TextLoader(documents_path)
    documents = loader.load()
    
    # 分割文档
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    
    # 创建嵌入
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 创建向量存储
    if os.path.exists(CHROMA_PATH):
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        print("加载现有向量数据库")
    else:
        vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
        vector_store.persist()
        print("创建并保存新的向量数据库")
    
    return vector_store
```
- 文档加载：使用TextLoader从文件中加载文本数据。这里我们假设有一个包含名字及其含义的文本文件。文档加载是RAG的第一步，确定信息源。
- 文本分割：使用CharacterTextSplitter将文档分成较小的块。
- 为什么要分割？ 大多数嵌入模型和检索系统对输入长度有限制，分割可以更精确地检索相关内容。
chunk_size=1000：每个块的最大字符数为1000
chunk_overlap=0：块之间不重叠（可以设置重叠来避免上下文割裂）
文本嵌入：使用HuggingFaceEmbeddings将文本转换为向量。
- 我们选择了"sentence-transformers/all-MiniLM-L6-v2"模型，这是一个轻量级但效果良好的嵌入模型。
嵌入是RAG的核心，它将文本转化为向量空间中的点，使得语义相似的文本在向量空间中也更接近。
向量存储：使用Chroma作为向量数据库。
- 持久化存储：使用persist_directory参数指定存储位置，使数据库可以跨会话保存。
重用机制：检查数据库是否已存在，避免重复创建。

向量数据库对比

| 特性           | Chroma   | FAISS   | Pinecone |
|----------------|----------|---------|----------|
| 部署方式       | 本地/嵌入式 | 本地    | 云服务   |
| 持久化存储     | ✅        | ❌      | ✅       |
| 多模态支持     | ✅        | ❌      | ✅       |
| 自动更新       | ✅        | 手动    | ✅       |


### 3.3 实现RAG处理流程
```py
def generate_name_with_rag(gender, context_file=None):
    # 初始化LLM
    llm = ChatDeepSeek(
        model="deepseek-chat",
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
    
    # 如果提供了上下文文件，则使用RAG流程
    if context_file:
        # 创建或加载向量数据库
        vector_store = create_vector_db(context_file)
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
        # 如果没有上下文文件，执行基本链
        basic_chain = base_prompt_template | llm
        response = basic_chain.invoke({'gender': gender})
    
    return response.content
```
- LLM初始化：
temperature=0.7：控制生成的随机性，较高的值会产生更多样化但可能不太集中的输出。
max_tokens=None：不限制输出的令牌数。
max_retries=2：API调用失败时最多重试2次。
- 两种工作模式：
基本模式：当没有提供上下文文件时，直接使用基本提示模板。
RAG模式：当提供上下文文件时，使用检索增强生成流程。
- 检索器配置：
search_kwargs={"k": 3}：检索最相关的3条记录作为上下文。
这个参数非常重要，太少可能信息不足，太多会引入噪音并可能超出上下文窗口。
- RAG链的构建：
rag_chain = (
    {"context": retriever, "gender": RunnablePassthrough()}
    | rag_prompt_template
    | llm
)
这里使用了LangChain的链式API：
首先创建一个字典，其中context键对应检索器，gender键对应直接传入的值。
然后将结果传给提示模板格式化。
最后将格式化的提示传给语言模型生成回答。
- RunnablePassthrough：
这是一个特殊的组件，允许将输入直接传递给下一个组件，而不进行任何处理。
在这里，我们用它来传递gender参数。

### 3.4提示词工程
- PromptTemplate 示例中， 我们使用了最基础的纯文本模板，除了基础模板， 还有
  - ChatPromptTemplate  
    - 为对话型模型(如ChatGPT/Claude)设计的提示模板,它被设计用于更复杂的对话系统，尤其是处理多轮对话时,你通常会看到角色（例如：系统、用户）之间的对话历史。
    - 生成一系列消息对象(ChatMessages)，而非纯文本
  - SystemMessagePromptTemplate
    - 专门用于创建系统消息的模板
    - 系统消息代表对AI助手的指令或设定其行为的信息，通常放在对话的开头，设置整个交互的基调和规则，不被视为用户输入，而是模型应遵循的指导方针。
  - HumanMessagePromptTemplate
    - 专门用于创建角色消息的模板, 主要用于生成或格式化用户的输入消息。在对话系统中，用户与系统之间的交互通常由用户发起的消息组成

综合示例
```py
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI  # 可以替换为DeepSeek或其他模型

# 创建系统消息模板
system_message = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant that specializes in {specialty}."
)

# 创建人类消息模板
human_message = HumanMessagePromptTemplate.from_template(
    "Can you suggest five {gender} baby names with {characteristic} characteristics?"
)

# 组合成聊天提示模板
chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    human_message
])

# 生成格式化消息
messages = chat_prompt.format_messages(
    specialty="multicultural naming traditions",
    gender="unisex",
    characteristic="nature-inspired"
)

# 初始化LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # 或其他模型
    temperature=0.7,
    max_tokens=500
)

lcel_chain = chat_prompt | llm

lcel_response = lcel_chain.invoke({
    "specialty": "multicultural naming traditions",
    "gender": "unisex",
    "characteristic": "nature-inspired"
})

print(lcel_response.content)

```

### 3.5主调用流程
```py
def generate_name(gender):
    return generate_name_with_rag(gender)

# 当该脚本作为主程序运行时，执行以下代码
if __name__ == "__main__":
    # 使用原始生成（不使用RAG）
    print("=== 不使用RAG的名字生成 ===")
    print(generate_name('girl'))
    
    # 使用RAG进行生成（如果有名字数据文件）
    print("\n=== 使用RAG的名字生成 ===")
    # 假设有一个包含名字信息的文本文件
    context_file = "ragname.txt"  # 替换为实际的文件路径
    
    print(generate_name_with_rag('girl', context_file))
```

## 4.总结
这个系统展示了RAG的强大之处：它能够将语言模型的生成能力与特定领域知识结合起来，提供更丰富、更准确的信息。这种方法不仅适用于名字推荐，还可以扩展到各种需要结合专业知识的生成任务中。

最重要的是，由于RAG使用的是外部知识库而非模型内部知识，您可以随时更新知识库以反映最新的信息，而无需重新训练模型，这使得系统更加灵活和可维护。


git源码：https://github.com/lovelly/aicode.git