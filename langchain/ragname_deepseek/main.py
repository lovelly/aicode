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