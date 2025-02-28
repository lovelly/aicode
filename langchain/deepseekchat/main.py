from langchain_deepseek import ChatDeepSeek  #导 入deepseek库
from langchain.prompts import PromptTemplate  # 导入PromptTemplate模块
from dotenv import load_dotenv  # 导入dotenv库，用于加载环境变量


load_dotenv()  # 加载.env文件中的环境变量

def generate_name(gender):
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )  # 创建OpenAI模型的实例，设置temperature参数为0.7以调整生成的多样性

    # 创建PromptTemplate实例，用于构造输入提示
    prompt_template_name = PromptTemplate(
        input_variables = ['gender'],
        template = "I have a {gender} baby and I want a cool name for them. Suggest me five cool names for my {gender} baby."
    )
    name_chain = prompt_template_name | llm   # 创建LLMChain实例，将OpenAI模型和PromptTemplate传入
    response = name_chain.invoke({'gender': gender})  # chain调用

    return response.content  # 返回消息文本

# 当该脚本作为主程序运行时，执行以下代码
if __name__ == "__main__":
    print(generate_name('girl'))  # 调用generate_name函数
