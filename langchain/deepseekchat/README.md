# 使用 LangChain 和 DeepSeek 生成名字 - 入门教程

本教程将帮助你使用 `LangChain` 和 `DeepSeek` 来生成个性化的名字。我们将从安装所需的依赖开始，逐步讲解代码实现过程。

## 1. 环境要求

在开始之前，确保你已经安装了以下工具：

- **Python** 版本 >= 3.7
- **pip** 包管理工具

## 2. 安装依赖

### 2.1 安装 `langchain` 和 `deepseek` 依赖

在项目目录下创建一个虚拟环境，并激活它（可选，但推荐）：

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```
### 2.2 安装依赖库
```bash
pip install langchain deepseek python-dotenv
```
  或者
```bash
pip install -r requirements.txt
```


### 2.3 配置环境变量
 调用deepseek 需要一个apikey 和api url, 从deepseek官网申请
创建.env文件
```
DEEPSEEK_API_KEY = "sk-XXXXXXXXXXXXXX"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
```


## 3. 代码
```py
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

```

## 4. 代码解读
- ChatDeepSeek: 是 DeepSeek 的接口，它使用 Deepseek 的模型来生成文本。你可以调整temperature（ 用于控制模型生成文本时的随机性和创造性。它通过调整模型输出的概率分布，影响生成结果的多样性和确定性。）、最大 token 数（max_tokens）等参数
- PromptTemplate: 这个类用于构建prompt模板 （复用提示词），模板中使用 {} 来嵌入动态变量。在这个例子中，我们使用了 gender 变量来生成个性化的名字。
- name_chain: 这是一个链式调用，使用 | 操作符(LCEL) 将 PromptTemplate 和 ChatDeepSeek 连接起来，使其成为一个可以直接调用的对象。
- invoke: 执行链，并传入特定的输入（这里是 'gender'）。
- LCEL  表达式语言 是通过重载 | 运算符， 类构建了类似Unix管道运算符的设计，实现更简洁的LLM调用形式

## 扩展 
- 你可以修改提示模板中的内容，要求生成不同类型的名字（比如：搞笑名字、可爱的名字等）。
- 你还可以调整 temperature 参数来改变名字的多样性。较高的值会生成更富有创意的名字，较低的值则生成更保守的名字。

git源码：https://github.com/lovelly/aicode.git