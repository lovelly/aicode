from langchain.prompts import (
    PromptTemplate,
    StringPromptTemplate,
    FewShotPromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_core.prompts.pipeline import PipelinePromptTemplate
from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.prompts.image import ImagePromptTemplate
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from typing import List, Dict, Any
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

# 初始化 LLM
def get_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        temperature=0.7,
        callbacks=[StdOutCallbackHandler(), CustomCallbackHandler()]
    )

# 1. PromptTemplate - 基础模板
def basic_prompt_template_example():
    """基础提示模板示例"""
    llm = get_llm()
    
    # 创建产品介绍模板
    # 定义了三个输入变量：product_name, features, price
    # 模板结构清晰，包含了具体的输出要求
    product_prompt = PromptTemplate(
        input_variables=["product_name", "features", "price"],
        template="""请为以下产品生成一个简短的介绍：

产品名称：{product_name}
产品特点：{features}
价格：{price}

请用3-5句话介绍这个产品的主要优势和价值。"""
    )
    
    # 使用format方法填充模板变量
    prompt = product_prompt.format(
        product_name="智能手表 Mini",
        features="防水、心率监测、运动追踪",
        price="599元"
    )
    
    # 调用LLM生成响应
    response = llm.invoke(prompt)
    return response.content

# 2. StringPromptTemplate示例
def string_prompt_template_example():
    """字符串提示模板示例"""
    llm = get_llm()
    
    # 创建自定义的StringPromptTemplate
    class TranslationPromptTemplate(StringPromptTemplate):
        # 使用类型注解定义输入变量
        input_variables: List[str] = ["text", "target_language", "formality_level"]
        
        def format(self, **kwargs) -> str:
            # 自动获取系统时间
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 自动检测源语言（这里用简单逻辑演示）
            def detect_language(text):
                import re
                # 简单判断：如果包含中文字符则认为是中文
                if re.search(r'[\u4e00-\u9fff]', text):
                    return "中文"
                return "英文"
            
            # 扩展变量
            kwargs["timestamp"] = current_time
            kwargs["source_language"] = detect_language(kwargs["text"])
            
            # 使用模板字符串
            template = """[系统时间: {timestamp}]

请将以下文本从{source_language}翻译成{target_language}：

原文：{text}

翻译要求：
- 语言风格：{formality_level}
- 保持原文的语气和风格
- 确保专业术语的准确性
- 适应目标语言的文化背景"""
            
            return template.format(**kwargs)
    
    # 创建翻译模板实例
    translation_prompt = TranslationPromptTemplate()
    
    # 格式化提示模板（只需提供必要参数，其他参数会自动处理）
    prompt = translation_prompt.format(
        text="人工智能正在重塑我们的生活方式。",
        target_language="英语",
        formality_level="formal"
    )
    
    # 调用LLM生成翻译
    response = llm.invoke(prompt)
    return response.content

# 3. 结构化提示模板示例
def structured_prompt_template_example():
    """结构化提示模板示例"""
    llm = get_llm()
    
    # 创建结构化提示模板
    # 定义产品分析的结构化输入模式，包含多层嵌套结构
    product_analysis = StructuredPrompt(
        name="product_analysis",
        template="""请对以下产品进行全面分析：

产品基本信息：
- 产品名称：{product_info[name]}
- 产品代号：{product_info[code]}
- 产品版本：{product_info[version]}

市场分析：
1. 目标市场
   - 主要用户群：{market_analysis[target_market][user_groups]}
   - 市场规模：{market_analysis[target_market][market_size]}
   - 地域分布：{market_analysis[target_market][regions]}

2. 竞品分析
   - 主要竞品：{market_analysis[competitors][main_competitors]}
   - 竞争优势：{market_analysis[competitors][advantages]}
   - 竞争劣势：{market_analysis[competitors][disadvantages]}

3. 价格策略
   - 建议零售价：{market_analysis[pricing][retail_price]}
   - 成本结构：{market_analysis[pricing][cost_structure]}
   - 利润空间：{market_analysis[pricing][profit_margin]}

技术分析：
1. 功能特性
   - 核心功能：{tech_analysis[features][core_features]}
   - 创新点：{tech_analysis[features][innovations]}
   - 扩展性：{tech_analysis[features][scalability]}

2. 性能指标
   - 性能参数：{tech_analysis[performance][parameters]}
   - 稳定性：{tech_analysis[performance][stability]}
   - 兼容性：{tech_analysis[performance][compatibility]}

3. 技术架构
   - 系统架构：{tech_analysis[architecture][system]}
   - 关键技术：{tech_analysis[architecture][key_technologies]}
   - 技术难点：{tech_analysis[architecture][challenges]}

请基于以上信息提供：
1. 产品定位分析
2. 市场机会评估
3. 技术可行性分析
4. 投资风险评估
5. 发展建议""",
        input_schema={
            "product_info": {
                "name": str,
                "code": str,
                "version": str
            },
            "market_analysis": {
                "target_market": {
                    "user_groups": str,
                    "market_size": str,
                    "regions": str
                },
                "competitors": {
                    "main_competitors": str,
                    "advantages": str,
                    "disadvantages": str
                },
                "pricing": {
                    "retail_price": str,
                    "cost_structure": str,
                    "profit_margin": str
                }
            },
            "tech_analysis": {
                "features": {
                    "core_features": str,
                    "innovations": str,
                    "scalability": str
                },
                "performance": {
                    "parameters": str,
                    "stability": str,
                    "compatibility": str
                },
                "architecture": {
                    "system": str,
                    "key_technologies": str,
                    "challenges": str
                }
            }
        }
    )
    
    # 使用结构化提示模板生成分析
    response = llm.invoke(
        product_analysis.format(
            product_info={
                "name": "智能手表Pro Max",
                "code": "SW-2024",
                "version": "2.0"
            },
            market_analysis={
                "target_market": {
                    "user_groups": "高端商务人士、运动爱好者、健康管理人群",
                    "market_size": "预计2024年达到500亿元",
                    "regions": "一线及新一线城市为主"
                },
                "competitors": {
                    "main_competitors": "Apple Watch、华为Watch、小米手环",
                    "advantages": "更长续航时间、更全面的健康监测、更优性价比",
                    "disadvantages": "品牌知名度较低、生态系统不够完善"
                },
                "pricing": {
                    "retail_price": "1999-2999元",
                    "cost_structure": "硬件成本65%、研发成本20%、营销成本15%",
                    "profit_margin": "毛利率35%-40%"
                }
            },
            tech_analysis={
                "features": {
                    "core_features": "心率监测、血氧检测、运动追踪、睡眠分析",
                    "innovations": "AI健康助手、智能运动规划、情绪管理",
                    "scalability": "支持第三方应用开发、可扩展传感器模块"
                },
                "performance": {
                    "parameters": "续航15天、防水50米、心率误差<3%",
                    "stability": "MTBF>5000小时、系统稳定性99.9%",
                    "compatibility": "支持iOS/Android、蓝牙5.0、NFC"
                },
                "architecture": {
                    "system": "自研RTOS系统、双核处理器、独立AI协处理器",
                    "key_technologies": "低功耗蓝牙、AI算法、传感器融合",
                    "challenges": "续航与功能平衡、数据精度提升、散热优化"
                }
            }
        )
    )
    return response.content

# 4. PipelinePromptTemplate - 管道模板
def pipeline_prompt_template_example():
    """管道提示模板示例"""
    llm = get_llm()
    
    # 创建大纲生成模板
    # 第一阶段：根据主题和要求生成教学大纲
    outline_prompt = PromptTemplate(
        input_variables=["topic", "education_level", "learning_objectives"],
        template="""请为以下教育内容创建详细大纲：

主题：{topic}
教育水平：{education_level}
学习目标：{learning_objectives}

大纲要求：
1. 符合教育水平的认知能力
2. 循序渐进的知识结构
3. 包含互动和实践环节"""
    )
    
    # 创建内容编写模板
    # 第二阶段：基于大纲生成详细内容
    content_prompt = PromptTemplate(
        input_variables=["outline", "teaching_style", "examples"],
        template="""基于以下大纲，创建详细的教学内容：

大纲内容：
{outline}

教学风格：{teaching_style}
实例要求：{examples}"""
    )
    
    # 创建练习题生成模板
    # 第三阶段：根据内容生成练习题
    exercise_prompt = PromptTemplate(
        input_variables=["content", "difficulty_level", "question_types"],
        template="""基于以下教学内容，生成练习题：

课程内容：
{content}

难度等级：{difficulty_level}
题型要求：{question_types}"""
    )
    
    # 创建管道模板
    # 将三个模板串联起来，形成完整的教学内容生成流程
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=exercise_prompt,
        pipeline_prompts=[
            ("outline", outline_prompt),
            ("content", content_prompt)
        ]
    )
    
    # 第一步：生成大纲
    outline_result = llm.invoke(outline_prompt.format(
        topic="Python基础编程：循环结构",
        education_level="高中信息技术",
        learning_objectives="理解和掌握Python中的for和while循环使用"
    ))
    
    # 第二步：基于大纲生成内容
    content_result = llm.invoke(content_prompt.format(
        outline=outline_result.content,
        teaching_style="互动式，以实例为导向",
        examples="日常生活中的循环案例"
    ))
    
    # 第三步：生成练习题
    final_result = llm.invoke(exercise_prompt.format(
        content=content_result.content,
        difficulty_level="中等",
        question_types="选择题、编程题、应用题"
    ))
    
    return {
        "outline": outline_result.content,
        "content": content_result.content,
        "exercises": final_result.content
    }

# 5. FewShotPromptTemplate - 少样本模板
def few_shot_prompt_template_example():
    """少样本提示模板示例
    
    FewShotPromptTemplate是一种基于示例学习的模板，通过提供少量示例来指导模型理解任务。
    这种方法特别适合需要通过具体例子来说明任务要求的场景。
    
    主要特点：
    1. 支持示例学习，通过具体例子引导模型
    2. 可以设置示例的展示格式
    3. 适合分类、情感分析等需要参考样本的任务
    """
    llm = get_llm()
    
    # 创建示例
    # 提供了三个不同情感倾向的文本样本
    examples = [
        {"text": "这个产品质量很好，很耐用", "sentiment": "正面"},
        {"text": "价格太贵了，不太划算", "sentiment": "负面"},
        {"text": "一般般，没什么特别的", "sentiment": "中性"}
    ]
    
    # 创建示例模板
    # 定义了如何展示每个示例
    example_prompt = PromptTemplate(
        input_variables=["text", "sentiment"],
        template="文本: {text}\n情感: {sentiment}"
    )
    
    # 创建少样本模板
    # 组合示例和模板，构建完整的少样本学习结构
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="以下是一些文本情感分类的例子：\n\n",
        suffix="\n\n文本: {input_text}\n情感:",
        input_variables=["input_text"]
    )
    
    # 格式化提示
    prompt = few_shot_prompt.format(
        input_text="这个商品的包装很精美，但是发货太慢了"
    )
    
    # 调用LLM进行情感分析
    response = llm.invoke(prompt)
    return response.content

# 6. ChatPromptTemplate - 对话模板
def chat_prompt_template_example():
    """对话提示模板示例
    
    ChatPromptTemplate专门用于构建对话系统，它可以组合多个角色的消息，
    创建自然流畅的对话交互。
    
    主要特点：
    1. 支持多角色对话
    2. 可以定制对话风格和语气
    3. 适合客服、咨询等对话场景
    """
    llm = get_llm()
    
    # 创建消息模板
    # 分别定义系统、用户和AI的消息模板
    system_template = "你是一个专业的{role}，专门解答关于{product}的问题。请使用{tone}的语气。"
    human_template = "{question}"
    ai_template = "我理解您的问题是关于{product}的{question_type}。让我为您详细解答。"
    
    # 创建消息提示模板
    # 将不同角色的模板转换为对应的消息类型
    system_message = SystemMessagePromptTemplate.from_template(system_template)
    human_message = HumanMessagePromptTemplate.from_template(human_template)
    ai_message = AIMessagePromptTemplate.from_template(ai_template)
    
    # 创建对话模板
    # 按照对话流程组合各个消息
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message,
        human_message,
        ai_message
    ])
    
    # 格式化消息
    # 填充模板变量，生成完整的对话内容
    messages = chat_prompt.format_messages(
        role="技术支持专家",
        product="智能手机",
        tone="专业友好",
        question="如何解决电池续航问题？",
        question_type="技术支持"
    )
    
    # 调用LLM生成回答
    response = llm.invoke(messages)
    return response.content

# 7. ImagePromptTemplate - 图像提示模板
def image_prompt_template_example():
    """图像提示模板示例
    
    ImagePromptTemplate用于处理图像相关的任务，它可以结合图像URL和其他参数，
    构建适合图像分析、描述等任务的提示。
    
    主要特点：
    1. 支持图像URL和详细度参数
    2. 可以与文本提示结合
    3. 适合图像分析、描述生成等任务
    """
    llm = get_llm()
    
    # 创建图像处理模板
    # 定义图像URL和详细度参数
    image_prompt = ImagePromptTemplate(
        template={"url": "{url}", "detail": "{detail}"},
        template_format="f-string"
    )
    
    # 格式化提示
    # 设置图像URL和期望的分析详细度
    prompt = image_prompt.format(
        url="https://example.com/sunset.jpg",
        detail="high"
    )
    
    # 创建分析提示
    # 将图像数据与分析要求结合
    analysis_prompt = PromptTemplate(
        input_variables=["image_data"],
        template="请分析这张图片的内容和风格特点：\n{image_data}"
    )
    
    # 调用LLM生成图像分析
    response = llm.invoke(analysis_prompt.format(image_data=str(prompt)))
    return response.content

if __name__ == "__main__":
    print("\n1. 基础提示模板示例：")
    print(basic_prompt_template_example())
    
    print("\n2. 自定义字符串提示模板示例：")
    print(string_prompt_template_example())
    
    print("\n3. 结构化提示模板示例：")
    print(structured_prompt_template_example())
    
    print("\n4. 管道提示模板示例：")
    result = pipeline_prompt_template_example()
    print("\n大纲：")
    print(result["outline"])
    print("\n内容：")
    print(result["content"])
    print("\n练习题：")
    print(result["exercises"])
    
    print("\n5. 少样本提示模板示例：")
    print(few_shot_prompt_template_example())
    
    print("\n6. 对话提示模板示例：")
    print(chat_prompt_template_example())
    
    print("\n7. 图像提示模板示例：")
    print(image_prompt_template_example())