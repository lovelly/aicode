from langchain.prompts import (
    PromptTemplate,
    StringPromptTemplate,
    PipelinePromptTemplate,
    FewShotPromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.prompts.image import ImagePromptTemplate

from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict, Any

# 1. PromptTemplate - 基础模板
# 核心原理：PromptTemplate是LangChain中最基础的模板类型，它通过变量插值机制实现动态提示词生成
# 工作机制：
# 1. 定义模板字符串，使用{variable}语法标记变量位置
# 2. 指定input_variables列表，声明所需的变量名
# 3. 调用format方法时，将实际值替换到对应的变量位置
# 4. 内部使用Python的str.format()方法实现字符串格式化
def basic_prompt_template_example():
    """
    PromptTemplate是最基础的模板类型，用于创建简单的提示模板。
    
    主要特点：
    - 支持变量插值
    - 简单直观
    - 易于使用
    
    示例说明：
    以下示例展示了如何使用PromptTemplate创建一个简单的产品介绍生成器。
    """
    # 创建一个简单的产品介绍模板
    # 实现原理：通过定义模板字符串和变量列表，创建一个可重用的提示模板
    # 变量插值过程：format方法将实际参数映射到模板中的占位符
    product_prompt = PromptTemplate(
        input_variables=["product_name", "features", "price"],
        template="""请为以下产品生成一个简短的介绍：

产品名称：{product_name}
产品特点：{features}
价格：{price}

请用3-5句话介绍这个产品的主要优势和价值。"""
    )
    
    # 使用示例
    # 实际应用：传入具体参数，生成完整的提示文本
    description = product_prompt.format(
        product_name="智能手表 Mini",
        features="防水、心率监测、运动追踪",
        price="599元"
    )
    return description


# 2. StringPromptTemplate - 字符串提示模板
def string_prompt_template_example():
    """
    StringPromptTemplate示例：展示自定义字符串模板的高级特性
    """
    # 创建一个继承自StringPromptTemplate的自定义模板类
    # 实现原理：通过重写format方法实现变量自动注入和自定义处理
    class TranslationPromptTemplate(StringPromptTemplate):
        input_variables: List[str] = ["text", "target_language", "formality_level"]
        template: str
    
        def __init__(self, template: str, input_variables: List[str]):
            super().__init__(template=template, input_variables=input_variables)
            self.template = template
    
        def format(self, **kwargs) -> str:
            # 自动注入系统时间
            from datetime import datetime
            kwargs["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 自动检测源语言（这里简化处理）
            if "source_language" not in kwargs:
                kwargs["source_language"] = "自动检测"
            
            # 设置默认的语言风格
            kwargs["formality_level"] = kwargs.get("formality_level", "standard")
            
            # 使用template进行字符串格式化
            return self.template.format(**kwargs)
    
    # 创建翻译模板实例
    translation_prompt = TranslationPromptTemplate(
        input_variables=["text", "target_language"],  # 只需要指定必要的变量
        template="""
        [系统时间: {timestamp}]

        请将以下文本从{source_language}翻译成{target_language}：

        原文：{text}

        翻译要求：
        - 语言风格：{formality_level}
        - 保持原文的语气和风格
        - 确保专业术语的准确性
        - 适应目标语言的文化背景

        翻译结果：
        [在此处提供翻译]

        注意事项：
        - 如遇专业术语，请提供标准翻译
        - 如有文化差异，请适当调整表达方式
        - 确保符合目标语言的表达习惯"""
    )
    
    # 使用示例 - 只需提供必要参数，其他参数由模板自动处理
    translation = translation_prompt.format(
        text="人工智能正在重塑我们的生活方式。",
        target_language="英语",
        formality_level = "formal"
    )
    
    # 返回翻译结果
    return translation

# 3. StructuredPromptTemplate - 结构化提示模板
# 核心原理：将提示信息组织为结构化的消息列表，支持角色定义和多轮对话
# 工作机制：
# 1. 使用StructuredPrompt定义输入模式和验证规则
# 2. 通过自定义数据转换实现复杂的数据处理
# 3. 支持结构化的输入验证和格式化
def structured_prompt_template_example():
    """
    结构化提示模板示例：展示StructuredPrompt的高级特性
    """
    # 创建结构化提示模板实例
    # 实现原理：通过定义输入模式和验证规则，实现结构化的数据处理
    # 创建系统消息模板
    system_template = SystemMessagePromptTemplate.from_template("你是一个专业的市场分析师，请根据提供的产品信息进行分析。")
    
    # 创建人类消息模板
    human_template = HumanMessagePromptTemplate.from_template("""
    请对以下产品进行全面的市场分析：
        产品信息：
            名称：{product_info[name]}
            描述：{product_info[description]}
            价格：{product_info[price]}元
            特点：{product_info[features]}
        市场分析：
            目标用户：{market_analysis[target_users]}
            竞品分析：{market_analysis[competitors]}
            市场规模：{market_analysis[market_size]}
    请提供详细的SWOT分析报告。""")
    
    # 创建结构化提示模板实例
    analysis_prompt = StructuredPrompt(
        messages=[system_template, human_template],
        schema={
            "type": "object",
            "properties": {
                "product_info": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "price": {"type": "number", "minimum": 0},
                        "features": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["name", "description", "price"]
                },
                "market_analysis": {
                    "type": "object",
                    "properties": {
                        "target_users": {"type": "string"},
                        "competitors": {"type": "array", "items": {"type": "string"}},
                        "market_size": {"type": "string"}
                    },
                    "required": ["target_users"]
                }
            },
            "required": ["product_info"]
        }
    )
    
    # 使用示例
    # 参数映射：传入结构化的产品和市场信息
    product_info = {
        "name": "智能手表Pro",
        "description": "高性能智能运动手表",
        "price": 1499,
        "features": ["心率监测", "运动追踪", "睡眠分析", "7天续航"]
    }
    
    market_analysis = {
        "target_users": "注重健康的年轻白领和运动爱好者",
        "competitors": ["Apple Watch", "华为手表", "小米手环"],
        "market_size": "预计2024年达到1000亿元"
    }
    
    analysis = analysis_prompt.format(
        product_info=product_info,
        market_analysis=market_analysis
    )
    return analysis

# 4. PipelinePromptTemplate - 管道模板
# 核心原理：实现多个提示模板的串联处理，支持复杂的提示生成流程
# 工作机制：
# 1. 定义多个独立的提示模板，每个模板负责特定的处理阶段
# 2. 通过pipeline_prompts参数指定模板之间的连接关系
# 3. 前一个模板的输出作为后一个模板的输入
# 4. 最终模板（final_prompt）生成最终的提示文本
def pipeline_prompt_template_example():
    """
    PipelinePromptTemplate用于创建多阶段的提示处理流程。
    
    主要特点：
    - 支持多个模板串联
    - 前一个模板的输出可以作为下一个模板的输入
    - 适合复杂的处理流程
    
    使用场景：
    - 多步骤的文本处理
    - 复杂的内容生成
    - 数据转换和处理流程
    - 多阶段的分析任务
    
    示例说明：
    以下示例展示了如何创建一个教育内容生成的管道，包括大纲生成、
    内容编写和练习题生成三个阶段。
    """
    # 创建大纲生成模板
    # 第一阶段：定义课程大纲结构
    outline_prompt = PromptTemplate(
        input_variables=["topic", "education_level", "learning_objectives"],
        template="""请为以下教育内容创建详细大纲：

主题：{topic}
教育水平：{education_level}
学习目标：{learning_objectives}

大纲要求：
1. 符合教育水平的认知能力
2. 循序渐进的知识结构
3. 包含互动和实践环节

请生成：
1. 课程概述
2. 知识点分解
3. 教学重点难点
4. 课程活动设计"""
    )
    
    # 创建内容编写模板
    # 第二阶段：基于大纲生成详细内容
    content_prompt = PromptTemplate(
        input_variables=["outline", "teaching_style", "examples"],
        template="""基于以下大纲，创建详细的教学内容：

大纲内容：
{outline}

教学风格：{teaching_style}
实例要求：{examples}

请按照以下结构编写：
1. 导入环节（引起兴趣）
2. 主要内容（知识讲解）
3. 互动环节（案例分析）
4. 总结回顾"""
    )
    
    # 创建练习题生成模板
    # 第三阶段：生成配套练习题
    exercise_prompt = PromptTemplate(
        input_variables=["content", "difficulty_level", "question_types"],
        template="""基于以下教学内容，生成练习题：

课程内容：
{content}

难度等级：{difficulty_level}
题型要求：{question_types}

请生成：
1. 基础巩固题
2. 应用实践题
3. 思考拓展题"""
    )
    
    # 创建管道模板
    # 实现原理：通过pipeline_prompts参数定义模板之间的数据流
    # 数据流转：outline_prompt -> content_prompt -> exercise_prompt
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=exercise_prompt,
        pipeline_prompts=[
            ("outline", outline_prompt),
            ("content", content_prompt)
        ]
    )
    
    # 使用示例
    # 参数传递：所有需要的参数都在这里指定，由管道自动处理数据流转
    final_prompt = pipeline_prompt.format(
        topic="Python基础编程：循环结构",
        education_level="高中信息技术",
        learning_objectives="理解和掌握Python中的for和while循环使用",
        teaching_style="互动式，以实例为导向",
        examples="日常生活中的循环案例",
        difficulty_level="中等",
        question_types="选择题、编程题、应用题"
    )

    return final_prompt

# 5. FewShotPromptTemplate - 少样本模板
# 核心原理：通过提供少量示例来指导模型理解任务模式
# 工作机制：
# 1. 定义示例列表，每个示例包含输入和期望输出
# 2. 创建示例格式模板，规定示例的展示方式
# 3. 设置prefix和suffix，构建完整的提示结构
# 4. 在运行时，将示例和待处理数据组合成最终提示
def few_shot_prompt_template_example():
    """
    FewShotPromptTemplate用于基于少量示例进行提示。
    
    主要特点：
    - 支持示例学习
    - 可以动态调整示例数
    - 适合需要参考案例的场景
    
    使用场景：
    - 文本分类任务
    - 风格迁移
    - 特定格式的内容生成
    - 模式识别和复制
    
    示例说明：
    以下示例展示了如何创建一个文本情感分析的少样本学习模板，
    通过提供少量带标注的示例，帮助模型更好地理解任务要求。
    """
    # 创建一个文本分类的示例模板
    # 示例设计：选择具有代表性的样本，覆盖不同情感类别
    examples = [
        {"text": "这个产品质量很好，很耐用", "sentiment": "正面"},
        {"text": "价格太贵了，不太划算", "sentiment": "负面"},
        {"text": "一般般，没什么特别的", "sentiment": "中性"}
    ]
    
    # 定义示例格式
    # 格式设计
    example_prompt = PromptTemplate(
        input_variables=["text", "sentiment"],
        template="文本: {text}\n情感: {sentiment}"
    )
    
    # 实现原理：通过组合示例和模板，构建完整的少样本学习提示
    # 数据流设计：examples -> example_prompt -> prefix/suffix -> final prompt
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="以下是一些文本情感分类的例子：\n\n",
        suffix="\n\n文本: {input_text}\n情感:",
        input_variables=["input_text"]
    )
    
    # 使用示例
    # 实际应用：将新的输入文本与示例结合，生成完整的提示
    classification = few_shot_prompt.format(
        input_text="这个商品的包装很精美，但是发货太慢了"
    )
    return classification

# 6. ChatPromptTemplate - 对话模板
# 核心原理：构建结构化的对话流程，支持多角色交互和上下文管理
# 工作机制：
# 1. 定义不同角色的消息模板（系统、用户、AI）
# 2. 通过模板组合构建完整的对话流程
# 3. 支持变量插值和角色切换
# 4. 维护对话的上下文连续性
def chat_prompt_template_example():
    """
    ChatPromptTemplate用于创建对话式的提示模板。
    
    主要特点：
    - 支持多轮对话
    - 可以设定不同的角色
    - 适合对话场景
    
    使用场景：
    - 聊天机器人
    - 客服对话
    - 角色扮演
    - 教育对话
    
    示例说明：
    以下示例展示了如何创建一个专业的客服对话模板，包含系统指令、
    用户输入和AI响应三个部分，可以处理产品咨询、技术支持等场景。
    """
    # 创建一个客服对话模板
    # 实现原理：通过定义不同角色的消息模板，构建完整的对话流程
    # 角色设计：系统指令设定基础行为，用户输入触发交互，AI响应提供服务
    system_template = "你是一个专业的{role}，专门解答关于{product}的问题。请使用{tone}的语气。"
    human_template = "{question}"
    ai_template = "我理解您的问题是关于{product}的{question_type}。让我为您详细解答：\n{answer}"
    
    # 消息模板实例化
    # 模板组装：将不同角色的模板转换为对应的消息类型
    system_message = SystemMessagePromptTemplate.from_template(system_template)
    human_message = HumanMessagePromptTemplate.from_template(human_template)
    ai_message = AIMessagePromptTemplate.from_template(ai_template)
    
    # 创建对话模板
    # 流程设计：按照系统指令->用户问题->AI回答的顺序组织对话流程
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message,
        human_message,
        ai_message
    ])
    
    # 使用示例
    # 参数映射：将具体的角色、产品和问题信息注入到对话模板中
    messages = chat_prompt.format_messages(
        role="技术支持专家",
        product="智能手机",
        tone="专业友好",
        question="如何解决电池续航问题？",
        question_type="技术支持",
        answer="以下是几个有效的解决方案：\n1. 检查耗电应用\n2. 开启省电模式\n3. 调整屏幕亮度\n4. 关闭不必要的后台进程\n5. 定期进行系统更新"
    )
    
    # 格式化输出消息内容
    formatted_messages = "\n"
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        formatted_messages += f"\n消息 {i+1} ({msg_type}):\n{'-'*50}\n{msg.content}\n{'-'*50}\n"
    
    # 返回消息列表和格式化后的内容
    return formatted_messages

# 7. ImagePromptTemplate - 图像提示模板
# 核心原理：扩展基础模板以支持图像处理任务，实现多模态交互
# 工作机制：
# 1. 支持图像URL和细节信息的模板变量
# 2. 通过template_format指定格式化方式
# 3. 可以结合其他模板类型处理复杂的多模态任务
def image_prompt_template_example():
    """
    ImagePromptTemplate用于处理多模态模型的图像提示。
    
    主要特点：
    - 支持图像URL处理
    - 可以包含图像细节信息
    - 适合多模态模型任务
    
    使用场景：
    - 图像分析
    - 图像问答
    - 视觉内容理解
    - 多模态交互
    
    示例说明：
    以下示例展示了如何创建一个图像处理提示模板，通过指定图像URL
    和细节信息，帮助多模态模型理解和处理图像。
    """
    # 创建一个图像处理提示模板
    image_prompt = ImagePromptTemplate(
        template={"url": "https://example.com/image.jpg"},
        template_format="f-string"
    )
    
    # 使用示例 - 基本URL
    image_prompt_basic = image_prompt.format(
        url="https://example.com/sunset.jpg"
    )
    
    # 使用示例 - 带细节信息
    image_prompt_detailed = image_prompt.format(
        url="https://example.com/product.jpg",
        detail="high"
    )
    
    return f"基本图像URL示例:\n{image_prompt_basic}\n\n带细节信息的图像URL示例:\n{image_prompt_detailed}"

if __name__ == "__main__":
    # 测试所有提示模板示例
    print("\n1. 基础提示模板示例：")
    print(basic_prompt_template_example())
    
    print("\n2. 自定义字符串提示模板示例：")
    print(string_prompt_template_example())
    
    print("\n3. 结构化提示模板示例：")
    print(structured_prompt_template_example())
    
    print("\n4. 管道提示模板示例：")
    print(pipeline_prompt_template_example())
    
    print("\n5. 少样本提示模板示例：")
    print(few_shot_prompt_template_example())
    
    print("\n6. 对话提示模板示例：")
    print(chat_prompt_template_example())
    
    print("\n7. 图像提示模板示例：")
    print(image_prompt_template_example())

