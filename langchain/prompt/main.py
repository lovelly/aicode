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
    product_prompt = PromptTemplate(
        input_variables=["product_name", "features", "price"],
        template="""请为以下产品生成一个简短的介绍：

产品名称：{product_name}
产品特点：{features}
价格：{price}

请用3-5句话介绍这个产品的主要优势和价值。"""
    )
    
    # 使用示例
    description = product_prompt.format(
        product_name="智能手表 Mini",
        features="防水、心率监测、运动追踪",
        price="599元"
    )
    return description


def string_prompt_template_example():
    """
    StringPromptTemplate是一个抽象基类，用于创建自定义的提示模板。
    
    主要特点：
    - 支持自定义字符串格式化逻辑
    - 可以添加额外的处理步骤
    - 适合复杂的模板需求
    - 支持输入验证和错误处理
    
    使用场景：
    - 需要自定义格式化逻辑的场合
    - 复杂的模板处理需求
    - 多语言支持
    - 特殊格式的输出需求
    
    示例说明：
    以下示例展示了如何创建一个自定义的翻译模板，包含输入验证和错误处理。
    """
    class CustomTranslationPrompt(StringPromptTemplate):
        template: str
        input_variables: list[str]
        
        def format(self, **kwargs) -> str:
            # 输入验证
            required_vars = ["text", "target_language", "formality_level"]
            for var in required_vars:
                if var not in kwargs:
                    raise ValueError(f"Missing required variable: {var}")
                    
            # 格式化处理
            if kwargs["formality_level"] not in ["formal", "casual", "neutral"]:
                raise ValueError("formality_level must be one of: formal, casual, neutral")
                
            # 添加额外的处理逻辑
            kwargs["source_language"] = "自动检测"
            kwargs["timestamp"] = "2024-03-15"
            
            return self.template.format(**kwargs)
    
    # 创建一个自定义的翻译模板
    translation_prompt = CustomTranslationPrompt(
        template="""[系统时间: {timestamp}]

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
- 确保符合目标语言的表达习惯""",
        input_variables=["text", "target_language", "formality_level"]
    )
    
    try:
        # 使用示例 - 正确用法
        translation = translation_prompt.format(
            text="人工智能正在重塑我们的生活方式。",
            target_language="英语",
            formality_level="formal"
        )
        
        # 使用示例 - 错误用法（用于演示错误处理）
        error_translation = translation_prompt.format(
            text="测试文本",
            target_language="英语",
            formality_level="invalid_level"  # 这将触发验证错误
        )
    except ValueError as e:
        translation = f"错误：{str(e)}"
    
    return translation

# 3. StructuredPrompt - 结构化模板
def structured_prompt_template_example():
    """
    StructuredPrompt用于创建具有固定输出结构的提示模板，它继承自ChatPromptTemplate。
    
    主要特点：
    - 支持通过Pydantic模型定义输出结构
    - 确保输出格式符合预定义的schema
    - 适合需要结构化输出的场景
    - 支持对话式提示模板
    
    使用场景：
    - API响应生成
    - 结构化数据提取
    - 标准化数据处理
    - 对话式数据分析
    
    示例说明：
    以下示例展示了如何使用StructuredPrompt创建一个产品分析助手，
    通过定义输出schema和对话模板，生成结构化的产品分析结果。
    """
    from pydantic import BaseModel, Field
    
    # 定义输出结构的schema
    class ProductAnalysis(BaseModel):
        product_name: str = Field(description="产品名称")
        target_market: str = Field(description="目标市场定位")
        strengths: list[str] = Field(description="产品优势，最多3点")
        weaknesses: list[str] = Field(description="产品劣势，最多3点")
        opportunities: list[str] = Field(description="市场机会，最多3点")
        threats: list[str] = Field(description="潜在威胁，最多3点")
        recommendations: list[str] = Field(description="改进建议，最多3点")
    
    # 创建对话式提示模板
    messages = [
        ("system", "你是一个专业的产品分析师，擅长进行SWOT分析。请根据用户提供的产品信息，生成结构化的分析报告。"),
        ("human", "请分析这个产品：{product_description}\n价格区间：{price_range}\n目标用户：{target_users}"),
        ("ai", "我会基于您提供的信息进行全面分析。")
    ]
    
    # 创建结构化提示模板
    analysis_prompt = StructuredPrompt(
        messages=messages,
        schema_=ProductAnalysis,
        structured_output_kwargs={
            "name": "analyze_product",
            "description": "分析产品并生成结构化的SWOT分析报告"
        }
    )
    
    # 使用示例
    analysis = analysis_prompt.format(
        product_description="智能手表Pro，支持心率监测、运动追踪、睡眠分析，续航时间7天",
        price_range="1299-1599元",
        target_users="注重健康的年轻白领和运动爱好者"
    )
    return analysis

# 4. PipelinePromptTemplate - 管道模板
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
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=exercise_prompt,
        pipeline_prompts=[
            ("outline", outline_prompt),
            ("content", content_prompt)
        ]
    )
    
    # 使用示例
    try:
        final_prompt = pipeline_prompt.format(
            topic="Python基础编程：循环结构",
            education_level="高中信息技术",
            learning_objectives="理解和掌握Python中的for和while循环使用",
            teaching_style="互动式，以实例为导向",
            examples="日常生活中的循环案例",
            difficulty_level="中等",
            question_types="选择题、编程题、应用题"
        )
    except Exception as e:
        final_prompt = f"错误：{str(e)}"
    
    return final_prompt

# 5. FewShotPromptTemplate - 少样本模板
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
    examples = [
        {"text": "这个产品质量很好，很耐用", "sentiment": "正面"},
        {"text": "价格太贵了，不太划算", "sentiment": "负面"},
        {"text": "一般般，没什么特别的", "sentiment": "中性"}
    ]
    
    example_prompt = PromptTemplate(
        input_variables=["text", "sentiment"],
        template="文本: {text}\n情感: {sentiment}"
    )
    
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="以下是一些文本情感分类的例子：\n\n",
        suffix="\n\n文本: {input_text}\n情感:",
        input_variables=["input_text"]
    )
    
    # 使用示例
    classification = few_shot_prompt.format(
        input_text="这个商品的包装很精美，但是发货太慢了"
    )
    return classification

# 6. ChatPromptTemplate - 对话模板
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
    system_template = "你是一个专业的{role}，专门解答关于{product}的问题。请使用{tone}的语气。"
    human_template = "{question}"
    ai_template = "我理解您的问题是关于{product}的{question_type}。让我为您详细解答：\n{answer}"
    
    system_message = SystemMessagePromptTemplate.from_template(system_template)
    human_message = HumanMessagePromptTemplate.from_template(human_template)
    ai_message = AIMessagePromptTemplate.from_template(ai_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message,
        human_message,
        ai_message
    ])
    
    # 使用示例
    messages = chat_prompt.format_messages(
        role="技术支持专家",
        product="智能手机",
        tone="专业友好",
        question="如何解决电池续航问题？",
        question_type="技术支持",
        answer="以下是几个有效的解决方案：\n1. 检查耗电应用\n2. 开启省电模式\n3. 调整屏幕亮度\n4. 关闭不必要的后台进程\n5. 定期进行系统更新"
    )
    return messages

# 7. ImagePromptTemplate - 图像提示模板
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

