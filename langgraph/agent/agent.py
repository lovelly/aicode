from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
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
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import OutputParserException
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import re
from collections import Counter
import jieba
import requests
import pandas as pd
import matplotlib.pyplot as plt
import io
import subprocess

# 加载环境变量
load_dotenv()
print(os.getenv("MODEL"))

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
    """智能体状态模型，用于维护系统运行状态。"""
    messages: List[Any]  # 对话历史
    task_status: str     # 任务状态
    task_plan: Dict      # 任务规划
    execution_context: Dict  # 执行上下文
    results: Dict        # 执行结果
    memory: Dict         # 记忆状态
    context: Dict        # 上下文信息

def get_llm()->ChatOpenAI:
    llm = ChatOpenAI(
        model=os.getenv("MODEL", "deepseek-chat"),
        temperature=0.7,
        callbacks=[CustomCallbackHandler()]
    )
    return llm

# 2. 工具定义
@tools.tool
def browse_web(url: str) -> Dict:
    """浏览网页并提取旅游相关信息"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else "无标题"
        paragraphs = soup.find_all('p')
        content = "\n".join([p.get_text().strip() for p in paragraphs[:15]])
        
        # 提取旅游相关信息
        travel_keywords = ["旅游", "景点", "酒店", "美食", "交通", "门票", "攻略", "行程", "住宿"]
        travel_paragraphs = []
        for p in paragraphs:
            text = p.get_text().strip()
            if any(keyword in text for keyword in travel_keywords):
                travel_paragraphs.append(text)
        
        # 提取链接
        links = []
        for link in soup.find_all('a', href=True)[:8]:
            link_text = link.get_text().strip() or "无文本"
            if any(keyword in link_text for keyword in travel_keywords):
                links.append({"text": link_text, "url": link['href']})
        
        return {
            "title": title,
            "content": content[:1500] + ("..." if len(content) > 1500 else ""),
            "travel_content": "\n".join(travel_paragraphs[:10]),
            "links": links,
            "status": "success"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "url": url
        }

@tools.tool
def analyze_data(data: Dict, analysis_type: str = "basic") -> Dict:
    """分析数据工具：对网页内容进行多维度分析"""
    results = {}
    
    # 数据验证
    if not isinstance(data, dict):
        return {"status": "error", "error": "输入数据必须是字典格式"}

    if 'content' not in data:
        return {"status": "error", "error": "缺少content字段"}

    if not isinstance(data['content'], str):
        return {"status": "error", "error": "content字段必须是字符串类型"}

    if not data['content'].strip():
        return {"status": "error", "error": "content字段不能为空"}

    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(data['content'], 'html.parser')
    
    # 提取纯文本内容
    text_content = soup.get_text(separator='\n', strip=True)
    
    # 基础分析
    if analysis_type in ["basic", "all"]:
        paragraphs = [p for p in text_content.split('\n') if p.strip()]
        words = list(jieba.cut(text_content))
        results["basic"] = {
            "total_chars": len(text_content),
            "total_words": len(words),
            "paragraphs": len(paragraphs),
            "links": len(soup.find_all('a')),
            "images": len(soup.find_all('img'))
        }
    
    # 统计分析
    if analysis_type in ["stats", "all"]:
        # 词频统计（取前20个最常见的词）
        words = [w for w in jieba.cut(text_content) if len(w.strip()) > 1]
        word_freq = Counter(words).most_common(20)
        
        # 字符类型分布
        char_types = {
            "chinese": len(re.findall(r'[\u4e00-\u9fff]', text_content)),
            "english": len(re.findall(r'[a-zA-Z]', text_content)),
            "digit": len(re.findall(r'\d', text_content)),
            "punctuation": len(re.findall(r'[\.,!?;:"]', text_content))
        }
        
        results["stats"] = {
            "word_frequency": dict(word_freq),
            "character_distribution": char_types
        }
    
    # 提取关键信息
    if analysis_type in ["extract", "all"]:
        # 提取标题
        titles = {
            "h1": [h.get_text(strip=True) for h in soup.find_all('h1')],
            "h2": [h.get_text(strip=True) for h in soup.find_all('h2')],
            "h3": [h.get_text(strip=True) for h in soup.find_all('h3')]
        }
        
        # 提取链接
        links = [{
            "text": a.get_text(strip=True),
            "href": a.get('href')
        } for a in soup.find_all('a') if a.get('href')]
        
        # 提取图片
        images = [{
            "alt": img.get('alt', ''),
            "src": img.get('src')
        } for img in soup.find_all('img') if img.get('src')]
        
        results["extracted"] = {
            "titles": titles,
            "links": links[:10],  # 限制只返回前10个链接
            "images": images[:10]  # 限制只返回前10个图片
        }
    
    return {"status": "success", "analysis_result": results}

@tools.tool
def search_knowledge_base(query: str) -> Dict:
    """搜索知识库工具，用于检索已存储的知识内容"""
    # 模拟知识库检索
    results = [
        {
            "content": f"关于{query}的知识条目1",
            "source": "旅游百科",
            "timestamp": datetime.now().isoformat()
        },
        {
            "content": f"关于{query}的知识条目2",
            "source": "旅游攻略",
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    return {
        "query": query,
        "results": results,
        "total_results": len(results),
        "status": "success"
    }

@tools.tool
def summarize_text(text: str, max_length: int = 200) -> Dict:
    """对长文本进行摘要"""
    # 分割文本为句子
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
    
    if not sentences:
        return {"summary": "", "original_length": len(text), "summary_length": 0, "status": "error"}
    
    # 计算每个句子的重要性（这里简单地使用句子长度）
    importance = [len(s.split()) for s in sentences]
    
    # 选择最重要的句子（这里简单地选择最长的句子）
    sorted_sentences = [s for _, s in sorted(zip(importance, sentences), reverse=True)]
    
    # 生成摘要
    summary = ". ".join(sorted_sentences[:3]) + "."
    
    # 确保摘要不超过最大长度
    if len(summary) > max_length:
        summary = summary[:max_length-3] + "..."
    
    return {
        "summary": summary,
        "original_length": len(text),
        "summary_length": len(summary),
        "status": "success"
    }

@tools.tool
def get_weather(destination: str) -> Dict:
    """获取目的地的天气信息"""
    # 这里可以调用天气API获取天气信息
    # 模拟天气数据
    weather_data = {
        "北京": {"condition": "晴天", "temperature": "25°C", "humidity": "40%"},
        "上海": {"condition": "多云", "temperature": "28°C", "humidity": "65%"},
        "广州": {"condition": "小雨", "temperature": "30°C", "humidity": "80%"},
        "深圳": {"condition": "阵雨", "temperature": "29°C", "humidity": "75%"},
        "成都": {"condition": "阴天", "temperature": "22°C", "humidity": "60%"},
        "杭州": {"condition": "晴天", "temperature": "26°C", "humidity": "55%"},
        "西安": {"condition": "晴天", "temperature": "24°C", "humidity": "45%"},
        "重庆": {"condition": "多云", "temperature": "27°C", "humidity": "70%"},
        "厦门": {"condition": "晴天", "temperature": "29°C", "humidity": "60%"},
        "三亚": {"condition": "晴天", "temperature": "32°C", "humidity": "75%"}
    }
    
    # 获取天气数据，如果目的地不在预设数据中，返回默认数据
    weather = weather_data.get(destination, {"condition": "晴天", "temperature": "25°C", "humidity": "50%"})
    
    return {
        "destination": destination,
        "weather": weather["condition"],
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "forecast": [
            {"day": "今天", "condition": weather["condition"], "temperature": weather["temperature"]},
            {"day": "明天", "condition": "多云", "temperature": "26°C"},
            {"day": "后天", "condition": "晴天", "temperature": "27°C"}
        ],
        "status": "success"
    }

@tools.tool
def get_travel_recommendations(destination: str, interests: List[str], budget: str, duration: int) -> Dict:
    """获取旅行推荐信息，包括景点、住宿和活动"""
    # 调用大模型获取旅行推荐
    llm = get_llm()
    
    prompt = f"""
    请为目的地"{destination}"生成旅行推荐，包括景点、住宿、活动和美食。
    用户兴趣: {', '.join(interests)}
    预算级别: {budget}
    旅行天数: {duration}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # 假设大模型返回的内容是一个字典格式
    return {
        "destination": destination,
        "recommendations": response.content,
        "status": "success"
    }

@tools.tool
def calculate_travel_budget(destination: str, duration: int, accommodation_level: str, activities: List[str]) -> Dict:
    """计算旅行预算"""
    # 调用大模型计算预算
    llm = get_llm()
    
    prompt = f"""
    请为目的地"{destination}"计算旅行预算。
    旅行天数: {duration}
    住宿级别: {accommodation_level}
    活动: {', '.join(activities)}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # 假设大模型返回的内容是一个字典格式
    return {
        "destination": destination,
        "duration": duration,
        "accommodation_level": accommodation_level,
        "budget_details": response.content,
        "status": "success"
    }

# 3. 记忆和上下文管理
class MemoryManager:
    """记忆管理器：管理短期和长期记忆"""
    
    def __init__(self):
        self.short_term_memory = []
        self.long_term_memory = []
        self.importance_threshold = 0.7
    
    def add_to_short_term(self, memory_item: Dict) -> None:
        """添加到短期记忆"""
        if not isinstance(memory_item, dict):
            memory_item = {"content": str(memory_item)}
        
        if "timestamp" not in memory_item:
            memory_item["timestamp"] = datetime.now().isoformat()
        
        self.short_term_memory.append(memory_item)
        
        # 如果短期记忆超过容量，移除最旧的记忆
        if len(self.short_term_memory) > 50:
            self.short_term_memory.pop(0)
    
    def promote_to_long_term(self, index: int, importance: float) -> bool:
        """将短期记忆提升为长期记忆"""
        if importance < self.importance_threshold:
            return False
        
        if 0 <= index < len(self.short_term_memory):
            memory_item = self.short_term_memory[index].copy()
            memory_item["importance"] = importance
            self.long_term_memory.append(memory_item)
            return True
        
        return False
    
    def retrieve_relevant(self, query: str, limit: int = 5) -> List[Dict]:
        """检索相关记忆"""
        # 简单实现：基于关键词匹配
        relevant_memories = []
        
        # 搜索长期记忆
        for memory in self.long_term_memory:
            content = memory.get("content", "")
            if isinstance(content, str) and any(keyword in content for keyword in query.split()):
                relevant_memories.append(memory)
        
        # 搜索短期记忆
        for memory in reversed(self.short_term_memory):
            content = memory.get("content", "")
            if isinstance(content, str) and any(keyword in content for keyword in query.split()):
                relevant_memories.append(memory)
                
        # 去重并限制数量
        unique_memories = []
        memory_contents = set()
        
        for memory in relevant_memories:
            content = str(memory.get("content", ""))
            if content not in memory_contents:
                memory_contents.add(content)
                unique_memories.append(memory)
                if len(unique_memories) >= limit:
                    break
        
        return unique_memories
    
    def get_memory_summary(self) -> Dict:
        """获取记忆状态摘要"""
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": len(self.long_term_memory),
            "latest_short_term": self.short_term_memory[-1] if self.short_term_memory else None,
            "latest_long_term": self.long_term_memory[-1] if self.long_term_memory else None
        }

class ContextManager:
    """上下文管理器：管理任务执行上下文"""
    
    def __init__(self):
        self.context = {}
        self.context_history = []
    
    def update_context(self, key: str, value: Any) -> None:
        """更新上下文"""
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """获取上下文值"""
        return self.context.get(key, default)
    
    def save_context_snapshot(self) -> None:
        """保存上下文快照"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "context": self.context.copy()
        }
        self.context_history.append(snapshot)
        
        # 限制历史记录数量
        if len(self.context_history) > 20:
            self.context_history.pop(0)
    
    def get_context_history(self) -> List[Dict]:
        """获取上下文历史"""
        return self.context_history

# 4. 任务分解与规划
def decompose_task(task_description: str) -> List[Dict]:
    """将复杂任务分解为子任务"""
    # 根据任务描述动态生成子任务
    if any(keyword in task_description for keyword in ["旅行", "旅游", "行程", "旅行计划", "旅游计划", "出游", "度假"]):
        return [
            {"id": 1, "name": "确定旅行目的地", "tools": ["browse_web"], "description": "根据用户需求确定最佳旅行目的地"},
            {"id": 2, "name": "收集目的地信息", "tools": ["browse_web", "search_knowledge_base"], "description": "收集目的地的详细信息，包括景点、文化、美食等"},
            {"id": 3, "name": "制定行程安排", "tools": ["analyze_data"], "description": "根据目的地信息制定详细的日程安排"},
            {"id": 4, "name": "查找住宿选项", "tools": ["browse_web"], "description": "查找并推荐合适的酒店或住宿选择"},
            {"id": 5, "name": "规划交通方案", "tools": ["browse_web"], "description": "规划往返交通和当地交通方案"},
            {"id": 6, "name": "推荐特色体验", "tools": ["browse_web", "search_knowledge_base"], "description": "推荐目的地的特色体验和活动"},
            {"id": 7, "name": "获取天气信息", "tools": ["get_weather"], "description": "获取目的地的天气信息并提供穿着建议"},
            {"id": 8, "name": "预算规划", "tools": ["analyze_data"], "description": "估算旅行预算并提供省钱建议"},
            {"id": 9, "name": "生成完整旅行计划", "tools": ["summarize_text"], "description": "整合所有信息生成完整的旅行计划"}
        ]
    else:
        # 默认任务分解
        return [
            {"id": 1, "name": "信息收集", "tools": ["browse_web"], "description": "收集相关信息"},
            {"id": 2, "name": "数据分析", "tools": ["analyze_data"], "description": "分析收集的数据"},
            {"id": 3, "name": "生成报告", "tools": ["summarize_text"], "description": "生成最终报告"}
        ]

def plan_execution(subtasks: List[Dict]) -> Dict:
    """规划任务执行顺序"""
    return {
        "execution_plan": subtasks,
        "total_steps": len(subtasks),
        "estimated_time": len(subtasks) * 5,  # 每个子任务估计5分钟
        "created_at": datetime.now().isoformat()
    }

# 5. 工作流节点函数
def understand_request(state: AgentState) -> AgentState:
    """理解用户请求并规划任务"""
    llm = get_llm()
    messages = state["messages"]
    
    # 初始化记忆管理器和上下文管理器
    memory_manager = state.get("memory", {}).get("manager", MemoryManager())
    context_manager = state.get("context", {}).get("manager", ContextManager())
    
    # 获取相关记忆
    relevant_memories = memory_manager.retrieve_relevant(messages[-1].content)
    
    # 构建提示，包含相关记忆
    memory_context = "\n".join([f"- {m.get('content', '')}" for m in relevant_memories])
    
    # 分析用户请求
    analysis_prompt = f"""
    详细分析以下旅行计划需求，提取关键信息并推断用户可能需要的信息。
    
    用户消息: {messages[-1].content}
    
    相关上下文:
    {memory_context}
    
    请提供以下信息:
    1. 旅行主要目标和类型（如休闲、探险、文化体验等）
    2. 可能的目的地偏好
    3. 旅行时间和持续时间（如果未指定，请推断合理的时间）
    4. 旅行预算级别（如果未指定，请推断合理的预算）
    5. 特殊兴趣或偏好（如美食、历史、自然风光等）
    6. 旅行同伴情况（如独自、家庭、情侣等）
    7. 其他可能的限制条件或特殊需求
    
    对于用户未明确指定的信息，请基于上下文和常识进行合理推断，以便能够制定完整的旅行计划。
    """
    
    response = llm.invoke([HumanMessage(content=analysis_prompt)])
    
    # 直接调用decompose_task函数进行任务分解
    subtasks = decompose_task(messages[-1].content)
    
    # 规划执行顺序
    execution_plan = plan_execution(subtasks)
    
    # 更新上下文
    context_manager.update_context("task_analysis", response.content)
    context_manager.update_context("current_task", messages[-1].content)
    
    # 提取目的地信息
    destination_prompt = f"""
    基于以下用户消息和分析，确定最合适的旅行目的地。
    
    用户消息: {messages[-1].content}
    
    分析结果: {response.content}
    
    如果用户明确指定了目的地，请直接提取。
    如果用户未指定目的地但有偏好，请推荐最合适的目的地。
    如果用户完全未提及目的地，请根据分析结果推荐一个热门且适合的旅游目的地。
    
    只返回目的地名称，不要包含其他解释。
    """
    
    destination_response = llm.invoke([HumanMessage(content=destination_prompt)])
    destination = destination_response.content.strip()
    context_manager.update_context("destination", destination)
    
    # 推断旅行时间和持续时间
    duration_prompt = f"""
    基于以下用户消息和分析，确定最合适的旅行持续时间。
    
    用户消息: {messages[-1].content}
    
    分析结果: {response.content}
    
    如果用户明确指定了旅行时间和持续时间，请直接提取。
    如果用户未指定，请根据目的地"{destination}"和旅行类型推荐合适的持续时间（天数）。
    
    只返回天数，例如"5"表示5天4晚，不要包含其他解释。
    """
    
    duration_response = llm.invoke([HumanMessage(content=duration_prompt)])
    try:
        duration = int(duration_response.content.strip())
    except:
        duration = 5  # 默认5天
    context_manager.update_context("duration", duration)
    
    # 推断旅行预算
    budget_prompt = f"""
    基于以下用户消息和分析，确定旅行预算级别。
    
    用户消息: {messages[-1].content}
    
    分析结果: {response.content}
    
    如果用户明确指定了预算，请直接提取。
    如果用户未指定，请根据目的地"{destination}"和旅行持续时间{duration}天推荐合适的预算级别。
    
    请从以下选项中选择一个：
    - 经济型
    - 舒适型
    - 豪华型
    
    只返回预算级别，不要包含其他解释。
    """
    
    budget_response = llm.invoke([HumanMessage(content=budget_prompt)])
    budget = budget_response.content.strip()
    context_manager.update_context("budget", budget)
    
    # 推断旅行类型和特殊兴趣
    interests_prompt = f"""
    基于以下用户消息和分析，确定用户的旅行类型和特殊兴趣。
    
    用户消息: {messages[-1].content}
    
    分析结果: {response.content}
    
    请从以下类别中选择最多3个与用户兴趣相符的选项：
    - 自然风光
    - 历史文化
    - 美食体验
    - 购物娱乐
    - 冒险活动
    - 休闲放松
    - 艺术欣赏
    - 体育运动
    
    以JSON格式返回，例如：["自然风光", "历史文化", "美食体验"]
    """
    
    interests_response = llm.invoke([HumanMessage(content=interests_prompt)])
    try:
        interests = json.loads(interests_response.content.strip())
    except:
        interests = ["自然风光", "历史文化", "美食体验"]
    context_manager.update_context("interests", interests)
    
    # 推断旅行同伴情况
    companions_prompt = f"""
    基于以下用户消息和分析，确定用户的旅行同伴情况。
    
    用户消息: {messages[-1].content}
    
    分析结果: {response.content}
    
    请从以下选项中选择一个：
    - 独自旅行
    - 情侣出游
    - 家庭旅行
    - 朋友结伴
    - 团队旅行
    
    只返回旅行同伴类型，不要包含其他解释。
    """
    
    companions_response = llm.invoke([HumanMessage(content=companions_prompt)])
    companions = companions_response.content.strip()
    context_manager.update_context("companions", companions)
    
    # 保存上下文快照
    context_manager.save_context_snapshot()
    
    # 添加到记忆
    memory_manager.add_to_short_term({
        "type": "task",
        "content": messages[-1].content,
        "analysis": response.content,
        "inferred_details": {
            "destination": destination,
            "duration": duration,
            "budget": budget,
            "interests": interests,
            "companions": companions
        }
    })
    
    return {
        **state,
        "task_plan": execution_plan,
        "task_status": "planned",
        "execution_context": {"current_subtask": 0},
        "memory": {"manager": memory_manager},
        "context": {"manager": context_manager}
    }

def execute_task(state: AgentState) -> AgentState:
    """执行任务（增强版）"""
    context = state["execution_context"]
    plan = state["task_plan"]
    current_subtask = plan["execution_plan"][context["current_subtask"]]
    
    # 获取记忆和上下文管理器
    memory_manager = state.get("memory", {}).get("manager", MemoryManager())
    context_manager = state.get("context", {}).get("manager", ContextManager())

    tools_map = {
        "browse_web": browse_web,
        "analyze_data": analyze_data,
        "search_knowledge_base": search_knowledge_base,
        "summarize_text": summarize_text,
        "get_weather": get_weather,
        "get_travel_recommendations": get_travel_recommendations,
        "calculate_travel_budget": calculate_travel_budget
    }
    
    # 初始化执行结果和日志
    results = {}
    execution_log = context.get("logs", [])
    
    # 获取上下文信息
    destination = context_manager.get_context("destination", "热门旅游目的地")
    duration = context_manager.get_context("duration", 5)
    budget = context_manager.get_context("budget", "舒适型")
    interests = context_manager.get_context("interests", ["自然风光", "历史文化", "美食体验"])
    companions = context_manager.get_context("companions", "家庭旅行")
    
    # 执行当前子任务
    for tool_name in current_subtask["tools"]:
        # 获取工具实例
        tool = tools_map.get(tool_name)
        if not tool:
            error_info = f"工具 {tool_name} 未找到"
            memory_manager.add_to_short_term({
                "type": "error",
                "content": error_info,
                "subtask": current_subtask["name"]
            })
            return {
                **state,
                "task_status": "error",
                "error_info": error_info,
                "memory": {"manager": memory_manager},
                "context": {"manager": context_manager}
            }
        
        # 准备工具输入参数
        tool_input = {}
        
        if tool_name == "browse_web":
            # 根据子任务名称确定搜索内容
            if current_subtask["name"] == "确定旅行目的地":
                tool_input = {"url": f"https://example.com/search?q=热门旅游目的地推荐"}
            elif current_subtask["name"] == "收集目的地信息":
                tool_input = {"url": f"https://example.com/search?q={destination}+旅游攻略"}
            elif current_subtask["name"] == "查找住宿选项":
                tool_input = {"url": f"https://example.com/search?q={destination}+{budget}+酒店推荐"}
            elif current_subtask["name"] == "规划交通方案":
                tool_input = {"url": f"https://example.com/search?q=前往{destination}+交通方式"}
            elif current_subtask["name"] == "推荐特色体验":
                interests_str = "、".join(interests)
                tool_input = {"url": f"https://example.com/search?q={destination}+{interests_str}+特色体验"}
            else:
                tool_input = {"url": f"https://example.com/search?q={destination}+旅游"}
        elif tool_name == "analyze_data":
            # 根据子任务名称确定分析类型
            if current_subtask["name"] == "制定行程安排":
                tool_input = {
                    "data": state.get("results", {}).get("browse_web", {"content": f"{destination}旅游信息"}),
                    "analysis_type": "extract"
                }
            elif current_subtask["name"] == "预算规划":
                tool_input = {
                    "data": {
                        "content": f"{destination} {duration}天 {budget}预算 {companions}"
                    },
                    "analysis_type": "basic"
                }
            else:
                tool_input = {
                    "data": state.get("results", {}).get("browse_web", {"content": "旅游信息"}),
                    "analysis_type": "basic"
                }
        elif tool_name == "search_knowledge_base":
            interests_str = "、".join(interests)
            tool_input = {"query": f"{destination} {interests_str} 旅游攻略"}
        elif tool_name == "summarize_text":
            # 收集所有已获取的信息
            all_results = state.get("results", {})
            combined_text = ""
            for tool_result in all_results.values():
                if isinstance(tool_result, dict) and "content" in tool_result:
                    combined_text += tool_result["content"] + "\n\n"
                elif isinstance(tool_result, str):
                    combined_text += tool_result + "\n\n"
            
            tool_input = {"text": combined_text, "max_length": 1000}
        elif tool_name == "get_weather":
            tool_input = {"destination": destination}
        elif tool_name == "get_travel_recommendations":
            tool_input = {
                "destination": destination,
                "interests": interests,
                "budget": budget,
                "duration": duration
            }
        elif tool_name == "calculate_travel_budget":
            # 获取活动列表
            activities = []
            if "get_travel_recommendations" in state.get("results", {}):
                rec_result = state["results"]["get_travel_recommendations"]
                if isinstance(rec_result, dict) and "recommended_spots" in rec_result:
                    activities = rec_result["recommended_spots"]
            
            tool_input = {
                "destination": destination,
                "duration": duration,
                "accommodation_level": budget,
                "activities": activities
            }
        
        # 执行工具调用
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
            
            # 添加到记忆
            memory_manager.add_to_short_term({
                "type": "tool_result",
                "tool": tool_name,
                "input": tool_input,
                "output": result,
                "subtask": current_subtask["name"]
            })
            
            # 如果结果特别重要，提升到长期记忆
            if tool_name in ["analyze_data", "search_knowledge_base", "get_travel_recommendations", "calculate_travel_budget"]:
                memory_manager.promote_to_long_term(len(memory_manager.short_term_memory) - 1, 0.8)
            
            # 如果是确定目的地任务，更新上下文中的目的地
            if current_subtask["name"] == "确定旅行目的地" and tool_name == "browse_web":
                # 从结果中提取可能的目的地
                if isinstance(result, dict) and "content" in result:
                    content = result["content"]
                    # 简单的目的地提取逻辑
                    popular_destinations = ["北京", "上海", "广州", "深圳", "成都", "杭州", "西安", "重庆", "厦门", "三亚"]
                    for dest in popular_destinations:
                        if dest in content:
                            context_manager.update_context("destination", dest)
                            break
        except Exception as e:
            # 记录错误日志
            error_info = f"工具 {tool_name} 执行错误: {str(e)}"
            execution_log.append({
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name,
                "status": "error",
                "error": error_info
            })
            
            # 添加到记忆
            memory_manager.add_to_short_term({
                "type": "error",
                "content": error_info,
                "subtask": current_subtask["name"]
            })
            
            return {
                **state,
                "task_status": "error",
                "error_info": error_info,
                "execution_context": {**context, "logs": execution_log},
                "memory": {"manager": memory_manager},
                "context": {"manager": context_manager}
            }
    
    # 更新执行上下文
    updated_context = {
        **context,
        "logs": execution_log,
        "last_executed": current_subtask["name"],
        "last_executed_at": datetime.now().isoformat()
    }
    
    # 检查是否还有更多子任务
    if context["current_subtask"] < len(plan["execution_plan"]) - 1:
        # 移动到下一个子任务
        updated_context["current_subtask"] = context["current_subtask"] + 1
        task_status = "executing"
    else:
        # 所有子任务已完成
        task_status = "completed"
    
    # 保存上下文快照
    context_manager.save_context_snapshot()
    
    return {
        **state,
        "results": {**state.get("results", {}), **results},
        "task_status": task_status,
        "execution_context": updated_context,
        "memory": {"manager": memory_manager},
        "context": {"manager": context_manager}
    }

def handle_errors(state: AgentState) -> AgentState:
    """处理执行过程中的错误"""
    # 获取记忆和上下文管理器
    memory_manager = state.get("memory", {}).get("manager", MemoryManager())
    context_manager = state.get("context", {}).get("manager", ContextManager())
    
    # 获取错误信息
    error_info = state.get("error_info", "未知错误")
    
    # 记录错误
    memory_manager.add_to_short_term({
        "type": "error_handling",
        "content": f"处理错误: {error_info}",
        "timestamp": datetime.now().isoformat()
    })
    
    # 尝试恢复执行
    execution_context = state["execution_context"]
    task_plan = state["task_plan"]
    
    # 检查是否有更多子任务可以执行
    if execution_context.get("current_subtask", 0) < len(task_plan.get("execution_plan", [])):
        # 可以继续执行下一个子任务
        return {
            **state,
            "task_status": "executing",
            "memory": {"manager": memory_manager},
            "context": {"manager": context_manager}
        }
    else:
        # 无法继续执行，标记为失败
        messages = state["messages"] + [
            AIMessage(content=f"很抱歉，在执行任务时遇到了问题: {error_info}。无法完成旅行计划的制定。")
        ]
        
        return {
            **state,
            "messages": messages,
            "task_status": "failed",
            "memory": {"manager": memory_manager},
            "context": {"manager": context_manager}
        }

def generate_report(state: AgentState) -> AgentState:
    """生成增强版任务报告，包含执行历史和记忆信息"""
    llm = get_llm()
    
    # 获取记忆和上下文管理器
    memory_manager = state.get("memory", {}).get("manager", MemoryManager())
    context_manager = state.get("context", {}).get("manager", ContextManager())
    
    # 获取执行结果和上下文
    results = state["results"]
    execution_context = state["execution_context"]
    task_plan = state["task_plan"]
    
    # 获取旅行相关信息
    destination = context_manager.get_context("destination", "未指定目的地")
    duration = context_manager.get_context("duration", 5)
    budget = context_manager.get_context("budget", "舒适型")
    interests = context_manager.get_context("interests", ["自然风光", "历史文化", "美食体验"])
    companions = context_manager.get_context("companions", "家庭旅行")
    
    # 构建报告提示
    report_prompt = f"""
    请为目的地"{destination}"生成一份详细的{duration}天旅行计划报告。
    
    旅行基本信息：
    - 目的地: {destination}
    - 旅行天数: {duration}天
    - 预算级别: {budget}
    - 旅行同伴: {companions}
    - 兴趣偏好: {', '.join(interests)}
    
    基于以下执行信息：
    
    1. 执行结果：
    {json.dumps(results, indent=2, ensure_ascii=False)}
    
    请生成包含以下内容的旅行计划：
    
    1. 目的地概览：简要介绍{destination}的特色和亮点，为什么它适合{companions}和{', '.join(interests)}的体验。
    
    2. 详细行程安排：
       - 请为{duration}天的旅行提供每天的详细行程
       - 每天应包括上午、下午和晚上的活动安排
       - 考虑景点之间的距离和游览时间
       - 安排适当的用餐和休息时间
       - 根据{', '.join(interests)}的兴趣偏好安排相应活动
    
    3. 住宿推荐：
       - 提供3-5个符合{budget}预算的住宿选择
       - 包括位置、价格范围和特色
       - 说明为什么这些住宿适合{companions}
    
    4. 交通信息：
       - 往返{destination}的交通建议
       - 当地交通方式和使用建议
       - 如何在景点之间高效移动
    
    5. 必游景点和特色体验：
       - 列出5-8个必游景点，并说明特色
       - 推荐2-3个符合{', '.join(interests)}兴趣的特色体验
       - 提供最佳游览时间和小贴士
    
    6. 美食推荐：
       - 推荐当地特色美食和餐厅
       - 包括不同价位的选择
       - 特别标注适合{companions}的餐厅
    
    7. 购物指南：
       - 推荐特色商品和购物场所
       - 价格参考和砍价技巧
    
    8. 天气和穿着建议：
       - 根据当地天气情况提供穿着建议
       - 季节性注意事项
    
    9. 预算规划：
       - 提供详细的预算明细（交通、住宿、餐饮、门票、购物等）
       - 省钱技巧和注意事项
    
    10. 旅行贴士：
       - 当地文化、习俗和注意事项
       - 安全提示
       - 紧急联系方式和医疗信息
    
    报告应当专业、全面且易于理解，适合{companions}参考使用。请确保行程安排合理，活动丰富多样，能够充分体验{destination}的特色。
    """
    
    # 调用LLM生成报告
    response = llm.invoke([HumanMessage(content=report_prompt)])
    
    # 将报告添加到对话历史
    messages = state["messages"] + [
        AIMessage(content=f"【{destination} {duration}天{budget}旅行计划】\n\n{response.content}")
    ]
    
    # 将报告添加到记忆
    memory_manager.add_to_short_term({
        "type": "report",
        "content": response.content,
        "timestamp": datetime.now().isoformat()
    })
    
    # 如果是重要报告，提升到长期记忆
    memory_manager.promote_to_long_term(len(memory_manager.short_term_memory) - 1, 0.9)
    
    # 更新上下文
    context_manager.update_context("final_report", response.content)
    context_manager.update_context("task_completed_at", datetime.now().isoformat())
    context_manager.save_context_snapshot()
    
    return {
        **state,
        "messages": messages,
        "task_status": "completed",
        "memory": {"manager": memory_manager},
        "context": {"manager": context_manager}
    }

def create_agent() -> StateGraph:
    """创建并配置工作流实例"""
    # 初始化工作流
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("understand_request", understand_request)
    workflow.add_node("execute_task", execute_task)
    workflow.add_node("handle_errors", handle_errors)
    workflow.add_node("generate_report", generate_report)
    
    # 配置节点转换关系
    workflow.set_entry_point("understand_request")
    
    workflow.add_edge("understand_request", "execute_task")
    
    # 配置execute_task节点的条件转换
    workflow.add_conditional_edges(
        "execute_task",
        lambda x: "generate_report" if x["task_status"] == "completed" else "handle_errors",
        {
            "generate_report": "generate_report",
            "handle_errors": "handle_errors"
        }
    )
    
    # 从generate_report到END
    workflow.add_edge("generate_report", END)
    
    # 配置handle_errors节点的条件转换
    workflow.add_conditional_edges(
        "handle_errors",
        lambda x: END if x["task_status"] == "failed" else "execute_task",
        {
            END: END,
            "execute_task": "execute_task"
        }
    )
    
    return workflow

def save_mermaid_graph(mermaid_code: str, output_file: str):
    with open("graph.mmd", "w") as f:
        f.write(mermaid_code)
    # 使用 Mermaid CLI 生成图形
    # mmdc -i graph.mmd -o agent.graph.png
    
    # 使用 Mermaid CLI 生成图形
    # subprocess.run(["mmdc", "-i", "graph.mmd", "-o", output_file])

def main():
    """主函数：初始化并执行工作流，展示执行结果"""
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='智能旅行计划助手')
    parser.add_argument('--task', type=str, help='要执行的任务描述', default='请帮我制定一个旅行计划')
    parser.add_argument('--full', action='store_true', help='显示完整报告', default=True)
    args = parser.parse_args()

    # 初始化记忆和上下文管理器
    memory_manager = MemoryManager()
    context_manager = ContextManager()
    
    # 初始化状态
    initial_state = {
        "messages": [HumanMessage(content=args.task)],
        "task_status": "new",
        "task_plan": {},
        "execution_context": {},
        "results": {},
        "memory": {"manager": memory_manager},
        "context": {"manager": context_manager}
    }

    # 创建Agent实例
    agent = create_agent().compile()
    mermaid_code = agent.get_graph().draw_mermaid()
    save_mermaid_graph(mermaid_code, "output_graph.png")

    # 然后使用 IPython 显示生成的图形
    display(Image("output_graph.png"))

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
            print(f"  - {key}: {str(value)[:100]}...")
    
    if result.get('messages'):
        print("\n💬 最终报告：")
        final_report = result['messages'][-1].content
        
        # 检查是否需要显示完整报告
        if args.full:
            print(final_report)
        else:
            # 提取并显示报告的主要部分
            sections = extract_report_sections(final_report)
            for section_title, section_content in sections.items():
                print(f"\n### {section_title} ###")
                # 对于每个部分显示摘要内容
                if len(section_content) > 300:
                    print(section_content[:300] + "...")
                else:
                    print(section_content)
            
            print("\n(使用 --full 参数查看完整报告)")
    
    # 显示记忆和上下文摘要
    if result.get('memory', {}).get('manager'):
        memory_summary = result['memory']['manager'].get_memory_summary()
        print("\n🧠 记忆摘要：")
        print(f"  - 短期记忆项数: {memory_summary['short_term_count']}")
        print(f"  - 长期记忆项数: {memory_summary['long_term_count']}")

def extract_report_sections(report_text):
    """从报告文本中提取各个部分"""
    sections = {}
    
    # 常见的报告部分标题
    section_patterns = [
        "目的地概览", "行程安排", "住宿推荐", "交通信息", 
        "必游景点", "特色体验", "美食推荐", "购物指南", 
        "天气和穿着建议", "预算规划", "旅行贴士"
    ]
    
    # 使用正则表达式提取各部分内容
    import re
    
    # 查找所有可能的部分标题
    all_headers = re.findall(r'#+\s*(.*?)\s*\n', report_text)
    all_headers.extend(re.findall(r'\d+\.\s*(.*?)\s*\n', report_text))
    all_headers.extend(re.findall(r'\*\*(.*?)\*\*', report_text))
    
    # 添加找到的标题到模式中
    for header in all_headers:
        for pattern in section_patterns:
            if pattern.lower() in header.lower() and pattern not in section_patterns:
                section_patterns.append(header)
    
    # 提取各部分内容
    current_section = "概述"
    sections[current_section] = ""
    
    for line in report_text.split('\n'):
        # 检查是否是新的部分标题
        is_new_section = False
        for pattern in section_patterns:
            if pattern.lower() in line.lower() and (
                line.startswith('#') or 
                re.match(r'\d+\.', line) or 
                line.startswith('**')
            ):
                current_section = pattern
                sections[current_section] = line + "\n"
                is_new_section = True
                break
        
        if not is_new_section and current_section in sections:
            sections[current_section] += line + "\n"
    
    return sections

if __name__ == "__main__":
    main()