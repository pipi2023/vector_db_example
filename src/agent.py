import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
API_KEY = os.getenv("DEEPSEEK_API_KEY")

def call_deepseek_api(data):
    try:
        message = _build_prompt(data)
        api_key = API_KEY
        if not api_key:
            logger.error("DeepSeek API Key 未配置")
            return None
        
        deepseek_answer = chat_with_deepseek(api_key, message)
        if not deepseek_answer:
            logger.error("DeepSeek API 未返回有效响应")
            return None
        
        return deepseek_answer

    except Exception as e:
        logger.error(f"调用 DeepSeek API 失败：{e}")
        return None


def chat_with_deepseek(api_key, message):
    """
        message: 用户消息
    """
    try:
        # 初始化客户端
        client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key=api_key,
        )
        
        # 创建对话
        response = client.chat.completions.create(
            model='deepseek-ai/DeepSeek-V3.1',
            messages=[{
                "role": "system",
                "content": """
                    # 角色
                    你是一位数据库课程专家，专门回答关于数据库系统、关系模型、SQL、数据规范化等方面的问题。

                    # 任务
                    基于你的专业知识回答用户关于数据库课程的问题。

                    # 技能
                    - 深入理解数据库系统概念
                    - 熟悉关系模型和关系运算
                    - 掌握SQL语言和数据库设计
                    - 了解数据规范化和事务处理

                    # 工作流程
                    1. 仔细分析用户的问题
                    2. 基于专业知识提供准确、详细的回答
                    3. 如果问题涉及具体概念，给出清晰的定义和示例
                    4. 保持回答的专业性和教育性

                    # 输出要求
                    - 回答要专业、准确、详细
                    - 使用中文回答
                    - 如果适用，可以给出示例或实际应用场景
                    - 避免过于简略的回答
                    """
                },
            {'role': 'user', 'content': message}],
            stream=False
        )
        
        # 处理响应
        if response and response.choices:
            full_response = response.choices[0].message.content
            return full_response
        else:
            return None
        
    except Exception as e:
        print(f"发生错误: {e}")
        return None

def _build_prompt(data):
    """
    构建大模型提示词
    """
    question = data.get('question', '')
    
    message = f"""
    请回答以下关于数据库课程的问题：
    
    问题：{question}
    
    请提供专业、详细、准确的回答。如果问题涉及具体概念，请给出定义和示例。
    """
    return message


def generate_ans_with_rag(data, search_results):
    """RAG增强问答生成"""
    if not search_results:
        # 无搜索结果，直接调用大模型
        return call_deepseek_api(data)
    else:
        # 有搜索结果，先输出知识库知识，再调用大模型补充
        return generate_combined_response(data, search_results)

def _build_rag_response(search_results, question):
    """基于搜索结果构建回答"""
    response = "🔍 **基于知识库的答案：**\n\n"
    
    # 整合搜索结果生成回答
    for i, result in enumerate(search_results, 1):
        response += f"**{i}. {result['chapter']}**\n"
        response += f"   📖 {result['content']}\n\n"
    
    response += f"\n💡 以上信息来自课程知识库，针对您的问题：\"{question}\""
    return response

def build_rag_message(search_results):
    """构建RAG提示消息"""
    if not search_results:
        return "未找到相关背景信息。"
    
    message_parts = ["## 相关背景知识："]
    for i, result in enumerate(search_results, 1):
        message_parts.append(f"{i}. **{result['chapter']}**: {result['content']}")
    
    return "\n".join(message_parts)

def call_deepseek_api_with_rag(data, search_results):
    """使用RAG增强调用大模型"""
    try:
        rag_message = build_rag_message(search_results)
        message = _build_enhanced_prompt(data, rag_message)
        api_key = API_KEY
        
        deepseek_answer = chat_with_deepseek(api_key, message)
        if not deepseek_answer:
            return None
        
        return deepseek_answer
        
    except Exception as e:
        logger.error(f"调用 DeepSeek API 失败：{e}")
        return None

def _build_enhanced_prompt(data, rag_message):
    """构建增强提示词"""
    question = data.get('question', '')

    message = f"""
    # 角色
    你是一位数据库课程专家，正在基于课程知识库回答学生问题。：

    # 可用知识
    {rag_message}

    # 用户问题
    {question}

    # 回答要求
    1. **优先使用**提供的课程知识来回答问题
    2. 如果知识库内容不足，可以补充你的专业知识，但要明确说明
    3. 确保回答准确、结构清晰、有教育意义
    4. 使用中文回答，适当使用示例说明
    5. 如果问题涉及多个概念，请分别解释并说明它们的关系

    # 输出格式
    - 先直接回答问题
    - 然后基于知识库内容详细解释
    - 最后可以补充相关知识点
    """
    return message

def generate_combined_response(data, search_results):
    """生成组合响应：知识库内容 + 大模型补充"""
    try:
        # 构建知识库内容部分
        knowledge_base_part = _build_rag_response(search_results, data.get('question', ''))
        
        # 调用大模型进行补充
        rag_message = build_rag_message(search_results)
        message = _build_supplement_prompt(data, rag_message)
        api_key = API_KEY
        
        supplement_answer = chat_with_deepseek(api_key, message)
        
        # 组合两部分内容
        if supplement_answer:
            combined_response = f"{knowledge_base_part}\n\n---\n\n## 🤖 **大模型补充说明**\n\n{supplement_answer}"
        else:
            combined_response = f"{knowledge_base_part}\n\n---\n\n*注：大模型补充暂时不可用，以上为知识库中找到的相关内容。*"
        
        return combined_response
        
    except Exception as e:
        logger.error(f"生成组合响应失败：{e}")
        # 失败时回退到只显示知识库内容
        return _build_rag_response(search_results, data.get('question', ''))

def _build_supplement_prompt(data, rag_message):
    """构建补充说明的提示词"""
    question = data.get('question', '')

    message = f"""
    # 任务
    你是一位数据库课程助教，需要基于已有的知识库内容对问题进行补充说明。

    # 知识库已有内容
    {rag_message}

    # 用户问题
    {question}

    # 重要说明
    用户已经看到了上面这些知识库内容，现在需要你进行补充说明。

    # 你的任务
    基于上面提供的知识库内容，对问题进行补充说明：

    1. **不要简单重复**知识库中已经明确的内容
    2. 对知识库内容进行**解释、扩展和深化**
    3. 可以补充相关的**示例、应用场景或注意事项**
    4. 如果知识库内容比较分散，可以进行**整合和总结**
    5. 如果知识库内容不足，适当补充你的专业知识
    6. 可以指出知识库内容中的**重点和关键概念**

    # 输出要求
    - 直接开始补充说明，不需要开场白
    - 保持专业、准确、易于理解
    - 使用中文回答
    - 重点在于深化理解，而不是重复信息
    - 可以适当使用"补充说明"、"进一步解释"等过渡语
    """
    return message
