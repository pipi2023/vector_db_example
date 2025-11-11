import os
import re
import gradio as gr
from pymilvus import MilvusClient, DataType
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
import time
from datetime import datetime
from agent import generate_ans_with_rag, call_deepseek_api_with_rag

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CourseQASystem")

# 初始化配置
COLLECTION_NAME = "DB_Agent"
DIMENSION = 384
DB_PATH = "http://localhost:19530"

# 停用词列表
STOP_WORDS = {
    "的", "是", "在", "和", "有", "这个", "那个", "什么", "怎么", "如何", "为什么",
    "吗", "呢", "了", "啊", "呀", "吧", "嗯", "哦", "哈", "哎", "呃", "那么",
    "这个", "那个", "这些", "那些", "一种", "一个", "一些", "一点", "一下",
    "可以", "应该"
}

class CourseQASystem:
    def __init__(self):
        self.client = None
        self.model = None
        self.initialized = False
        
    def initialize(self, force_recreate=False):
        """初始化系统"""
        try:
            logger.info("开始初始化课程知识问答系统...")

            self.client = self._init_milvus_client()

            self.model = self._init_model()

            self._create_collection(force_recreate)

            data_count = self._load_initial_data()
            
            self.initialized = True
            logger.info("系统初始化完成")
            return True, "系统初始化成功"
            
        except Exception as e:
            error_msg = f"系统初始化失败: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _init_milvus_client(self):
        """初始化 Milvus 客户端"""
        try:
            client = MilvusClient(DB_PATH)
            logger.info("MilvusClient 初始化成功")
            return client
        except Exception as e:
            logger.error(f"MilvusClient 初始化失败: {e}")
            raise
    
    def _init_model(self):
        """初始化模型"""
        try:
            model = SentenceTransformerEmbeddingFunction('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("模型加载成功")
            return model
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _create_collection(self, force_recreate=False):
        """创建集合"""
        collection_exists = self.client.has_collection(COLLECTION_NAME)
        
        if collection_exists:
            if force_recreate:
                self.client.drop_collection(COLLECTION_NAME)
                logger.info(f"集合 {COLLECTION_NAME} 已删除并重新创建")
            else:
                logger.info(f"集合 {COLLECTION_NAME} 已存在，直接使用")
                return
        else:
            logger.info(f"集合 {COLLECTION_NAME} 不存在，开始创建")
        
        # 创建新集合
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=False
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="chapter", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=2000)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
        
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128}
        )
        
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"集合 {COLLECTION_NAME} 创建成功")
    
    def _load_initial_data(self):
        """加载初始数据"""
        data_sources = []
        
        # 1. 尝试从CSV文件加载
        csv_path = "milvus_data/knowledge_data.csv"
        if os.path.exists(csv_path):
            csv_data = self._load_data_from_csv(csv_path)
            if csv_data:
                data_sources.append(("CSV文件", csv_data))
        
        # 2. 如果CSV文件不存在或为空，使用示例数据
        if not data_sources:
            example_data = self._get_example_data()
            data_sources.append(("示例数据", example_data))
        
        total_inserted = 0
        for source_name, data in data_sources:
            inserted = self._bulk_import_data(data)
            total_inserted += inserted
            logger.info(f"从{source_name}加载了{inserted}条数据")
        
        return total_inserted
    
    def _get_example_data(self):
        """获取示例数据"""
        return [
            {"chapter": "第一章 数据库系统概论", "content": "数据库技术是信息系统的核心技术和重要基础设施,广泛应用于OLTP、OLAP、CAD/CAM、CIMS、电子商务、电子政务和GIS等领域。"},
            {"chapter": "第一章 数据库系统概论", "content": "数据库发展经历了三代演变:层次/网状数据库、关系数据库和新一代数据库系统,并造就了多位图灵奖得主。"},
            {"chapter": "第一章 数据库系统概论", "content": "数据库系统的三级模式结构包括外模式、模式和内模式，提供了数据的物理独立性和逻辑独立性。"},
            {"chapter": "第二章 关系模型和关系运算理论", "content": "关系模型由关系数据结构、关系操作集合和关系完整性约束三部分组成。"},
            {"chapter": "第二章 关系模型和关系运算理论", "content": "关系操作包括查询(选择、投影、连接、除、并、交、差)和数据更新(插入、删除、修改)。"},
            {"chapter": "第三章 关系规范化基础", "content": "关系模式用来定义关系。关系模式中不合适的数据依赖会导致数据冗余、更新异常、插入异常和删除异常。"},
            {"chapter": "第三章 关系规范化基础", "content": "常见的数据依赖包括函数依赖和多值依赖。"},
        ]
    
    def _load_data_from_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """从CSV文件加载数据"""
        try:
            df = pd.read_csv(file_path)
            data = []
            for _, row in df.iterrows():
                # 尝试自动检测列名
                chapter_col = None
                content_col = None
                
                for col in df.columns:
                    if 'chapter' in col.lower() or '章节' in col:
                        chapter_col = col
                    elif 'content' in col.lower() or '内容' in col or 'knowledge' in col.lower():
                        content_col = col
                
                if chapter_col is None or content_col is None:
                    # 使用前两列作为默认
                    chapter_col = df.columns[0]
                    content_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                
                content = str(row[content_col])
                if content:  # 只添加非空内容
                    data.append({
                        "chapter": str(row[chapter_col]),
                        "content": content
                    })
            
            logger.info(f"从 {file_path} 成功加载 {len(data)} 条数据")
            return data
        except Exception as e:
            logger.error(f"从CSV文件加载数据失败: {e}")
            return []
    
    def _preprocess_content(self, content: str) -> str:
        """内容预处理"""
        if pd.isna(content):
            return ""
        
        # 去除特殊字符和多余空格
        content = re.sub(r'[^\w\u4e00-\u9fff\s.,!?;:，。！？；：]', '', str(content))
        content = re.sub(r'\s+', ' ', content).strip()
        
        # 过滤过短的内容
        if len(content) < 10:
            return ""
        
        return content
    
    def _bulk_import_data(self, data_batch: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """批量导入数据"""
        if not data_batch:
            logger.warning("数据批次为空")
            return 0
        
        total_inserted = 0
        
        for i in range(0, len(data_batch), batch_size):
            batch = data_batch[i:i + batch_size]
            
            texts = [item["content"] for item in batch]
            
            logger.info(f"正在为第 {i//batch_size + 1} 批数据生成向量，共 {len(texts)} 条文本")
            vectors = self.model(texts)
            
            insert_data = []
            for j, (item, vector) in enumerate(zip(batch, vectors)):
                insert_data.append({
                    "chapter": item.get("chapter", "default_chapter"),
                    "content": item["content"],
                    "vector": vector
                })
            
            try:
                insert_result = self.client.insert(COLLECTION_NAME, insert_data)
                inserted_count = len(insert_result['ids'])
                total_inserted += inserted_count
                logger.info(f"第 {i//batch_size + 1} 批数据插入成功，共 {inserted_count} 条记录")
            except Exception as e:
                logger.error(f"第 {i//batch_size + 1} 批数据插入失败: {e}")
        
        logger.info(f"批量数据插入完成，总共插入 {total_inserted} 条记录")
        return total_inserted
    
    def similarity_search(self, query_text: str, top_k: int = 5, score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """相似性搜索"""
        if not query_text.strip():
            return []
        
        try:
            query_vector = self.model([query_text])[0]
            
            search_results = self.client.search(
                collection_name=COLLECTION_NAME,
                data=[query_vector],
                limit=top_k * 3,  # 获取更多结果用于过滤
                output_fields=["chapter", "content"],
                search_params={"metric_type": "L2", "params": {"nprobe": 20}}
            )
            
            formatted_results = []
            for result in search_results[0]:
                score = 1 - result["distance"]
                
                if score >= score_threshold:
                    formatted_results.append({
                        "chapter": result["entity"]["chapter"],
                        "content": result["entity"]["content"],
                        "distance": result["distance"],
                        "score": score,
                        "id": result["id"]
                    })
            
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            return formatted_results[:top_k]
        
        except Exception as e:
            logger.error(f"相似性搜索失败: {e}")
            return []
    
    def multi_strategy_search(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """多策略搜索"""
        # 策略1: 直接向量搜索
        vector_results = self.similarity_search(query_text, top_k * 2)
        
        # 策略2: 关键词增强搜索
        keyword_results = self.keyword_enhanced_search(query_text, top_k)
        
        # 策略3: 分块搜索（对长查询）
        chunk_results = []
        if len(query_text) > 20:
            chunk_results = self.chunk_search(query_text, top_k)

        all_results = vector_results + keyword_results + chunk_results

        seen_contents = set()
        unique_results = []
        
        for result in all_results:
        # 使用内容前50个字符作为去重依据
            content_key = result["content"][:50] if len(result["content"]) > 50 else result["content"]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_results.append(result)
        
        # 按分数排序并返回
        unique_results.sort(key=lambda x: x["score"], reverse=True)
        return unique_results[:top_k]
    
    def keyword_enhanced_search(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """关键词增强搜索"""
        keywords = self.extract_keywords(query_text)
        
        if not keywords:
            return []
        
        all_keyword_results = []
        for keyword in keywords[:3]:  # 只使用前3个关键词
            keyword_results = self.similarity_search(keyword, top_k=2)
            all_keyword_results.extend(keyword_results)
        
        return all_keyword_results
    
    def chunk_search(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """分块搜索（处理长查询）"""
        # 简单的分块：按标点符号分割
        chunks = re.split(r'[，。！？；:,\.!?;]', query_text)
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 5]
        
        all_chunk_results = []
        for chunk in chunks[:2]:  # 只处理前2个分块
            chunk_results = self.similarity_search(chunk, top_k=1)
            all_chunk_results.extend(chunk_results)
        
        return all_chunk_results
    
    def extract_keywords(self, text: str) -> List[str]:
        """关键词提取"""
        words = re.findall(r'[\u4e00-\u9fa5]{2,}|[a-zA-Z]{3,}', text)
        
        keywords = [word for word in words if word not in STOP_WORDS]
        
        # 按长度排序，优先选择长词
        keywords.sort(key=len, reverse=True)
        return keywords
    
    def _build_knowledge_base_only_response(self, search_results, question):
        """仅构建知识库答案"""
        if not search_results:
            return "## 🔍 **知识库答案**\n\n未在知识库中找到相关信息。"
        
        response = "## 🔍 **知识库答案**\n\n"
        response += f"针对您的问题：**{question}**\n\n"
        response += "在知识库中找到以下相关内容：\n\n"
        
        # 整合搜索结果生成回答
        for i, result in enumerate(search_results, 1):
            response += f"**{result['chapter']}**\n"
            response += f"{result['content']}\n\n"
        
        response += "---\n\n"
        response += "*💡 以上内容来自课程知识库，如需大模型补充说明，请取消勾选\"仅查看知识库答案\"选项*"
        
        return response
    
    def answer_question(self, question: str, chat_history: List, knowledge_base_only: bool = False) -> Tuple[List, str]:
        """回答问题"""
        if not question.strip():
            return chat_history, ""
        
        start_time = time.time()
        
        try:
            # 向量数据库搜索
            search_results = self.multi_strategy_search(question, 5)
            vec_search_time = time.time() - start_time

            # 根据用户选择生成响应
            if knowledge_base_only:
                # 仅查看知识库答案
                response = self._build_knowledge_base_only_response(search_results, question)
                response_header = "🔍 **知识库答案**"
            else:
                # 完整回答（知识库 + 大模型补充）
                response = generate_ans_with_rag(
                    {'question': question}, 
                    search_results
                )
                response_header = "🔍 **找到相关知识并生成回答**" if search_results else "🤖 **大模型生成的答案**"
            
            total_search_time = time.time() - start_time
            
            # 添加时间信息
            time_info = f"💡 *向量数据库检索用时: {vec_search_time:.2f}秒*\n*总处理用时: {total_search_time:.2f}秒*"
            
            if knowledge_base_only:
                time_info += "\n*当前为仅知识库模式*"
            
            final_response = f"{response_header}\n\n{response}\n\n{time_info}"
            
            # 添加消息到聊天历史
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": final_response})
            
            return chat_history, ""
            
        except Exception as e:
            error_msg = f"❌ 回答问题时出现错误: {str(e)}"
            logger.error(error_msg)
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": error_msg})
            return chat_history, ""

    def insert_knowledge(self, chapter: str, content: str) -> Tuple[bool, str]:
        """插入单条知识"""
        try:
            if not chapter.strip() or not content.strip():
                return False, "章节和内容都不能为空"
            
            # 预处理内容
            processed_content = self._preprocess_content(content)
            if not processed_content:
                return False, "内容无效或过短"
            
            # 构建知识数据
            knowledge_data = [{
                "chapter": chapter.strip(),
                "content": processed_content
            }]
            
            # 插入到数据库
            inserted_count = self._bulk_import_data(knowledge_data, batch_size=1)
            
            if inserted_count > 0:
                return True, f"成功插入知识到章节 '{chapter}'"
            else:
                return False, "插入知识失败"
                
        except Exception as e:
            logger.error(f"插入知识失败: {e}")
            return False, f"插入知识失败: {str(e)}"
    
    def batch_insert_knowledge(self, knowledge_list: List[Dict[str, str]]) -> Tuple[bool, str]:
        """批量插入知识"""
        try:
            if not knowledge_list:
                return False, "知识列表为空"
            
            valid_data = []
            for knowledge in knowledge_list:
                chapter = knowledge.get('chapter', '').strip()
                content = knowledge.get('content', '').strip()
                
                if chapter and content:
                    processed_content = self._preprocess_content(content)
                    if processed_content:
                        valid_data.append({
                            "chapter": chapter,
                            "content": processed_content
                        })
            
            if not valid_data:
                return False, "没有有效的知识数据"
            
            # 批量插入到数据库
            inserted_count = self._bulk_import_data(valid_data, batch_size=len(valid_data))
            
            if inserted_count > 0:
                return True, f"成功批量插入 {inserted_count} 条知识"
            else:
                return False, "批量插入知识失败"
                
        except Exception as e:
            logger.error(f"批量插入知识失败: {e}")
            return False, f"批量插入知识失败: {str(e)}"
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            # 获取集合信息
            collection_info = self.client.describe_collection(COLLECTION_NAME)
            
            # 获取行数
            count_result = self.client.query(
                collection_name=COLLECTION_NAME,
                filter="",
                output_fields=["count(*)"]
            )
            
            row_count = len(count_result) if count_result else 0
            
            return {
                "row_count": row_count,
                "dimension": DIMENSION,
                "collection_name": COLLECTION_NAME,
                "description": collection_info.get('description', '')
            }
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {"row_count": 0, "dimension": DIMENSION, "collection_name": COLLECTION_NAME}
        
# 全局系统实例
qa_system = CourseQASystem()

def answer_question_interface(question, chat_history, knowledge_base_only):
    """回答问题接口"""
    return qa_system.answer_question(question, chat_history, knowledge_base_only)

def clear_chat():
    """清空对话"""
    return [], []

def get_system_info():
    """获取系统信息"""
    try:
        # 获取集合统计信息
        # stats = qa_system.client.get_collection_stats(COLLECTION_NAME)
        # row_count = stats['row_count']
        
        info = f"""
            ## 📊 系统信息

            - **数据库**: Milvus
            - **向量模型**: paraphrase-multilingual-MiniLM-L12-v2
            - **知识库大小**: 153 条记录
            - **向量维度**: {DIMENSION}
            - **状态**: ✅ 运行中
            - **最后更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
        return info
    except Exception as e:
        return f"获取系统信息失败: {str(e)}"

def main():
    """主函数"""
    # 自动初始化系统
    logger.info("正在自动初始化系统...")
    success, message = qa_system.initialize(force_recreate=False)
    if success:
        logger.info("系统自动初始化成功")
        initial_status = f"✅ **系统已自动初始化**\n\n{message}"
    else:
        logger.error("系统自动初始化失败")
        initial_status = f"❌ **系统自动初始化失败**\n\n{message}"
    
    def insert_single_knowledge(chapter, content):
        """插入单条知识"""
        success, message = qa_system.insert_knowledge(chapter, content)
        if success:
            return f"✅ {message}"
        else:
            return f"❌ {message}"
    
    def batch_insert_knowledge(knowledge_text):
        """批量插入知识"""
        try:
            if not knowledge_text.strip():
                return "请输入知识数据"
            
            knowledge_list = []
            lines = knowledge_text.strip().split('\n')
            
            current_chapter = ""
            current_content = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 检测章节行（包含"第X章"或"章节"等关键词）
                if re.match(r'第[零一二三四五六七八九十百千]+章', line) or '章节' in line:
                    # 保存上一条知识
                    if current_chapter and current_content:
                        knowledge_list.append({
                            "chapter": current_chapter,
                            "content": current_content.strip()
                        })
                    
                    # 开始新的章节
                    current_chapter = line
                    current_content = ""
                else:
                    # 内容行
                    current_content += line + " "
            
            # 添加最后一条知识
            if current_chapter and current_content:
                knowledge_list.append({
                    "chapter": current_chapter,
                    "content": current_content.strip()
                })
            
            if not knowledge_list:
                return "❌ 未识别到有效的知识格式，请确保包含章节信息"
            
            success, message = qa_system.batch_insert_knowledge(knowledge_list)
            if success:
                return f"✅ {message}"
            else:
                return f"❌ {message}"
                
        except Exception as e:
            return f"❌ 批量插入失败: {str(e)}"
    
    def get_updated_system_info():
        """获取更新的系统信息"""
        try:
            stats = qa_system.get_collection_stats()
            row_count = stats["row_count"]
            
            info = f"""
            ## 📊 系统信息

            - **数据库**: Milvus
            - **向量模型**: paraphrase-multilingual-MiniLM-L12-v2
            - **向量维度**: {DIMENSION}
            - **状态**: ✅ 运行中
            - **最后更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            return info
        except Exception as e:
            return f"获取系统信息失败: {str(e)}"

    # 创建 Gradio 界面
    with gr.Blocks(
        title="数据库课程知识问答系统",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .chatbot {
            min-height: 500px;
        }
        .knowledge-base-only {
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #d1e7ff;
        }
        .insert-knowledge {
            background-color: #f0fff0;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #d1ffd1;
        }
        """
    ) as demo:
        gr.Markdown("""
        # 🎓 数据库课程知识问答系统
        **基于 Milvus 向量数据库和 RAG 技术构建**
        """)
        
        # 系统状态区域
        with gr.Row():
            with gr.Column(scale=1):
                info_btn = gr.Button("📊 刷新系统信息", variant="secondary")
            
            with gr.Column(scale=2):
                status_output = gr.Markdown(initial_status)
        
        # 主聊天区域和知识插入区域
        with gr.Tab("💬 问答对话"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="问答对话",
                        height=500,
                        type="messages",
                        show_copy_button=True
                    )
                    chat_state = gr.State([])
                    
                with gr.Column(scale=1):
                    # 设置选项区域
                    with gr.Group(elem_classes="knowledge-base-only"):
                        gr.Markdown("### ⚙️ 回答设置")
                        knowledge_base_only = gr.Checkbox(
                            label="仅查看知识库答案",
                            value=False,
                            info="勾选后只显示知识库内容，不调用大模型"
                        )
                        gr.Markdown("""
                        **模式说明：**
                        - ✅ **关闭**：显示知识库内容 + 大模型补充（推荐）
                        - ☑️ **开启**：仅显示知识库内容，响应更快
                        """)
                    
                    gr.Markdown("""
                    ## 💡 使用提示
                    
                    1. 系统已自动初始化完成
                    2. 在下方输入您的问题
                    3. 系统会从课程知识库中查找相关信息
                    
                    ## 🎯 示例问题
                    - 数据库的三级模式结构是什么？
                    - 关系模型由哪几部分组成？
                    - 什么是函数依赖？
                    """)
            
            # 输入区域
            with gr.Row():
                question_input = gr.Textbox(
                    label="请输入您关于数据库课程的问题",
                    placeholder="例如：数据模型三要素是什么？",
                    lines=2,
                    max_lines=4
                )
            
            with gr.Row():
                submit_btn = gr.Button("📤 提交问题", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")
            
            # 示例问题
            gr.Examples(
                examples=[
                    "数据库DB",
                    "数据的物理独立性是什么？",
                    "码的定义是什么？",
                    "数据模型三要素是什么？",
                    "网状模型缺点是什么？"
                ],
                inputs=question_input,
                label="💡 点击示例问题快速提问"
            )
        
        # 知识插入标签页
        with gr.Tab("📚 插入知识"):
            with gr.Row():
                with gr.Column():
                    with gr.Group(elem_classes="insert-knowledge"):
                        gr.Markdown("### 📝 插入单条知识")
                        
                        single_chapter = gr.Textbox(
                            label="章节名称",
                            placeholder="例如：第一章 数据库系统概论",
                            lines=1
                        )
                        single_content = gr.Textbox(
                            label="知识内容",
                            placeholder="例如：数据库技术是信息系统的核心技术和重要基础设施...",
                            lines=3
                        )
                        insert_single_btn = gr.Button("💾 插入单条知识", variant="primary")
                        single_output = gr.Textbox(label="操作结果", interactive=False)
                
                with gr.Column():
                    with gr.Group(elem_classes="insert-knowledge"):
                        gr.Markdown("### 📚 批量插入知识")
                        
                        batch_knowledge = gr.Textbox(
                            label="批量知识数据",
                            placeholder="""格式示例：
第一章 数据库系统概论
数据库技术是信息系统的核心技术和重要基础设施...
第二章 关系模型
关系模型由关系数据结构、关系操作集合和关系完整性约束三部分组成...
                            """,
                            lines=8
                        )
                        insert_batch_btn = gr.Button("💾 批量插入知识", variant="primary")
                        batch_output = gr.Textbox(label="操作结果", interactive=False)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 📖 插入格式说明
                    
                    **单条插入：**
                    - 填写完整的章节名称和知识内容
                    - 点击"插入单条知识"按钮
                    
                    **批量插入：**
                    - 每段以章节名称开始
                    - 后面跟随该章节的知识内容
                    - 章节名称应该包含"第X章"字样
                    - 空行用于分隔不同章节
                    
                    **示例格式：**
                    ```
                    第一章 数据库系统概论
                    数据库技术是信息系统的核心技术和重要基础设施...
                    
                    第二章 关系模型
                    关系模型由关系数据结构、关系操作集合...
                    ```
                    """)
        
        # 事件绑定
        info_btn.click(
            fn=get_updated_system_info,
            outputs=status_output
        )
        
        submit_btn.click(
            fn=answer_question_interface,
            inputs=[question_input, chat_state, knowledge_base_only],
            outputs=[chatbot, question_input]
        ).then(
            lambda: chat_state.value,
            outputs=chat_state
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, chat_state]
        )
        
        # 插入知识的事件绑定
        insert_single_btn.click(
            fn=insert_single_knowledge,
            inputs=[single_chapter, single_content],
            outputs=single_output
        ).then(
            fn=lambda: ("", ""),  # 清空输入框
            outputs=[single_chapter, single_content]
        )
        
        insert_batch_btn.click(
            fn=batch_insert_knowledge,
            inputs=[batch_knowledge],
            outputs=batch_output
        ).then(
            fn=lambda: "",  # 清空输入框
            outputs=batch_knowledge
        )
        
        # 按Enter键提交
        question_input.submit(
            fn=answer_question_interface,
            inputs=[question_input, chat_state, knowledge_base_only],
            outputs=[chatbot, question_input]
        ).then(
            lambda: chat_state.value,
            outputs=chat_state
        )
    
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,192.168.0.*"
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    # 启动服务
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()