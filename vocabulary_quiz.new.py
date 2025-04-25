import uuid
import random
import logging
import json
import os
import traceback
import pickle
from datetime import datetime
from pydantic import Field, validator, root_validator
from typing import List, Dict, Any, Optional, Union
from proconfig.widgets.base import WIDGETS, BaseWidget

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocabulary_quiz.log'))  # 输出到文件
    ]
)
logger = logging.getLogger('VocabularyQuizWidget')

# 创建索引问题专用日志
index_logger = logging.getLogger('VocabularyQuizIndex')
index_logger.setLevel(logging.DEBUG)
# 确保索引日志不会传播到父日志处理器
index_logger.propagate = False
# 添加文件处理器，专门用于记录索引相关问题
index_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index_debug.log')
index_file_handler = logging.FileHandler(index_log_file)
index_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
index_logger.addHandler(index_file_handler)

# 创建专门针对transition_to问题的日志记录器
transition_logger = logging.getLogger('TransitionDebug')
transition_logger.setLevel(logging.DEBUG)
# 确保日志不会传播到父日志处理器
transition_logger.propagate = False
# 添加文件处理器
transition_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transition_debug.log')
transition_file_handler = logging.FileHandler(transition_log_file)
transition_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
transition_logger.addHandler(transition_file_handler)

# 添加一条启动日志，确认日志系统正常工作
logger.info("============== VocabularyQuizWidget 日志系统初始化 ==============")
index_logger.info("============== 索引问题专用日志初始化 ==============")
transition_logger.info("============== Transition问题专用日志初始化 ==============")

@WIDGETS.register_module()
class VocabularyQuizWidget(BaseWidget):
    """单词测验小部件，实现单词测验功能"""
    CATEGORY = "Custom Widgets/Education"
    NAME = "Vocabulary Quiz Widget"
    
    class InputsSchema(BaseWidget.InputsSchema):
        operation: str = Field(
            default="get_batch",
            description="操作类型：prepare（准备题目）, start_quiz（开始测验）, get_next_batch（获取下一批单词）, get_next_question（获取下一个问题）, submit_answer（提交答案）, end_quiz（结束测验）"
        )
        quiz_id: Optional[str] = Field(
            default=None,
            description="考试ID，如果提供则使用之前的测验会话，否则创建新会话"
        )
        questions: Union[str, List[Dict[str, Any]]] = Field(
            default="[]",
            description="题目列表，可直接输入JSON格式，如[{'word': '单词', 'options': ['选项1', '选项2', '选项3', '选项4'], 'correct_index': 0}, ...]"
        )
        word_list: Union[str, List[str]] = Field(
            default="",
            description="单词列表，可以是逗号分隔的字符串('apple,book,computer')或字符串列表(['apple', 'book', 'computer'])"
        )
        batch_size: int = Field(
            default=15,
            description="每批返回的题目数量"
        )
        answer: Optional[Union[str, int]] = Field(
            default=None,
            description="用户的答案，可以是选项序号(0-3)或选项字母(A-D)，用于submit_answer操作"
        )
        question_index: Optional[int] = Field(
            default=None,
            description="考试ID，如果提供则使用之前乱序的题目列表，否则重新乱序"
        )
        operation: str = Field(
            default="get_batch",
            description="操作类型：prepare（准备阶段）, start_quiz（开始测验）, get_next_batch（获取下一批单词）, get_next_question（获取下一个问题）, submit_answer（提交答案）, end_quiz（结束测验）"
        )
        selected_index: Optional[str] = Field(
            default=None,
            description="用户选择的选项，传入'选项A'、'选项B'、'选项C'或'选项D'，分别对应索引0、1、2、3"
        )
        question_index: Optional[int] = Field(
            default=None,
            description="当前问题的索引，用于submit_answer操作（可选，系统会尝试使用会话中保存的当前问题索引）"
        )
        
        @validator('batch_index', 'question_index', pre=True, check_fields=False)
        def handle_empty_string(cls, v):
            """处理空字符串，转换为None"""
            if v == '':
                logger.info(f"将空字符串转换为None")
                return None
            return v
        
        @validator('answer', pre=True, check_fields=False)
        def process_answer(cls, v):
            """处理答案，支持选项字母和数字"""
            if v is None:
                return None
                
            # 如果是字符串类型，并且是选项字母(A/B/C/D)，转换为数字索引
            if isinstance(v, str):
                # 去除空格并转为大写
                v = v.strip().upper()
                # 检查是否为'选项X'格式
                if v.startswith('选项') and len(v) > 2:
                    v = v[2:]  # 提取字母部分
                
                # 如果是A-D字母，转换为0-3的索引
                if v in ['A', 'B', 'C', 'D']:
                    return ord(v) - ord('A')  # A->0, B->1, C->2, D->3
                    
                # 如果是数字字符串，转换为整数
                if v.isdigit():
                    return int(v)
            
            # 如果已经是整数，直接返回
            if isinstance(v, int):
                return v
                
            # 其他情况返回原值
            return v
        
        @root_validator(pre=True)
        def handle_mode_operation_compatibility(cls, values):
            """处理mode和operation的兼容性"""
            # 获取mode和operation值
            mode = values.get('mode')
            operation = values.get('operation', 'get_batch')
            
            # 如果提供了mode且操作是默认值，根据mode更新operation
            if mode is not None and operation == 'get_batch':
                if mode == 'prepare':
                    values['operation'] = 'prepare'
                    logger.info(f"基于mode参数'{mode}'设置operation为'prepare'")
            
            return values
        
        @root_validator(pre=True)
        def validate_questions(cls, values):
            # 记录初始输入
            logger.info(f"输入验证开始: {values.keys()}")
            
            # 获取模式、questions和word_list
            mode = values.get('mode', 'quiz')
            questions = values.get('questions', [])
            word_list = values.get('word_list', None)
            
            # 处理字符串格式的JSON输入
            if isinstance(questions, str) and questions.strip():
                try:
                    questions = json.loads(questions)
                    logger.info("成功解析字符串格式的JSON题目列表")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误: {str(e)}")
                    questions = []
            
            # 记录初始values
            logger.info(f"操作模式: {mode}, 初始questions类型: {type(questions)}, word_list类型: {type(word_list)}")
            
            # 处理questions
            valid_questions = []
            
            # 如果是prepare模式，则允许questions为空，主要处理word_list
            if mode == 'prepare':
                logger.info("准备模式：主要处理word_list")
                # word_list为空时发出警告
                if not word_list:
                    logger.warning("prepare模式下word_list为空")
            # 如果是quiz模式，则需要验证questions
            elif mode == 'quiz':
                logger.info("测验模式：验证questions")
                # 确保questions是列表
                if not isinstance(questions, list):
                    if questions:  # 如果不为空，记录警告
                        logger.warning(f"questions不是列表类型: {type(questions)}")
                    questions = []
                
                # 过滤无效的questions项
                for item in questions:
                    if isinstance(item, dict) and 'word' in item:
                        # 确保选项是列表
                        if 'options' not in item or not isinstance(item['options'], list):
                            item['options'] = []
                        # 确保correct_index存在
                        if 'correct_index' not in item:
                            item['correct_index'] = 0
                        valid_questions.append(item)
                    else:
                        logger.warning(f"跳过无效的question项: {item}")
                
                # quiz模式下如果questions为空，检查是否有word_list可用
                if not valid_questions and not word_list:
                    logger.warning("quiz模式下questions为空且无word_list可用")
            else:
                logger.warning(f"未知的操作模式: {mode}")
            
            # 处理word_list
            if word_list:
                try:
                    words = []
                    if isinstance(word_list, str):
                        logger.info(f"处理字符串类型的word_list: {word_list[:100]}...")
                        words = [w.strip() for w in word_list.split(",") if w.strip()]
                    elif isinstance(word_list, list):
                        logger.info(f"处理列表类型的word_list, 长度: {len(word_list)}")
                        words = [str(w).strip() for w in word_list if str(w).strip()]
                    else:
                        logger.warning(f"未知类型的word_list: {type(word_list)}")
                    
                    logger.info(f"从word_list获取到{len(words)}个单词")
                    
                    # 将word_list转换为questions项
                    for word in words:
                        if not any(q.get('word') == word for q in valid_questions):
                            valid_questions.append({
                                "word": word,
                                "options": [],
                                "correct_index": 0
                            })
                except Exception as e:
                    logger.error(f"处理word_list时出错: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # 更新values中的questions
            logger.info(f"验证后的questions数量: {len(valid_questions)}")
            values['questions'] = valid_questions
            
            return values
    
    class OutputsSchema(BaseWidget.OutputsSchema):
        quiz_id: str = Field(description="考试唯一标识")
        status: str = Field(description="操作状态：success, error, batch_completed, 或考试状态：not_started, in_progress, completed")
        message: Optional[str] = Field(description="状态消息，尤其是错误信息")
        words: Optional[str] = Field(description="当前批次的单词，逗号分隔")
        total_questions: Optional[int] = Field(description="总题目数")
        answered_count: Optional[int] = Field(description="已回答题目数")
        correct_count: Optional[int] = Field(description="正确题目数")
        wrong_answers: Optional[List[Dict[str, Any]]] = Field(
            description="错误回答详情 [{'word': '单词', 'selected': '选择的选项', 'correct': '正确选项'}, ...]"
        )
        score_percentage: Optional[float] = Field(description="得分百分比")
        current_batch: Optional[int] = Field(description="当前批次索引")
        total_batches: Optional[int] = Field(description="总批次数")
        formatted_question: Optional[str] = Field(description="格式化后的问题文本，包含单词和ABCD选项，带HTML换行符")
        question_index: Optional[int] = Field(description="当前问题在题目列表中的索引，用于提交答案")
        should_transition: Optional[bool] = Field(default=False, description="是否应该转换状态，为true时表示需要流转到其他状态")
        transition_to: Optional[str] = Field(default=None, description="应该流转到的目标状态名称，可能的值：quiz_completed, get_next_batch")
        
    # 用于存储测验状态的字典，键为quiz_id
    sessions = {}
    
    # 会话数据文件路径
    _sessions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quiz_sessions.pkl')
    
    @classmethod
    def dump_sessions(cls):
        """将会话数据持久化保存到文件"""
        try:
            # 记录当前会话状态的摘要
            logger.info(f"准备保存会话数据，当前共有 {len(cls.sessions)} 个会话")
            
            # 记录每个会话的基本信息
            for quiz_id, session in cls.sessions.items():
                questions_count = len(session.get("shuffled_questions", []))
                answers_count = len(session.get("answers", {}))
                status = session.get("status", "unknown")
                logger.info(f"会话 {quiz_id[:8]}...: 状态={status}, 题目数={questions_count}, 已答题数={answers_count}")
            
            # 保存到文件
            with open(cls._sessions_file, 'wb') as f:
                pickle.dump(cls.sessions, f)
            
            logger.info(f"成功保存会话数据到 {cls._sessions_file}，共 {len(cls.sessions)} 个会话")
            return True
        except Exception as e:
            logger.error(f"保存会话数据失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    @classmethod
    def load_sessions(cls):
        """从文件加载会话数据"""
        try:
            if os.path.exists(cls._sessions_file):
                with open(cls._sessions_file, 'rb') as f:
                    cls.sessions = pickle.load(f)
                logger.info(f"成功从 {cls._sessions_file} 加载 {len(cls.sessions)} 个会话")
                return True
            else:
                logger.info(f"会话数据文件不存在: {cls._sessions_file}")
                return False
        except Exception as e:
            logger.error(f"加载会话数据失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def __init__(self):
        super().__init__()
        # 确保兼容性：使_quiz_sessions和sessions指向同一对象
        self.__class__._quiz_sessions = self.__class__.sessions
        # 尝试加载会话数据
        self.__class__.load_sessions()
    
    def execute(self, environ, config):
        """执行小部件的主要入口方法"""
        try:
            # 获取操作类型，operation会被验证器处理，已包含兼容性逻辑
            operation = config.operation
            
            logger.info(f"执行操作开始 - 操作类型: {operation}")
            logger.info(f"输入参数: quiz_id={config.quiz_id}, batch_size={config.batch_size}")
            
            # 记录questions数量和内容
            if hasattr(config, 'questions'):
                # 确保questions已经是列表类型
                if isinstance(config.questions, str):
                    try:
                        if config.questions.strip():
                            logger.info("questions是字符串类型，尝试解析为JSON")
                            # 确保处理JSON时不会出现Unicode编码问题
                            # 保留原始的JSON字符串以便可能需要的调试
                            logger.info(f"原始JSON字符串: {config.questions[:200]}...")
                            parsed_questions = json.loads(config.questions)
                            # 记录解析后的数据结构
                            logger.info(f"成功解析为JSON，数据类型: {type(parsed_questions)}")
                            if isinstance(parsed_questions, list):
                                logger.info(f"解析后的列表长度: {len(parsed_questions)}")
                                if len(parsed_questions) > 0:
                                    logger.info(f"第一个题目样例: {json.dumps(parsed_questions[0], ensure_ascii=False)}")
                            config.questions = parsed_questions
                        else:
                            config.questions = []
                    except Exception as e:
                        logger.error(f"解析questions字符串失败: {str(e)}")
                        logger.error(traceback.format_exc())
                        config.questions = []
                
                logger.info(f"题目数量: {len(config.questions)}")
                
                # 记录前5个题目的内容作为示例
                if len(config.questions) > 0:
                    sample_questions = config.questions[:min(5, len(config.questions))]
                    for i, q in enumerate(sample_questions):
                        logger.info(f"题目示例{i+1}: {json.dumps(q, ensure_ascii=False)}")
            
            # 处理prepare操作
            if operation == "prepare":
                logger.info("执行prepare操作")
                result = self._prepare(config)
            # 处理start_quiz操作
            elif operation == "start_quiz":
                logger.info("执行start_quiz操作")
                result = self._start_quiz(config)
            # 处理get_next_batch操作
            elif operation == "get_next_batch" or operation == "get_batch":  # 兼容旧版本操作名
                logger.info("执行get_next_batch操作")
                result = self._get_next_batch(config)
            # 处理get_next_question操作
            elif operation == "get_next_question":
                logger.info("执行get_next_question操作")
                result = self._get_next_question(config)
            # 处理submit_answer操作
            elif operation == "submit_answer":
                logger.info(f"执行submit_answer操作")
                
                # 记录所有配置参数
                log_params = {k: v for k, v in vars(config).items() if k != 'questions'}
                try:
                    logger.info(f"submit_answer完整参数: {json.dumps(log_params, ensure_ascii=False)}")
                except TypeError:
                    # 处理不可序列化对象
                    simple_log_params = {}
                    for k, v in log_params.items():
                        try:
                            json.dumps({k: v})
                            simple_log_params[k] = v
                        except TypeError:
                            simple_log_params[k] = str(v)
                    logger.info(f"submit_answer完整参数(简化版): {json.dumps(simple_log_params, ensure_ascii=False)}")
                
                result = self._submit_answer(config)
                
                # 记录返回结果的摘要
                logger.info(f"submit_answer操作完成，返回结果状态: {result.get('status')}")
            # 处理end_quiz操作
            elif operation == "end_quiz":
                logger.info("执行end_quiz操作")
                result = self._end_quiz(config)
            else:
                logger.error(f"不支持的操作类型: {operation}")
                result = {
                    "quiz_id": getattr(config, "quiz_id", str(uuid.uuid4())),
                    "status": "error",
                    "message": f"不支持的操作: {operation}",
                    "words": None,
                    "total_questions": 0,
                    "answered_count": 0,
                    "correct_count": 0,
                    "wrong_answers": [],
                    "score_percentage": 0.0,
                    "current_batch": 0,
                    "total_batches": 0,
                    "formatted_question": f"错误：不支持的操作: {operation}"
                }
            
            # 记录除questions外的所有参数（太长了）
            log_params = {k: v for k, v in vars(config).items() if k != 'questions'}
            logger.debug(f"输入参数详情: {log_params}")
            
            # 确保所有必要的输出字段都存在
            self._ensure_complete_output(result)
            
            # 记录结果状态（不记录完整结果，可能太大）
            logger.info(f"执行结果: status={result.get('status')}, message={result.get('message')}")
            
            # 在每次操作后保存会话数据
            self.__class__.dump_sessions()
            
            return result
        except Exception as e:
            logger.error(f"执行过程中发生异常: {str(e)}")
            logger.error(traceback.format_exc())
            # 返回带有所有必要字段的错误结果
            return {
                "quiz_id": getattr(config, "quiz_id", str(uuid.uuid4())),
                "status": "error",
                "message": f"执行异常: {str(e)}",
                "words": None,
                "total_questions": 0,
                "answered_count": 0,
                "correct_count": 0,
                "wrong_answers": [],
                "score_percentage": 0.0,
                "current_batch": 0,
                "total_batches": 0,
                "formatted_question": f"错误：{str(e)}"
            }
    
    def _ensure_complete_output(self, result):
        """确保结果字典包含所有必要的输出字段"""
        required_fields = {
            "quiz_id": str(uuid.uuid4()),
            "status": "error",
            "message": "",
            "words": None,
            "total_questions": 0,
            "answered_count": 0,
            "correct_count": 0,
            "wrong_answers": [],
            "score_percentage": 0.0,
            "current_batch": 0,
            "total_batches": 0,
            "formatted_question": "",  # 添加默认值为空字符串
            "question_index": -1,  # 添加默认值为-1，表示无效索引
            "should_transition": False,  # 添加默认值为False
            "transition_to": None  # 添加默认值为None
        }
        
        for field, default_value in required_fields.items():
            if field not in result:
                logger.warning(f"输出缺少必要字段 {field}，添加默认值: {default_value}")
                result[field] = default_value
        
        return result
    
    def _prepare(self, config):
        """准备阶段：处理单词列表，随机乱序"""
        try:
            logger.info(f"准备阶段 - 处理单词列表")
            
            # 获取单词列表和问题列表
            word_list = getattr(config, 'word_list', '')
            questions_input = getattr(config, 'questions', [])
            
            # 简化模式检查 - 如果questions或word_list是简单的单词列表，直接使用
            simple_words = []
            
            # 检查questions_input是否为简单单词列表
            if isinstance(questions_input, list):
                # 检查第一个元素是否为字符串或简单对象
                if len(questions_input) > 0:
                    if isinstance(questions_input[0], str):
                        logger.info("检测到questions是简单单词列表格式")
                        simple_words = questions_input
                    elif isinstance(questions_input[0], dict) and 'word' in questions_input[0] and len(questions_input[0].keys()) == 1:
                        logger.info("检测到questions是简单单词对象列表格式")
                        simple_words = [q['word'] for q in questions_input]
            
            # 如果找到简单单词列表，直接使用它
            if simple_words:
                logger.info(f"使用简单单词列表，共{len(simple_words)}个单词")
                words = simple_words
                
                # 随机乱序
                random.shuffle(words)
                logger.info(f"单词列表已乱序，共 {len(words)} 个单词")
                
                # 生成quiz_id
                quiz_id = str(uuid.uuid4())
                logger.info(f"生成新的quiz_id: {quiz_id}")
                
                # 组合成逗号分隔的字符串
                words_str = ",".join(words)
                
                # 创建只包含单词的列表
                shuffled_questions = words.copy()
                
                # 保存到会话中
                self.__class__.sessions[quiz_id] = {
                    "shuffled_questions": shuffled_questions,  # 现在是单词的字符串数组
                    "original_words": words.copy(),  # 保存原始单词列表
                    "status": "not_started",
                    "answers": {},
                    "correct_count": 0,
                    "wrong_answers": [],
                    "current_batch_index": 0  # 初始化当前批次索引
                }
                
                logger.info(f"会话 {quiz_id} 已创建，包含 {len(shuffled_questions)} 个单词")
                
                return {
                    "quiz_id": quiz_id,
                    "status": "success",
                    "message": "单词列表已成功准备",
                    "words": words_str,
                    "formatted_question": "词汇列表已准备完成"
                }
            
            # 以下是原有复杂逻辑，保留用于兼容旧版本
            if not word_list:
                logger.error("单词列表为空且没有有效的题目结构")
                return {
                    "quiz_id": str(uuid.uuid4()),
                    "status": "error",
                    "message": "单词列表为空且没有有效的题目结构",
                    "words": None,
                    "formatted_question": "错误：单词列表为空"
                }
            
            # 将单词列表转换为列表
            words = []
            if isinstance(word_list, str):
                words = [w.strip() for w in word_list.split(",") if w.strip()]
            elif isinstance(word_list, list):
                words = [str(w).strip() for w in word_list if str(w).strip()]
            
            if not words:
                logger.error("处理后的单词列表为空")
                return {
                    "quiz_id": str(uuid.uuid4()),
                    "status": "error",
                    "message": "处理后的单词列表为空",
                    "words": None,
                    "formatted_question": "错误：处理后的单词列表为空"
                }
            
            # 随机乱序
            random.shuffle(words)
            logger.info(f"单词列表已乱序，共 {len(words)} 个单词")
            
            # 生成quiz_id
            quiz_id = str(uuid.uuid4())
            logger.info(f"生成新的quiz_id: {quiz_id}")
            
            # 组合成逗号分隔的字符串
            words_str = ",".join(words)
            
            # 使用简化的字符串数组格式
            shuffled_questions = words.copy()
            
            # 保存到会话中
            self.__class__.sessions[quiz_id] = {
                "shuffled_questions": shuffled_questions,  # 现在是单词的字符串数组
                "original_words": words.copy(),  # 保存原始单词列表（可选）
                "status": "not_started",
                "answers": {},  # 用户回答的题目
                "correct_count": 0,
                "wrong_answers": [],
                "current_batch_index": 0  # 初始化当前批次索引
            }
            
            # 记录会话字典状态
            session_count = len(self.__class__.sessions)
            logger.info(f"保存会话数据，当前共有 {session_count} 个会话")
            logger.info(f"会话 {quiz_id} 已创建，包含 {len(shuffled_questions)} 个单词")
            
            # 检查会话是否成功保存
            if quiz_id in self.__class__.sessions:
                logger.info(f"确认：会话 {quiz_id} 已成功保存到内存中")
            else:
                logger.error(f"错误：会话 {quiz_id} 未能成功保存到内存中")
            
            # 列出所有的quiz_id
            all_ids = list(self.__class__.sessions.keys())
            logger.info(f"当前所有会话ID: {all_ids}")
            
            return {
                "quiz_id": quiz_id,
                "status": "success",
                "message": "单词列表已成功乱序",
                "words": words_str,
                "formatted_question": "词汇列表已准备完成"
            }
        except Exception as e:
            logger.error(f"准备阶段异常: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "quiz_id": str(uuid.uuid4()),
                "status": "error",
                "message": f"准备阶段出错: {str(e)}",
                "words": None,
                "formatted_question": f"错误：{str(e)}"
            }
    
    def _get_batch_words(self, config):
        """获取一批乱序的单词"""
        try:
            logger.info(f"获取批次单词 - batch_index={config.batch_index}, batch_size={config.batch_size}")
            
            questions = config.questions
            batch_size = config.batch_size
            batch_index = config.batch_index
            quiz_id = config.quiz_id
            
            # 记录操作开始
            log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            logger.info(f"[{log_time}] 开始获取批次单词: batch_index={batch_index}, batch_size={batch_size}, quiz_id={quiz_id}")
            
            if not questions and not quiz_id:
                logger.error("题目列表为空且未提供有效的quiz_id")
                return {
                    "quiz_id": str(uuid.uuid4()),
                    "status": "error",
                    "message": "题目列表为空且未提供有效的quiz_id",
                    "words": None,
                    "total_questions": 0,
                    "answered_count": 0,
                    "correct_count": 0,
                    "wrong_answers": [],
                    "score_percentage": 0.0,
                    "current_batch": 0,
                    "total_batches": 0
                }
            
            # 如果提供了quiz_id，尝试获取会话
            if quiz_id and quiz_id in self.__class__.sessions:
                logger.info(f"使用现有会话: quiz_id={quiz_id}")
                session = self.__class__.sessions[quiz_id]
                shuffled_questions = session["shuffled_questions"]
                logger.info(f"找到现有会话，总题目数: {len(shuffled_questions)}")
            else:
                # 创建新会话并乱序题目
                if quiz_id:
                    logger.info(f"未找到quiz_id={quiz_id}的会话，创建新会话")
                else:
                    quiz_id = str(uuid.uuid4())
                    logger.info(f"未提供quiz_id，生成新ID: {quiz_id}")
                
                shuffled_questions = questions.copy()
                logger.info(f"题目列表复制完成，题目数: {len(shuffled_questions)}")
                
                random.shuffle(shuffled_questions)
                logger.info("题目列表乱序完成")
                
                self.__class__.sessions[quiz_id] = {
                    "shuffled_questions": shuffled_questions,
                    "status": "not_started",
                    "answers": {},  # 用户回答的题目
                    "correct_count": 0,
                    "wrong_answers": []
                }
                logger.info(f"新会话创建完成: quiz_id={quiz_id}")
            
            # 计算总批次数和验证批次索引
            total_questions = len(shuffled_questions)
            total_batches = (total_questions + batch_size - 1) // batch_size
            
            logger.info(f"计算批次信息: total_questions={total_questions}, batch_size={batch_size}, total_batches={total_batches}")
            
            if batch_index < 0 or batch_index >= total_batches:
                logger.error(f"批次索引无效: batch_index={batch_index}, 有效范围: 0-{total_batches-1}")
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": f"批次索引无效，有效范围: 0-{total_batches-1}",
                    "words": None,
                    "total_questions": total_questions,
                    "answered_count": 0,
                    "correct_count": 0,
                    "wrong_answers": [],
                    "score_percentage": 0.0,
                    "current_batch": batch_index,
                    "total_batches": total_batches
                }
            
            # 获取当前批次的题目
            start_index = batch_index * batch_size
            end_index = min(start_index + batch_size, total_questions)
            logger.info(f"计算当前批次范围: start_index={start_index}, end_index={end_index}")
            
            current_batch_questions = shuffled_questions[start_index:end_index]
            logger.info(f"获取当前批次题目，数量: {len(current_batch_questions)}")
            
            # 提取单词并用逗号连接
            words = ",".join([q["word"] for q in current_batch_questions])
            logger.info(f"提取单词完成，单词列表: {words}")
            
            result = {
                "quiz_id": quiz_id,
                "status": "success",
                "message": f"成功获取第{batch_index+1}批单词",
                "words": words,
                "current_batch": batch_index,
                "total_batches": total_batches,
                "total_questions": total_questions,
                "answered_count": len(self.__class__.sessions[quiz_id]["answers"]) if quiz_id in self.__class__.sessions else 0,
                "correct_count": self.__class__.sessions[quiz_id]["correct_count"] if quiz_id in self.__class__.sessions else 0,
                "wrong_answers": self.__class__.sessions[quiz_id]["wrong_answers"] if quiz_id in self.__class__.sessions else [],
                "score_percentage": 0.0  # 在这里计算百分比
            }
            
            # 计算得分百分比
            if result["answered_count"] > 0:
                result["score_percentage"] = (result["correct_count"] / result["answered_count"]) * 100
                
            logger.info(f"get_batch_words执行完成: quiz_id={quiz_id}, status=success")
            return result
            
        except Exception as e:
            logger.error(f"_get_batch_words执行异常: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "quiz_id": quiz_id if 'quiz_id' in locals() else str(uuid.uuid4()),
                "status": "error",
                "message": f"获取批次单词时出错: {str(e)}",
                "words": None,
                "total_questions": 0,
                "answered_count": 0,
                "correct_count": 0,
                "wrong_answers": [],
                "score_percentage": 0.0,
                "current_batch": 0,
                "total_batches": 0
            }
    
    def _start_quiz(self, config):
        """开始一个新的测验会话"""
        try:
            quiz_id = config.quiz_id
            logger.info(f"开始测验: quiz_id={quiz_id}")
            index_logger.info(f"==================== 开始测验 ====================")
            index_logger.info(f"开始测验: quiz_id={quiz_id}")
            
            # 记录当前会话状态
            session_count = len(self.__class__.sessions)
            all_ids = list(self.__class__.sessions.keys())
            logger.info(f"当前共有 {session_count} 个会话")
            logger.info(f"所有会话ID: {all_ids}")
            index_logger.info(f"当前共有 {session_count} 个会话")
            index_logger.info(f"所有会话ID: {all_ids}")
            
            # 验证quiz_id是否有效
            if not quiz_id:
                logger.error("提供的quiz_id为空")
                index_logger.error("提供的quiz_id为空")
                return {
                    "quiz_id": str(uuid.uuid4()),
                    "status": "error",
                    "message": "提供的quiz_id为空",
                    "words": None,
                    "total_questions": 0,
                    "answered_count": 0,
                    "correct_count": 0,
                    "wrong_answers": [],
                    "score_percentage": 0.0,
                    "current_batch": 0,
                    "total_batches": 0,
                    "formatted_question": "错误：提供的quiz_id为空"
                }
            
            # 检查quiz_id是否存在于会话字典中
            if quiz_id not in self.__class__.sessions:
                logger.error(f"未找到有效的测验会话: quiz_id={quiz_id}")
                logger.info(f"检查是否存在数据文件，尝试加载...")
                index_logger.error(f"未找到有效的测验会话: quiz_id={quiz_id}")
                index_logger.info(f"检查是否存在数据文件，尝试加载...")
                
                # 尝试再次加载数据以确认
                loaded = self.__class__.load_sessions()
                logger.info(f"加载结果: {loaded}")
                index_logger.info(f"加载结果: {loaded}")
                
                # 再次检查quiz_id是否存在
                if quiz_id in self.__class__.sessions:
                    logger.info(f"加载后找到了会话 {quiz_id}")
                    index_logger.info(f"加载后找到了会话 {quiz_id}")
                else:
                    logger.error(f"加载后仍未找到会话 {quiz_id}")
                    index_logger.error(f"加载后仍未找到会话 {quiz_id}")
                return {
                        "quiz_id": quiz_id,
                    "status": "error",
                    "message": "未找到有效的测验会话",
                    "words": None,
                    "total_questions": 0,
                    "answered_count": 0,
                    "correct_count": 0,
                    "wrong_answers": [],
                    "score_percentage": 0.0,
                    "current_batch": 0,
                        "total_batches": 0,
                        "formatted_question": "错误：未找到有效的测验会话"
                }
            
            # 检查是否传入了新的问题集
            if hasattr(config, 'questions') and config.questions and len(config.questions) > 0:
                user_questions = None
                if isinstance(config.questions, list):
                    user_questions = config.questions
                    index_logger.info(f"发现用户传入questions，数量: {len(user_questions)}")
                elif isinstance(config.questions, str):
                    try:
                        parsed_questions = json.loads(config.questions)
                        if isinstance(parsed_questions, list):
                            user_questions = parsed_questions
                            index_logger.info(f"解析字符串得到questions，数量: {len(user_questions)}")
                    except Exception as e:
                        index_logger.error(f"解析questions字符串失败: {str(e)}")
                
                # 保存新的问题集到会话
                if user_questions is not None:
                    index_logger.info(f"使用新的问题集，数量: {len(user_questions)}")
                    session = self.__class__.sessions[quiz_id]
                    session["current_questions"] = user_questions
            
            # 更新会话状态
            session = self.__class__.sessions[quiz_id]
            old_status = session["status"]
            session["status"] = "in_progress"
            logger.info(f"会话状态从 {old_status} 更新为 in_progress")
            index_logger.info(f"会话状态从 {old_status} 更新为 in_progress")
            
            # 重置答案记录
            session["answers"] = {}
            session["correct_count"] = 0
            session["wrong_answers"] = []
            
            # 重置索引追踪
            session["current_question_index"] = None
            session["next_question_index"] = None
            session["next_batch_index"] = None
            session["current_batch_index"] = 0  # 初始化当前批次索引为0
            logger.info("重置答案记录和索引追踪")
            index_logger.info("重置答案记录和索引追踪数据，批次索引设为0")
            
            # 获取题目集合
            questions_to_use = session.get("current_questions", session.get("shuffled_questions", []))
            total_questions = len(questions_to_use)
            logger.info(f"测验题目总数: {total_questions}")
            index_logger.info(f"测验题目总数: {total_questions}")
            
            # 保存更新的会话状态
            self.__class__.sessions[quiz_id] = session
            logger.info(f"会话 {quiz_id} 状态已更新并保存")
            index_logger.info(f"会话 {quiz_id} 状态已更新并保存")
            index_logger.info(f"==================== 开始测验完成 ====================")
            
            return {
                "quiz_id": quiz_id,
                "status": "in_progress",
                "message": "测验开始",
                "words": None,
                "total_questions": total_questions,
                "answered_count": 0,
                "correct_count": 0,
                "wrong_answers": [],
                "score_percentage": 0.0,
                "current_batch": 0,
                "total_batches": (total_questions + config.batch_size - 1) // config.batch_size,
                "formatted_question": "测验已开始，请获取题目"
            }
        except Exception as e:
            logger.error(f"_start_quiz执行异常: {str(e)}")
            logger.error(traceback.format_exc())
            index_logger.error(f"_start_quiz执行异常: {str(e)}")
            index_logger.error(traceback.format_exc())
            return {
                "quiz_id": quiz_id if 'quiz_id' in locals() else "unknown",
                "status": "error",
                "message": f"提交答案时出错: {str(e)}",
                "formatted_message": f"错误：{str(e)}"
            }
    
    def _end_quiz(self, config):
        """结束测验并返回最终结果"""
        try:
            quiz_id = config.quiz_id
            logger.info(f"结束测验: quiz_id={quiz_id}")
            
            # 验证quiz_id是否有效
            if not quiz_id or quiz_id not in self.__class__.sessions:
                logger.error(f"未找到有效的测验会话: quiz_id={quiz_id}")
                return {
                    "quiz_id": quiz_id or str(uuid.uuid4()),
                    "status": "error",
                    "message": "未找到有效的测验会话",
                    "words": None,
                    "total_questions": 0,
                    "answered_count": 0,
                    "correct_count": 0,
                    "wrong_answers": [],
                    "score_percentage": 0.0,
                    "current_batch": 0,
                    "total_batches": 0,
                    "formatted_question": "错误：未找到有效的测验会话"
                }
            
            # 获取会话
            session = self.__class__.sessions[quiz_id]
            old_status = session["status"]
            session["status"] = "completed"
            logger.info(f"会话状态从 {old_status} 更新为 completed")
            
            # 计算统计信息
            total_questions = len(session["shuffled_questions"])
            answered_count = len(session["answers"])
            correct_count = session["correct_count"]
            score_percentage = (correct_count / answered_count) * 100 if answered_count > 0 else 0
            
            logger.info(f"测验统计: total_questions={total_questions}, answered_count={answered_count}, correct_count={correct_count}, score_percentage={score_percentage}")
            
            batch_size = config.batch_size
            total_batches = (total_questions + batch_size - 1) // batch_size
            
            result = {
                "quiz_id": quiz_id,
                "status": "completed",
                "message": f"测验完成，正确率: {round(score_percentage, 2)}%",
                "words": None,
                "total_questions": total_questions,
                "answered_count": answered_count,
                "correct_count": correct_count,
                "score_percentage": round(score_percentage, 2),
                "wrong_answers": session["wrong_answers"],
                "current_batch": 0,
                "total_batches": total_batches,
                "formatted_question": f"测验已完成！\n正确率: {round(score_percentage, 2)}%\n正确题数: {correct_count}/{answered_count}"
            }
            
            logger.info(f"测验结束，返回结果: status={result['status']}, score_percentage={result['score_percentage']}")
            return result
        except Exception as e:
            logger.error(f"_end_quiz执行异常: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "quiz_id": quiz_id if 'quiz_id' in locals() else str(uuid.uuid4()),
                "status": "error",
                "message": f"结束测验时出错: {str(e)}",
                "words": None,
                "total_questions": 0,
                "answered_count": 0,
                "correct_count": 0,
                "wrong_answers": [],
                "score_percentage": 0.0,
                "current_batch": 0,
                "total_batches": 0,
                "formatted_question": f"错误：{str(e)}"
            }
    
    def _get_next_batch(self, config):
        """获取下一批单词"""
        try:
            logger.info(f"获取下一批单词 - batch_size={config.batch_size}")
            index_logger.info(f"==================== 获取下一批单词开始 ====================")
            transition_logger.info(f"==================== _get_next_batch 开始 ====================")
            
            quiz_id = config.quiz_id
            batch_size = config.batch_size
            
            # 验证quiz_id
            if not quiz_id or quiz_id not in self.__class__.sessions:
                logger.error(f"未找到有效的测验会话: quiz_id={quiz_id}")
                index_logger.error(f"未找到有效的测验会话: quiz_id={quiz_id}")
                return {
                    "quiz_id": str(uuid.uuid4()),
                    "status": "error",
                    "message": "未找到有效的测验会话",
                    "words": None,
                    "formatted_question": "错误：未找到有效的测验会话"
                }
            
            session = self.__class__._quiz_sessions[quiz_id]
            index_logger.info(f"会话状态: {session.get('status', '未知')}")
            
            # 获取answers记录，判断是否所有问题都已回答
            answers = session["answers"]
            index_logger.info(f"当前已有回答数: {len(answers)}, 回答索引: {list(answers.keys())}")
            
            # 确定当前批次索引，优先级：
            # 1. 会话中的next_batch_index（如果存在）
            # 2. 用户指定的batch_index（如果提供）
            # 3. 会话中的current_batch_index
            # 4. 默认值0
            
            batch_index = 0  # 默认值
            
            # 优先从会话中获取保存的下一批次索引
            next_batch_index_from_session = session.get("next_batch_index", None)
            current_batch_index_from_session = session.get("current_batch_index", 0)
            user_batch_index = getattr(config, 'batch_index', None)
            
            # 确保用户批次索引不是空字符串
            if user_batch_index == '':
                user_batch_index = None
                index_logger.info("处理空字符串batch_index为None")
            
            index_logger.info(f"会话中的下一批次索引: {next_batch_index_from_session}")
            index_logger.info(f"会话中的当前批次索引: {current_batch_index_from_session}")
            index_logger.info(f"用户提供的批次索引: {user_batch_index}")
            
            if next_batch_index_from_session is not None:
                batch_index = next_batch_index_from_session
                index_logger.info(f"使用会话中的下一批次索引: {batch_index}")
                # 已使用会话中的next_batch_index，重置
                session["next_batch_index"] = None
            elif user_batch_index is not None:
                batch_index = user_batch_index
                index_logger.info(f"使用用户提供的批次索引: {batch_index}")
            else:
                batch_index = current_batch_index_from_session
                index_logger.info(f"使用会话中的当前批次索引: {batch_index}")
            
            logger.info(f"当前批次索引: {batch_index}")
            index_logger.info(f"最终确定的批次索引: {batch_index}")
            
            # 检查会话中是否已保存问题集
            saved_questions = session.get("current_questions", None)
            if saved_questions:
                index_logger.info(f"从会话中获取到保存的问题集，数量: {len(saved_questions)}")
            
            # 检查用户是否传入了新的问题列表
            user_questions = None
            if hasattr(config, 'questions') and config.questions and len(config.questions) > 0:
                logger.info(f"用户传入新的问题列表，数量: {len(config.questions) if isinstance(config.questions, list) else '未知'}")
                index_logger.info(f"用户传入新的问题列表，数量: {len(config.questions) if isinstance(config.questions, list) else '未知'}")
                # 检查questions的格式
                if isinstance(config.questions, list):
                    user_questions = config.questions
                    logger.info(f"使用用户传入的questions，总题目数: {len(user_questions)}")
                    index_logger.info(f"使用用户传入的questions，总题目数: {len(user_questions)}")
                elif isinstance(config.questions, str):
                    try:
                        # 尝试解析JSON字符串
                        parsed_questions = json.loads(config.questions)
                        if isinstance(parsed_questions, list):
                            user_questions = parsed_questions
                            logger.info(f"解析字符串得到questions，总题目数: {len(user_questions)}")
                            index_logger.info(f"解析字符串得到questions，总题目数: {len(user_questions)}")
                    except Exception as e:
                        logger.error(f"解析questions字符串失败: {str(e)}")
                        index_logger.error(f"解析questions字符串失败: {str(e)}")
            
            # 确定使用哪个问题集
            questions_to_use = None
            if user_questions is not None:
                questions_to_use = user_questions
                index_logger.info("使用用户传入的问题集")
                # 保存到会话中，以便后续使用
                session["current_questions"] = user_questions
                
                # 如果传入了新的问题集，对当前批次的答题记录进行重置
                # 计算批次范围
                start_index = batch_index * batch_size
                end_index = min(start_index + batch_size, len(user_questions))
                # 清除当前批次范围内的回答记录
                new_answers = {}
                for idx, ans in answers.items():
                    if idx < start_index or idx >= end_index:
                        new_answers[idx] = ans
                old_count = len(answers)
                new_count = len(new_answers)
                if old_count != new_count:
                    index_logger.info(f"清除当前批次的回答记录: 原有{old_count}个 -> 现有{new_count}个")
                    session["answers"] = new_answers
                    answers = new_answers
                    # 更新正确答题数
                    correct_count = sum(1 for ans in answers.values() if ans.get("is_correct", False))
                    session["correct_count"] = correct_count
                    index_logger.info(f"更新正确答题数: {correct_count}")
            elif saved_questions is not None:
                questions_to_use = saved_questions
                index_logger.info("使用会话中保存的问题集")
            else:
                index_logger.error("没有找到可用的问题集")
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": "请提供questions参数",
                    "words": None,
                    "formatted_question": "错误：请提供questions参数"
                }
            
            # 首先判断是否所有问题都已回答
            total_questions = len(questions_to_use)
            
            # 计算总批次数和当前批次范围
            total_batches = (total_questions + batch_size - 1) // batch_size
            start_index = batch_index * batch_size
            end_index = min(start_index + batch_size, total_questions)
            
            index_logger.info(f"批次计算: total_questions={total_questions}, batch_size={batch_size}, total_batches={total_batches}")
            index_logger.info(f"当前批次范围: start_index={start_index}, end_index={end_index}")
            index_logger.info(f"[DEBUG] 详细批次信息: batch_index={batch_index}, batch_size={batch_size}, start_index={start_index}, end_index={end_index}, total_questions={total_questions}, total_batches={total_batches}")
            
            transition_logger.info(f"批次信息: batch_index={batch_index}/{total_batches-1}, range={start_index}-{end_index}, total={total_questions}")
            
            # 确保批次索引在有效范围内
            if batch_index < 0 or batch_index >= total_batches:
                logger.error(f"批次索引超出范围: {batch_index}, 有效范围: 0-{total_batches-1}")
                index_logger.error(f"批次索引超出范围: {batch_index}, 有效范围: 0-{total_batches-1}")
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": f"批次索引超出范围: {batch_index}, 有效范围: 0-{total_batches-1}",
                    "words": None,
                    "formatted_question": f"错误：批次索引超出范围"
                }
            
            # 判断是否所有问题都已回答
            if len(answers) >= total_questions:
                index_logger.info(f"所有问题都已回答，当前答题数: {len(answers)}, 总题数: {total_questions}")
                index_logger.info(f"[DEBUG] 所有问题已回答判断: answers数量({len(answers)}) >= 总题目数({total_questions})")
                transition_logger.info(f"所有题目已回答: {len(answers)}/{total_questions}")
                return {
                    "quiz_id": quiz_id,
                    "status": "completed",
                    "message": "所有问题都已回答",
                    "words": None,
                    "total_questions": total_questions,
                    "answered_count": len(answers),
                    "correct_count": session["correct_count"],
                    "wrong_answers": session.get("wrong_answers", []),
                    "score_percentage": (session["correct_count"] / len(answers)) * 100 if len(answers) > 0 else 0,
                    "current_batch": batch_index,
                    "batch_index": batch_index,
                    "batch_size": batch_size,
                    "total_batches": total_batches,
                    "formatted_question": "所有问题都已回答完成！"
                }
            
            # 更新会话状态为"in_progress"
            session["status"] = "in_progress"
            
            # 更新会话中的当前批次索引
            session["current_batch_index"] = batch_index
            logger.info(f"更新会话状态为 in_progress")
            index_logger.info(f"更新会话状态为 in_progress，当前批次索引更新为 {batch_index}")
            
            # 检查当前批次中的问题是否已经全部回答
            batch_answered = True
            for i in range(start_index, end_index):
                global_index = i  # 全局索引就是i
                if global_index not in answers:
                    batch_answered = False
                    index_logger.info(f"[DEBUG] 当前批次未全部回答，索引 {global_index} 未答")
                    transition_logger.info(f"当前批次未全部回答，索引 {global_index} 未回答")
                    break
            
            index_logger.info(f"[DEBUG] 当前批次是否已全部回答: {batch_answered}, 批次索引: {batch_index}")
            transition_logger.info(f"当前批次{batch_index}是否已全部回答: {batch_answered}")
            
            # 如果当前批次已完成但还有下一批，返回批次完成状态
            if batch_answered:
                index_logger.info(f"当前批次{batch_index}的问题已全部回答，检查是否有下一批")
                index_logger.info(f"[DEBUG] 当前批次全部回答完成，检查是否有下一批")
                
                # 检查是否有下一批
                has_next_batch = end_index < total_questions
                index_logger.info(f"[DEBUG] 是否有下一批判断: end_index({end_index}) < total_questions({total_questions}) = {has_next_batch}")
                
                if has_next_batch:
                    next_batch_index = batch_index + 1
                    index_logger.info(f"存在下一批，批次索引: {next_batch_index}")
                    index_logger.info(f"[DEBUG] 有下一批次，索引: {next_batch_index}")
                    
                    # 保存下一批次索引到会话
                    session["next_batch_index"] = next_batch_index
                    
                    # 更新会话中的当前批次索引为下一批
                    session["current_batch_index"] = next_batch_index
                    index_logger.info(f"自动更新会话中的当前批次索引为下一批: {next_batch_index}")
                    index_logger.info(f"[DEBUG] 更新会话: next_batch_index={next_batch_index}, current_batch_index={next_batch_index}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "batch_completed",
                        "message": f"当前批次已完成，请使用批次索引{next_batch_index}获取下一批",
                        "words": None,
                        "total_questions": total_questions,
                        "answered_count": len(answers),
                        "correct_count": session["correct_count"],
                        "wrong_answers": session.get("wrong_answers", []),
                        "score_percentage": (session["correct_count"] / len(answers)) * 100 if len(answers) > 0 else 0,
                        "current_batch": batch_index,
                        "batch_index": batch_index,
                        "batch_size": batch_size,
                        "next_batch_index": next_batch_index,
                        "total_batches": total_batches,
                        "formatted_question": f"当前批次已完成，请使用批次索引{next_batch_index}获取下一批"
                    }
                else:
                    index_logger.info("已是最后一批，且所有问题都已回答")
                    index_logger.info(f"[DEBUG] 没有下一批次，end_index({end_index}) >= total_questions({total_questions})")
                return {
                    "quiz_id": quiz_id,
                    "status": "completed",
                    "message": "所有问题都已回答",
                    "words": None,
                    "total_questions": total_questions,
                    "answered_count": len(answers),
                    "correct_count": session["correct_count"],
                    "wrong_answers": session.get("wrong_answers", []),
                    "score_percentage": (session["correct_count"] / len(answers)) * 100 if len(answers) > 0 else 0,
                    "current_batch": batch_index,
                    "batch_index": batch_index,
                    "batch_size": batch_size,
                    "total_batches": total_batches,
                    "formatted_question": "所有问题都已回答完成！"
                }
            
            # 提取当前批次的问题
            current_batch_questions = questions_to_use[start_index:end_index]
            logger.info(f"当前批次问题数量: {len(current_batch_questions)}")
            index_logger.info(f"当前批次问题数量: {len(current_batch_questions)}")
            
            # 如果当前批次没有问题，返回错误
            if not current_batch_questions:
                logger.error(f"当前批次没有问题，批次索引: {batch_index}, 总题数: {total_questions}")
                index_logger.error(f"当前批次没有问题，批次索引: {batch_index}, 总题数: {total_questions}")
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": f"当前批次{batch_index}没有问题，请检查批次索引",
                    "words": None,
                    "formatted_question": f"错误：当前批次{batch_index}没有问题，请检查批次索引"
                }
            
            # 记录获取的单词列表
            words_list = [q.get("word", "") for q in current_batch_questions]
            words = ",".join(words_list)
            logger.info(f"获取的单词数量: {len(words_list)}, 单词列表: {words_list}")
            index_logger.info(f"获取的单词数量: {len(words_list)}, 单词列表: {words_list}")
            
            # 计算分数百分比
            score_percentage = (session["correct_count"] / len(answers)) * 100 if len(answers) > 0 else 0
            
            # 计算当前批次中第一个未回答问题的全局索引
            first_unanswered_index = -1
            for i in range(start_index, end_index):
                if i not in answers:
                    first_unanswered_index = i
                    break
            
            logger.info(f"批次中第一个未回答问题的全局索引: {first_unanswered_index}")
            index_logger.info(f"批次中第一个未回答问题的全局索引: {first_unanswered_index}")
            
            # 如果找到了未回答的问题，更新当前问题索引
            if first_unanswered_index >= 0:
                session["current_question_index"] = first_unanswered_index
                index_logger.info(f"更新当前问题索引: current_question_index={first_unanswered_index}")
            
            index_logger.info(f"==================== 获取下一批单词结束 ====================")
            transition_logger.info(f"==================== _get_next_batch 结束 ====================")
            
            return {
                "quiz_id": quiz_id,
                "status": "in_progress",
                "message": "成功获取下一批单词",
                "words": words,
                "total_questions": total_questions,
                "answered_count": len(answers),
                "correct_count": session["correct_count"],
                "wrong_answers": session.get("wrong_answers", []),
                "score_percentage": score_percentage,
                "current_batch": batch_index,
                "batch_index": batch_index,
                "batch_size": batch_size,
                "total_batches": total_batches,
                "current_batch_indices": list(range(start_index, end_index)),
                "first_unanswered_index": first_unanswered_index,
                "formatted_question": f"已获取{len(words_list)}个单词：" + words
            }
        except Exception as e:
            logger.error(f"获取下一批单词异常: {str(e)}")
            logger.error(traceback.format_exc())
            index_logger.error(f"获取下一批单词异常: {str(e)}")
            index_logger.error(traceback.format_exc())
            transition_logger.error(f"_get_next_batch执行异常: {str(e)}")
            transition_logger.error(traceback.format_exc())
            return {
                "quiz_id": quiz_id if 'quiz_id' in locals() else str(uuid.uuid4()),
                "status": "error",
                "message": f"获取下一批单词时出错: {str(e)}",
                "words": None,
                "formatted_question": f"错误：{str(e)}"
            }
    
    def _get_next_question(self, config):
        try:
            # 获取会话ID和相关信息
            quiz_id = config.quiz_id
            
            # 为调试批次和索引问题创建特殊logger
            transition_logger = logging.getLogger("TransitionDebug")
            transition_logger.info(f"==================== _get_next_question 开始 ====================")
            
            # 获取批次索引和大小
            batch_index = config.batch_index if hasattr(config, "batch_index") else 0
            batch_size = config.batch_size if hasattr(config, "batch_size") else 5
            
            # 确保会话存在，并获取会话信息
            if quiz_id not in self.__class__._quiz_sessions:
                # 尝试加载会话数据
                self.__class__.load_sessions()
                
                # 再次检查会话是否存在
                if quiz_id not in self.__class__._quiz_sessions:
                    transition_logger.info(f"批次信息: batch_index={batch_index}/-1, range={batch_index*batch_size}-{batch_index*batch_size}, total=0")
                    return {
                        "quiz_id": quiz_id,
                        "status": "error",
                        "message": "未找到有效的测验会话",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 获取会话信息
            session = self.__class__._quiz_sessions[quiz_id]
            
            # 输出完整会话信息以便调试
            #index_logger.info(f"会话信息: {session}")
            
            # 获取原始问题列表（未打乱的）
            original_questions = session.get("original_questions", [])
            total_original_questions = len(original_questions)
            total_original_batches = (total_original_questions + batch_size - 1) // batch_size
            
            # 获取已经打乱的问题列表
            shuffled_questions = session.get("shuffled_questions", [])
            
            # 获取当前批次范围
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, total_original_questions)
            
            # 详细记录计算过程
            transition_logger.info(f"批次计算: batch_index={batch_index}, batch_size={batch_size}, start={start_index}, end={end_index}, total={total_original_questions}, total_batches={total_original_batches}")
            transition_logger.info(f"原始题目数: {total_original_questions}, 原始总批次: {total_original_batches}")
            
            # 获取当前批次的问题
            questions_to_use = original_questions[start_index:end_index] if start_index < len(original_questions) else []
            
            # 获取已回答的问题索引
            answers = session.get("answers", {})
            answered_indices = list(map(int, answers.keys()))
            
            transition_logger.info(f"答题进度: {len(answered_indices)}/{total_original_questions} = {(len(answered_indices)/total_original_questions*100 if total_original_questions > 0 else 0):.1f}%")
            transition_logger.info(f"已回答题目索引: {answered_indices}")
            
            # 计算当前批次内已回答问题的数量
            batch_answered_count = sum(1 for i in range(start_index, end_index) if i in answered_indices)
            batch_total = end_index - start_index
            
            # 检查当前批次是否已全部回答
            batch_fully_answered = batch_answered_count >= batch_total
            
            # 如果当前批次无问题，设置为已完成
            if batch_total == 0:
                batch_fully_answered = True
                
            transition_logger.info(f"当前批次{batch_index}是否已全部回答: {batch_fully_answered}")
            
            # 添加额外的批次边界日志
            transition_logger.info(f"批次边界: end={end_index}, total={total_original_questions}, batch_index={batch_index}/{total_original_batches-1 if total_original_batches > 0 else '?'}")
            
            # 所有题目已经回答完毕或当前批次没有题目
            all_answered = len(answered_indices) >= total_original_questions or total_original_questions == 0
            transition_logger.info(f"所有题目已回答检查: {len(answered_indices)}/{total_original_questions} = {all_answered}")
            
            # 检查所有题目是否都已回答
            if all_answered:
                transition_logger.info(f"设置transition_to=quiz_completed: 所有题目都已回答完成")
                
                # 计算得分
                correct_count = session.get("correct_count", 0)
                total_answered = len(session.get("answers", {}))
                score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                
                # 将会话状态设置为已完成
                session["status"] = "completed"
                
                # 更新会话信息
                self.__class__._quiz_sessions[quiz_id] = session
                self.__class__.dump_sessions()
                
                # 返回测验完成状态
                return {
                    "quiz_id": quiz_id,
                    "status": "completed",
                    "message": "测验已完成",
                    "question": None,
                    "question_index": -1,
                    "words": None,
                    "total_questions": total_original_questions,
                    "answered_count": total_answered,
                    "correct_count": correct_count,
                    "score_percentage": score_percentage,
                    "should_transition": True,
                    "transition_to": "quiz_completed"
                }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                transition_logger.info(f"下一批判断: 当前批次={batch_index}, 总批次={total_original_batches}, 是否有下一批={has_next_batch}")
                transition_logger.info(f"当前批次结束索引={end_index}, 原始总题目={total_original_questions}")
                
                next_start_index = (batch_index + 1) * batch_size
                in_range = next_start_index < total_original_questions
                transition_logger.info(f"下一批起始索引={next_start_index}, 是否在范围内={in_range}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = batch_index + 1
                    transition_logger.info(f"有下一批次: {next_batch_index}，设置transition_to=get_next_batch")
                    transition_logger.info(f"下一批范围: {next_start_index}-{min((batch_index + 2) * batch_size, total_original_questions)}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": len(answered_indices),
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / len(answered_indices)) * 100 if len(answered_indices) > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: batch_index({batch_index}) >= total_original_batches({total_original_batches})")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__._quiz_sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_original_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
                
                # 判断是否有下一批
                has_next_batch = end_index < total_original_questions and batch_index < total_original_batches - 1
                
                "quiz_id": quiz_id if 'quiz_id' in locals() else "unknown",
                "status": "error",
                "message": f"提交答案时出错: {str(e)}",
                "formatted_message": f"错误：{str(e)}"
            }