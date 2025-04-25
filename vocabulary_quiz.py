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
        # logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocabulary_quiz.log'))  # 注释掉文件输出
    ]
)
logger = logging.getLogger('VocabularyQuizWidget')

# 创建索引问题专用日志
index_logger = logging.getLogger('VocabularyQuizIndex')
index_logger.setLevel(logging.DEBUG)
# 确保索引日志不会传播到父日志处理器
index_logger.propagate = False
# 注释掉文件处理器
# index_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index_debug.log')
# index_file_handler = logging.FileHandler(index_log_file)
# index_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# index_logger.addHandler(index_file_handler)

# 创建专门针对transition_to问题的日志记录器
transition_logger = logging.getLogger('TransitionDebug')
transition_logger.setLevel(logging.DEBUG)
# 确保日志不会传播到父日志处理器
transition_logger.propagate = False
# 注释掉文件处理器
# transition_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transition_debug.log')
# transition_file_handler = logging.FileHandler(transition_log_file)
# transition_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
# transition_logger.addHandler(transition_file_handler)

# 添加控制台处理器，以便可以在控制台查看日志（可选）
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
index_logger.addHandler(console_handler)
transition_logger.addHandler(console_handler)

# 添加一条启动日志，确认日志系统正常工作
logger.info("============== VocabularyQuizWidget 日志系统初始化 ==============")
index_logger.info("============== 索引问题专用日志初始化（已关闭文件输出）==============")
transition_logger.info("============== Transition问题专用日志初始化（已关闭文件输出）==============")

@WIDGETS.register_module()
class VocabularyQuizWidget(BaseWidget):
    """单词测验小部件，实现单词测验功能"""
    CATEGORY = "Custom Widgets/Education"
    NAME = "Vocabulary Quiz Widget"
    
    class InputsSchema(BaseWidget.InputsSchema):
        operation: str = Field(
            type="string",
            default="get_batch",
            description="操作类型：prepare（准备题目）, start_quiz（开始测验）, get_next_batch（获取下一批单词）, get_next_question（获取下一个问题）, submit_answer（提交答案）, end_quiz（结束测验）"
        )
        quiz_id: Optional[str] = Field(
            default=None,
            description="考试ID，如果提供则使用之前的测验会话，否则创建新会话"
        )
        questions: Union[str, List[Dict[str, Union[str, List[str], int]]]] = Field(
            type="string",
            default="[]",
            description="题目列表，可直接输入JSON格式，如[{'word': '单词', 'options': ['选项1', '选项2', '选项3', '选项4'], 'correct_index': 0}, ...]"
        )
        word_list: Union[str, List[str]] = Field(
            type="string",
            default="",
            description="单词列表，可以是逗号分隔的字符串('apple,book,computer')或字符串列表(['apple', 'book', 'computer'])"
        )
        batch_size: int = Field(
            type="integer",
            default=15,
            description="每批返回的题目数量"
        )
        answer: Optional[Union[str, int]] = Field(
            default=None,
            description="用户的答案，可以是选项序号(0-3)或选项字母(A-D)，用于submit_answer操作"
        )
        selected_index: Optional[str] = Field(
            default=None,
            description="用户选择的选项，传入'选项A'、'选项B'、'选项C'或'选项D'，分别对应索引0、1、2、3"
        )
        question_index: Optional[int] = Field(
            default=None,
            description="当前问题的索引，用于submit_answer操作（可选，系统会尝试使用会话中保存的当前问题索引）"
        )
        
        @validator('question_index', pre=True)
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
        quiz_id: str = Field(type="string", description="考试唯一标识")
        status: str = Field(type="string", description="操作状态：success, error, batch_completed, 或考试状态：not_started, in_progress, completed")
        message: Optional[str] = Field(type="string", description="状态消息，尤其是错误信息")
        words: Optional[str] = Field(type="string", description="当前批次的单词，逗号分隔")
        total_questions: Optional[int] = Field(type="integer", description="总题目数")
        answered_count: Optional[int] = Field(type="integer", description="已回答题目数")
        correct_count: Optional[int] = Field(type="integer", description="正确题目数")
        wrong_answers: Optional[List[str]] = Field(
            type="array", 
            items={"type": "string"},  # 添加items属性
            description="错误回答的单词列表，每个元素是一个单词字符串"
        )
        score_percentage: Optional[float] = Field(type="number", description="得分百分比")
        current_batch: Optional[int] = Field(type="integer", description="当前批次索引")
        total_batches: Optional[int] = Field(type="integer", description="总批次数")
        formatted_question: Optional[str] = Field(type="string", description="格式化后的问题文本，包含单词和ABCD选项，带HTML换行符")
        question_index: Optional[int] = Field(type="integer", description="当前问题在题目列表中的索引，用于提交答案")
        should_transition: Optional[bool] = Field(type="boolean", default=False, description="是否应该转换状态，为true时表示需要流转到其他状态")
        transition_to: Optional[str] = Field(type="string", default=None, description="应该流转到的目标状态名称，可能的值：quiz_completed, get_next_batch")
        
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
            
            # 确保wrong_answers字段始终是数组类型
            if isinstance(result.get("wrong_answers", []), str):
                # 如果是字符串，转换成数组
                if result["wrong_answers"]:
                    result["wrong_answers"] = result["wrong_answers"].split(",")
                else:
                    result["wrong_answers"] = []
            
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
            "wrong_answers": [],  # 确保是一个空数组
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
            
            # questions参数变为可选
            questions = getattr(config, 'questions', [])
            batch_size = config.batch_size
            batch_index = config.batch_index
            quiz_id = config.quiz_id
            
            # 记录操作开始
            log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            logger.info(f"[{log_time}] 开始获取批次单词: batch_index={batch_index}, batch_size={batch_size}, quiz_id={quiz_id}")
            
            if not quiz_id:
                logger.error("未提供有效的quiz_id")
                return {
                    "quiz_id": str(uuid.uuid4()),
                    "status": "error",
                    "message": "未提供有效的quiz_id",
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
            if quiz_id in self.__class__.sessions:
                logger.info(f"使用现有会话: quiz_id={quiz_id}")
                session = self.__class__.sessions[quiz_id]
                
                # 优先使用session中的shuffled_questions - 修改为更明确的优先顺序
                shuffled_questions = session.get("shuffled_questions", [])
                current_questions = session.get("current_questions", [])
                
                # 记录会话中题目集合的情况
                logger.info(f"会话状态摘要: shuffled_questions长度={len(shuffled_questions)}, current_questions长度={len(current_questions)}")
                
                # 优先使用shuffled_questions
                if shuffled_questions and len(shuffled_questions) > 0:
                    logger.info(f"优先使用会话中的shuffled_questions，总题目数: {len(shuffled_questions)}")
                elif current_questions and len(current_questions) > 0:
                    shuffled_questions = current_questions
                    logger.info(f"shuffled_questions为空，使用current_questions，题目数: {len(shuffled_questions)}")
                elif questions:  # 如果会话中没有题目但传入了questions，则使用传入的
                    shuffled_questions = questions
                    logger.info(f"会话中无题目，使用传入的questions，题目数: {len(shuffled_questions)}")
                
                logger.info(f"最终使用的题目集，总题目数: {len(shuffled_questions)}")
            else:
                # 创建新会话并乱序题目
                if quiz_id:
                    logger.info(f"未找到quiz_id={quiz_id}的会话，创建新会话")
                else:
                    quiz_id = str(uuid.uuid4())
                    logger.info(f"未提供quiz_id，生成新ID: {quiz_id}")
                
                if not questions:
                    logger.error("未提供题目列表且会话中无题目")
                    return {
                        "quiz_id": quiz_id,
                        "status": "error",
                        "message": "未提供题目列表且会话中无题目",
                        "words": None,
                        "total_questions": 0,
                        "answered_count": 0,
                        "correct_count": 0,
                        "wrong_answers": [],
                        "score_percentage": 0.0,
                        "current_batch": 0,
                        "total_batches": 0
                    }
                
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
            
            # 记录shuffled_questions的类型
            if len(shuffled_questions) > 0:
                logger.info(f"shuffled_questions类型: {type(shuffled_questions).__name__}, 第一个元素类型: {type(shuffled_questions[0]).__name__}")
            
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
            
            # 将当前批次题目保存到会话的current_questions中
            session = self.__class__.sessions[quiz_id]
            session["current_questions"] = current_batch_questions
            session["current_batch_index"] = batch_index
            logger.info(f"将当前批次题目保存到会话的current_questions, 数量: {len(current_batch_questions)}")
            
            # 提取单词并用逗号连接 - 增加对字符串数组的兼容性处理
            if len(current_batch_questions) > 0:
                if isinstance(current_batch_questions[0], dict) and 'word' in current_batch_questions[0]:
                    # 如果是字典数组，提取word字段
                    words = ",".join([q["word"] for q in current_batch_questions])
                    logger.info(f"从字典数组中提取单词，数量: {len(current_batch_questions)}")
                else:
                    # 如果是字符串数组，直接使用
                    words = ",".join(current_batch_questions)
                    logger.info(f"直接使用字符串数组，数量: {len(current_batch_questions)}")
            else:
                words = ""
                logger.warning("当前批次没有单词")
            
            logger.info(f"提取单词完成，单词列表: {words}")
            
            # 保存当前批次索引和范围到会话中
            session["batch_start_index"] = start_index
            session["batch_end_index"] = end_index
            session["batch_size"] = batch_size
            session["total_questions"] = total_questions  # 保存总题目数
            session["total_batches"] = total_batches      # 保存总批次数
            logger.info(f"保存批次信息到会话: batch_index={batch_index}, start={start_index}, end={end_index}, total_questions={total_questions}, total_batches={total_batches}")
            
            # 更新会话
            self.__class__.sessions[quiz_id] = session
            
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
            
            # 计算统计信息 - 确保使用current_questions或shuffled_questions
            questions_to_use = session.get("current_questions", session.get("shuffled_questions", []))
            total_questions = len(questions_to_use)
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
                "wrong_answers": session["wrong_answers"],  # 保持数组类型
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
            
            # 调试会话状态
            self.debug_session(quiz_id)
            
            # 使用_update_session_state方法更新会话状态
            session = self._update_session_state(quiz_id, config)
            if not session:
                logger.error(f"更新会话状态失败: quiz_id={quiz_id}")
                transition_logger.error(f"更新会话状态失败: quiz_id={quiz_id}")
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": "无法更新会话状态",
                    "words": None,
                    "formatted_question": "错误：无法更新会话状态"
                }
                
            # 检查当前批次是否已完成
            batch_fully_answered = session.get("batch_fully_answered", False)
            current_batch_index = session.get("current_batch_index", 0)
            total_batches = session.get("total_batches", 0)
            
            # 检查是否需要切换到下一批
            if batch_fully_answered and current_batch_index + 1 < total_batches:
                transition_logger.info(f"当前批次{current_batch_index}已完成，切换到下一批次{current_batch_index+1}, quiz_id={quiz_id}")
                
                # 更新为下一批次
                new_batch_index = current_batch_index + 1
                
                # 保存新的批次索引
                session = self._update_session_state(quiz_id, None, {
                    "current_batch_index": new_batch_index
                })
                
                if session:
                    current_batch_index = new_batch_index
                    transition_logger.info(f"成功更新批次索引: current_batch_index={current_batch_index}, quiz_id={quiz_id}")
                else:
                    transition_logger.error(f"更新批次索引失败, quiz_id={quiz_id}")
            
            # 确保session中的shuffled_questions被正确使用
            if "shuffled_questions" not in session:
                logger.error(f"会话中缺少shuffled_questions: quiz_id={quiz_id}")
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": "会话中缺少shuffled_questions",
                    "words": None,
                    "formatted_question": "错误：会话中缺少shuffled_questions"
                }
                
            shuffled_questions = session["shuffled_questions"]
            logger.info(f"从会话中获取shuffled_questions，数量: {len(shuffled_questions)}")
            transition_logger.info(f"从会话中获取shuffled_questions，数量: {len(shuffled_questions)}, 类型: {type(shuffled_questions)}")
            
            # 准备获取批次单词的数据
            from types import SimpleNamespace
            batch_config = SimpleNamespace(
                quiz_id=quiz_id,
                batch_size=batch_size,
                batch_index=current_batch_index,
                # 不再传递questions参数，让_get_batch_words自己从session中获取
            )
            
            # 添加日志，记录当前的session状态
            transition_logger.info(f"调用_get_batch_words前的会话状态: shuffled_questions长度={len(session.get('shuffled_questions', []))}, current_batch_index={current_batch_index}")
            
            # 使用_get_batch_words方法获取一批乱序的单词
            result = self._get_batch_words(batch_config)
            if result["status"] != "success":
                return result
            
            # 更新会话状态
            session["status"] = "in_progress"
            session["current_batch_index"] = result["current_batch"]
            session["next_batch_index"] = result["current_batch"] + 1 if result["current_batch"] < result["total_batches"] - 1 else None
            
            # 记录批次信息
            transition_logger.info(f"批次信息: batch_index={result['current_batch']}/{result['total_batches']-1 if result['total_batches'] > 0 else 0}, range={result['current_batch']*batch_size}-{min((result['current_batch']+1)*batch_size, result['total_questions'])}, quiz_id={quiz_id}")
            
            # 再次更新会话状态以保存当前批次索引
            self._update_session_state(quiz_id, None, {
                "current_batch_index": result["current_batch"]
            })
            
            # 返回结果
            result["status"] = "batch_completed"
            result["message"] = f"当前批次已完成，请使用批次索引{result['current_batch']}获取下一批"
            result["should_transition"] = True
            result["transition_to"] = "get_next_batch"
            
            logger.info(f"get_next_batch执行完成: quiz_id={quiz_id}, status=success")
            index_logger.info(f"==================== 获取下一批单词结束 ====================")
            transition_logger.info(f"==================== _get_next_batch 结束 ====================")
            return result
        except Exception as e:
            logger.error(f"_get_next_batch执行异常: {str(e)}")
            logger.error(traceback.format_exc())
            index_logger.error(f"_get_next_batch执行异常: {str(e)}")
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
            # 获取批次索引和大小 - 注：这两个参数只用于初始值，实际应该以会话中的值为准
            batch_index = config.batch_index if hasattr(config, "batch_index") else 0
            batch_size = config.batch_size if hasattr(config, "batch_size") else 5
            
            # 指定日志级别和格式
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
            logger = logging.getLogger("VocabularyQuizWidget._get_next_question")
            
            # 为调试批次和索引问题创建特殊logger
            index_logger = logging.getLogger("BatchIndexDebug")
            transition_logger = logging.getLogger("TransitionDebug")
            
            index_logger.info(f"[DEBUG] 开始_get_next_question，quiz_id={quiz_id}, batch_index={batch_index}, batch_size={batch_size}")
            
            # 添加过渡状态日志
            transition_logger.info(f"==================== _get_next_question 开始 ====================")
            
            # 检查会话是否存在
            if quiz_id not in self.__class__.sessions:
                logger.warning(f"会话ID {quiz_id} 不存在")
                error_msg = f"会话ID {quiz_id} 不存在，请先初始化测验"
                logger.error(error_msg)
                return self.create_error_response(error_msg)
            
            # 调试会话状态
            self.debug_session(quiz_id)
            
            # 使用_update_session_state方法更新会话状态
            session = self._update_session_state(quiz_id, config)
            if not session:
                logger.error(f"更新会话状态失败: quiz_id={quiz_id}")
                transition_logger.error(f"更新会话状态失败: quiz_id={quiz_id}")
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": "无法更新会话状态",
                    "question": None,
                    "question_index": -1,
                    "words": None,
                    "formatted_question": "错误：无法更新会话状态"
                }
            
            transition_logger.info(f"获取会话成功: quiz_id={quiz_id}, session_id={id(session)}")
            
            # 记录关键会话状态值，用于调试
            transition_logger.info(f"会话状态摘要: shuffled_questions长度={len(session.get('shuffled_questions', []))}, current_questions长度={len(session.get('current_questions', []))}")
            transition_logger.info(f"会话中的批次信息: total_questions={session.get('total_questions', 0)}, total_batches={session.get('total_batches', 0)}, current_batch_index={session.get('current_batch_index', 0)}")
            
            # 检查会话中是否有题目
            total_questions = session.get("total_questions", 0)
            if total_questions == 0:
                transition_logger.error(f"会话中没有题目: quiz_id={quiz_id}")
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": "会话中没有题目，请先准备题目",
                    "question": None,
                    "question_index": -1,
                    "words": None,
                    "formatted_question": "错误：会话中没有题目"
                }
            
            # 获取总批次数和会话中的当前批次索引（这是真正应该使用的索引）
            total_batches = session.get("total_batches", 0)
            current_batch_index = session.get("current_batch_index", 0)
            
            # 获取当前批次范围
            start_index = session.get("batch_start_index", 0)
            end_index = session.get("batch_end_index", 0)
            
            # 确保使用会话中的batch_size而不是config中的
            batch_size = session.get("batch_size", 5)
            
            # 检查是否所有题目都已回答
            all_answered = session.get("all_answered", False)
            if all_answered:
                transition_logger.info(f"设置transition_to=quiz_completed: 所有题目都已回答完成, quiz_id={quiz_id}")
                
                # 计算得分
                correct_count = session.get("correct_count", 0)
                total_answered = len(session.get("answers", {}))
                score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                
                # 将会话状态设置为已完成
                session["status"] = "completed"
                
                # 更新会话信息
                self.__class__.sessions[quiz_id] = session
                self.__class__.dump_sessions()
                
                # 返回测验完成状态
                return {
                    "quiz_id": quiz_id,
                    "status": "completed",
                    "message": "测验已完成",
                    "question": None,
                    "question_index": -1,
                    "words": None,
                    "total_questions": total_questions,
                    "answered_count": total_answered,
                    "correct_count": correct_count,
                    "score_percentage": score_percentage,
                    "should_transition": True,
                    "transition_to": "quiz_completed"
                }
            
            # 检查当前批次是否全部回答
            batch_fully_answered = session.get("batch_fully_answered", False)
            if batch_fully_answered:
                transition_logger.info(f"批次{current_batch_index}全部回答完毕，检查是否有下一批, quiz_id={quiz_id}")
                
                # 判断是否有下一批
                has_next_batch = current_batch_index + 1 < total_batches
                
                transition_logger.info(f"下一批判断: current_batch_index+1={current_batch_index+1} < total_batches={total_batches} = {has_next_batch}, quiz_id={quiz_id}")
                
                # 计算剩余未答题目数
                answered_count = session.get("answered_count", 0)
                remaining_questions = total_questions - answered_count
                transition_logger.info(f"剩余未答题目: {remaining_questions}, quiz_id={quiz_id}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = current_batch_index + 1
                    transition_logger.info(f"存在下一批次: {next_batch_index}, quiz_id={quiz_id}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": answered_count,
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / answered_count) * 100 if answered_count > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: current_batch_index+1({current_batch_index+1}) >= total_batches({total_batches}), quiz_id={quiz_id}")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__.sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 获取当前批次的问题
            if "current_questions" in session and session["current_questions"]:
                questions_to_use = session["current_questions"]
                transition_logger.info(f"使用current_questions获取当前批次题目，数量: {len(questions_to_use)}, 类型: {type(questions_to_use)}")
            elif "shuffled_questions" in session and session["shuffled_questions"]:
                questions_to_use = session["shuffled_questions"]
                transition_logger.info(f"使用shuffled_questions获取当前批次题目，数量: {len(questions_to_use)}, 类型: {type(questions_to_use)}")
            else:
                questions_to_use = []
                transition_logger.error(f"会话中没有题目集，无法继续, quiz_id={quiz_id}")
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": "会话中没有题目集，无法继续",
                    "question": None,
                    "question_index": -1,
                    "words": None,
                    "formatted_question": "错误：会话中没有题目集"
                }
            
            # 获取已回答的问题索引
            answers = session.get("answers", {})
            answered_indices = list(map(int, answers.keys()))
            
            transition_logger.info(f"答题进度: {len(answered_indices)}/{total_questions} = {(len(answered_indices)/total_questions*100 if total_questions > 0 else 0):.1f}%, quiz_id={quiz_id}")
            transition_logger.info(f"已回答题目索引: {answered_indices}, quiz_id={quiz_id}")
            
            # 计算当前批次内已回答问题的数量
            batch_answered_count = sum(1 for i in range(start_index, end_index) if i in answered_indices)
            batch_total = end_index - start_index
            
            # 检查当前批次是否已全部回答
            batch_fully_answered = batch_answered_count >= batch_total
            transition_logger.info(f"当前批次{current_batch_index}是否已全部回答: {batch_fully_answered}, quiz_id={quiz_id}")
            
            # 添加额外的批次边界日志
            transition_logger.info(f"批次边界: end={end_index}, total={total_questions}, batch_index={current_batch_index}/{total_batches-1 if total_batches > 0 else 0}, quiz_id={quiz_id}")
            
            # 所有题目已经回答完毕
            all_answered = len(answered_indices) >= total_questions
            transition_logger.info(f"所有题目已回答检查: {len(answered_indices)}/{total_questions} = {all_answered}, quiz_id={quiz_id}")
            
            # 检查所有题目是否都已回答
            if all_answered:
                transition_logger.info(f"设置transition_to=quiz_completed: 所有题目都已回答完成, quiz_id={quiz_id}")
                
                # 计算得分
                correct_count = session.get("correct_count", 0)
                total_answered = len(session.get("answers", {}))
                score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                
                # 将会话状态设置为已完成
                session["status"] = "completed"
                
                # 更新会话信息
                self.__class__.sessions[quiz_id] = session
                self.__class__.dump_sessions()
                
                # 返回测验完成状态
                return {
                    "quiz_id": quiz_id,
                    "status": "completed",
                    "message": "测验已完成",
                    "question": None,
                    "question_index": -1,
                    "words": None,
                    "total_questions": total_questions,
                    "answered_count": total_answered,
                    "correct_count": correct_count,
                    "score_percentage": score_percentage,
                    "should_transition": True,
                    "transition_to": "quiz_completed"
                }
            
            # 如果当前批次已全部回答，检查是否有下一批
            if batch_fully_answered:
                transition_logger.info(f"批次{current_batch_index}全部回答完毕，检查是否有下一批, quiz_id={quiz_id}")
                
                # 判断是否有下一批
                has_next_batch = current_batch_index + 1 < total_batches
                
                transition_logger.info(f"下一批判断: current_batch_index+1={current_batch_index+1} < total_batches={total_batches} = {has_next_batch}, quiz_id={quiz_id}")
                
                # 计算剩余未答题目数
                answered_count = session.get("answered_count", 0)
                remaining_questions = total_questions - answered_count
                transition_logger.info(f"剩余未答题目: {remaining_questions}, quiz_id={quiz_id}")
                
                if has_next_batch:
                    # 有下一批，转到下一批
                    next_batch_index = current_batch_index + 1
                    transition_logger.info(f"存在下一批次: {next_batch_index}, quiz_id={quiz_id}")
                    
                    return {
                        "quiz_id": quiz_id,
                        "status": "in_progress",
                        "message": "当前批次已完成，准备进入下一批次",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_questions,
                        "batch_progress": batch_fully_answered,  # 表示当前批次已完成
                        "answered_count": answered_count,
                        "correct_count": session.get("correct_count", 0),
                        "score_percentage": (session.get("correct_count", 0) / answered_count) * 100 if answered_count > 0 else 0,
                        "should_transition": True,
                        "transition_to": "get_next_batch",
                        "next_batch_index": next_batch_index
                    }
                else:
                    # 没有下一批，测验已结束
                    transition_logger.info(f"没有下一批次: current_batch_index+1({current_batch_index+1}) >= total_batches({total_batches}), quiz_id={quiz_id}")
                    
                    # 计算得分
                    correct_count = session.get("correct_count", 0)
                    total_answered = len(session.get("answers", {}))
                    score_percentage = (correct_count / total_answered) * 100 if total_answered > 0 else 0
                    
                    # 将会话状态设置为已完成
                    session["status"] = "completed"
                    
                    # 更新会话信息
                    self.__class__.sessions[quiz_id] = session
                    self.__class__.dump_sessions()
                    
                    # 返回测验完成状态
                    return {
                        "quiz_id": quiz_id,
                        "status": "completed",
                        "message": "测验已完成",
                        "question": None,
                        "question_index": -1,
                        "words": None,
                        "total_questions": total_questions,
                        "answered_count": total_answered,
                        "correct_count": correct_count,
                        "score_percentage": score_percentage,
                        "should_transition": True,
                        "transition_to": "quiz_completed"
                    }
            
            # 查找当前批次中第一个未回答的问题
            global_index = -1
            for i in range(start_index, end_index):
                if i not in answered_indices:
                    global_index = i
                    transition_logger.info(f"找到未回答的问题: global_index={global_index}, quiz_id={quiz_id}")
                    break
            
            # 如果当前批次中没有未回答的问题，使用该批次的第一个问题
            if global_index == -1 and start_index < end_index:
                global_index = start_index
                transition_logger.info(f"当前批次所有问题都已回答，使用第一个问题: global_index={global_index}, quiz_id={quiz_id}")
            
            # 如果仍然没有找到有效的索引，返回错误
            if global_index == -1 or len(questions_to_use) == 0:
                transition_logger.error(f"无法确定要显示的问题: global_index={global_index}, questions_count={len(questions_to_use)}, quiz_id={quiz_id}")
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": "无法确定要显示的问题",
                    "question": None,
                    "question_index": -1,
                    "words": None,
                    "formatted_question": "错误：无法确定要显示的问题"
                    }
            
            # 获取问题并处理输出
            local_index = global_index - start_index  # 将全局索引转换为当前批次内的本地索引
            
            # 添加额外的验证日志
            transition_logger.info(f"索引转换: global_index={global_index}, start_index={start_index}, local_index={local_index}, questions_count={len(questions_to_use)}, quiz_id={quiz_id}")
            
            if local_index < 0 or local_index >= len(questions_to_use):
                logger.error(f"本地索引超出问题集范围: global_index={global_index}, local_index={local_index}, questions_count={len(questions_to_use)}, quiz_id={quiz_id}")
                index_logger.error(f"本地索引超出问题集范围: global_index={global_index}, local_index={local_index}, questions_count={len(questions_to_use)}, quiz_id={quiz_id}")
                transition_logger.error(f"本地索引超出问题集范围: global_index={global_index}, local_index={local_index}, questions_count={len(questions_to_use)}, quiz_id={quiz_id}")
                
                # 添加更详细的错误信息和调试信息
                detailed_error = (f"索引超出范围: global_index={global_index}, local_index={local_index}, "
                                  f"questions_to_use长度={len(questions_to_use)}, "
                                  f"当前批次范围={start_index}-{end_index}, "
                                  f"current_questions长度={len(session.get('current_questions', []))}, "
                                  f"shuffled_questions长度={len(session.get('shuffled_questions', []))}")
                
                transition_logger.error(detailed_error)
                
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": detailed_error,
                    "question": None,
                    "question_index": -1,
                    "words": None,
                    "formatted_question": f"错误：{detailed_error}"
                }
            
            question = questions_to_use[local_index]  # 使用本地索引访问当前批次的问题
            index_logger.info(f"获取问题: global_index={global_index}, local_index={local_index}, question={question if isinstance(question, str) else json.dumps(question, ensure_ascii=False)}, quiz_id={quiz_id}")
            
            # 更新当前问题索引
            session["current_question_index"] = global_index
            index_logger.info(f"更新会话中的当前问题索引: current_question_index={global_index}, quiz_id={quiz_id}")
            
            # 生成格式化问题文本
            formatted_question = ""
            if question:
                # 处理字符串类型的问题（单词列表）
                if isinstance(question, str):
                    formatted_question = f"# {question}\n\n请回答这个单词的含义"
                # 处理字典类型的问题（带选项的问题）
                elif isinstance(question, dict) and 'word' in question:
                    formatted_question = f"# {question['word']}\n\n"
                    
                    if 'options' in question and isinstance(question['options'], list) and len(question['options']) > 0:
                        option_letters = ['A', 'B', 'C', 'D']
                        for j, option in enumerate(question['options']):
                            if j < len(option_letters) and j < len(question['options']):
                                formatted_question += f"{option_letters[j]}. {option}\n\n"
                    else:
                        logger.warning(f"问题没有有效的选项列表: {question['word']}, quiz_id={quiz_id}")
                        index_logger.warning(f"问题没有有效的选项列表: {question['word']}, quiz_id={quiz_id}")
                        formatted_question = f"# {question['word']}\n\n请提供选项"
                else:
                    # 处理其他类型
                    formatted_question = f"# {str(question)}\n\n请回答此题"
            
            # 计算统计信息
            answered_count = len(answers)
            correct_count = session.get("correct_count", 0)
            score_percentage = (correct_count / answered_count) * 100 if answered_count > 0 else 0
            
            # 确保会话状态正确
            if session.get("status", "") != "in_progress":
                session["status"] = "in_progress"
                index_logger.info(f"更新会话状态为 in_progress, quiz_id={quiz_id}")
            
            # 如果question是字符串类型（单词列表），将其包装成字典以保持一致性
            if isinstance(question, str):
                question_obj = {"word": question}
                index_logger.info(f"将字符串类型的问题转换为字典: {question} -> {question_obj}, quiz_id={quiz_id}")
            else:
                question_obj = question
            
            # 返回结果 - 使用会话中的current_batch_index
            result = {
                "quiz_id": quiz_id,
                "status": session.get("status", "in_progress"),
                "message": "成功获取问题",
                "question": question_obj,
                "question_index": global_index,
                "global_index": global_index,
                "batch_index": current_batch_index,
                "batch_size": batch_size,
                "words": None,
                "total_questions": total_questions,
                "answered_count": answered_count,
                "correct_count": correct_count,
                "score_percentage": round(score_percentage, 2),
                "current_batch": current_batch_index,
                "total_batches": total_batches,
                "formatted_question": formatted_question,
                "wrong_answers": session.get("wrong_answers", []),
                "should_transition": False
            }
            
            logger.info(f"返回结果: status={result['status']}, question_index={result['question_index']}, batch_index={result['batch_index']}, quiz_id={quiz_id}")
            index_logger.info(f"返回结果: status={result['status']}, question_index={result['question_index']}, batch_index={result['batch_index']}, quiz_id={quiz_id}")
            transition_logger.info(f"返回结果: status={result.get('status', 'unknown')}, question_index={result.get('question_index', -1)}, transition={result.get('should_transition', False)}/{result.get('transition_to', 'None')}, quiz_id={quiz_id}")
            index_logger.info(f"==================== 获取下一个问题结束 ====================")
            transition_logger.info(f"==================== _get_next_question 结束 ====================")
            return result
            
        except Exception as e:
            logger.error(f"_get_next_question执行异常: {str(e)}")
            logger.error(traceback.format_exc())
            index_logger.error(f"_get_next_question执行异常: {str(e)}")
            index_logger.error(traceback.format_exc())
            transition_logger.error(f"_get_next_question执行异常: {str(e)}")
            transition_logger.error(traceback.format_exc())
            return {
                "quiz_id": quiz_id if 'quiz_id' in locals() else str(uuid.uuid4()),
                "status": "error",
                "message": f"获取下一个问题时出错: {str(e)}",
                "question": None,
                "question_index": -1,
                "words": None,
                "formatted_question": f"错误：{str(e)}"
            }
    
    def _submit_answer(self, config):
        """提交答案"""
        try:
            logger.info(f"提交答案")
            index_logger.info(f"==================== 提交答案开始 ====================")
            transition_logger.info(f"==================== _submit_answer 开始 ====================")
            
            # 获取必要参数
            quiz_id = config.quiz_id
            answer = getattr(config, 'answer', None)
            selected_index = getattr(config, 'selected_index', None)  # 兼容旧参数
            question_index = getattr(config, 'question_index', None)
            
            # 调试会话状态
            self.debug_session(quiz_id)
            
            # 使用_update_session_state方法获取和同步会话状态
            session = self._update_session_state(quiz_id, config)
            if not session:
                logger.error(f"更新会话状态失败: quiz_id={quiz_id}")
                transition_logger.error(f"更新会话状态失败: quiz_id={quiz_id}")
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": "无法更新会话状态",
                    "formatted_message": "错误：无法更新会话状态"
                }
            
            # 兼容处理：如果提供了selected_index但没有answer，使用selected_index
            if answer is None and selected_index is not None:
                answer = selected_index
                index_logger.info(f"使用旧参数selected_index: {selected_index}")
            
            # 处理选项格式的答案 (例如："选项A", "选项B", "选项C", "选项D")
            if isinstance(answer, str) and answer.startswith("选项"):
                option_letter = answer[2:].strip().upper()
                if option_letter in ['A', 'B', 'C', 'D']:
                    answer_index = ord(option_letter) - ord('A')  # A->0, B->1, C->2, D->3
                    index_logger.info(f"转换选项格式答案 '{answer}' 为索引 {answer_index}")
                    answer = answer_index
                else:
                    index_logger.error(f"无效的选项格式: {answer}")
                    return {
                        "quiz_id": quiz_id,
                        "status": "error",
                        "message": f"无效的选项格式: {answer}，期望格式为'选项A'、'选项B'等",
                        "formatted_message": f"错误：无效的选项格式: {answer}"
                    }
            
            # 处理空字符串
            if question_index == '':
                question_index = None
                index_logger.info("处理空字符串question_index为None")
            
            # 日志记录操作细节
            logger.info(f"提交答案: quiz_id={quiz_id}, answer={answer}, question_index={question_index}")
            index_logger.info(f"提交答案参数: quiz_id={quiz_id}, answer={answer}, question_index={question_index}")
            
            # 验证answer
            if answer is None:
                logger.error(f"未提供答案")
                index_logger.error(f"未提供答案")
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": "未提供答案",
                    "formatted_message": "错误：未提供答案"
                }
            
            # 记录操作开始
            index_logger.info(f"会话状态: {session.get('status', '未知')}")
            index_logger.info(f"当前会话已回答题目数: {len(session.get('answers', {}))}")
            
            # 如果没有提供问题索引，尝试从会话中获取当前问题索引
            if question_index is None:
                session_question_index = session.get("current_question_index", None)
                if session_question_index is not None:
                    question_index = session_question_index
                    index_logger.info(f"使用会话中的当前问题索引: {question_index}")
                else:
                    logger.error(f"未提供问题索引，且会话中无当前问题索引")
                    index_logger.error(f"未提供问题索引，且会话中无当前问题索引")
                    return {
                        "quiz_id": quiz_id,
                        "status": "error",
                        "message": "未提供问题索引，且会话中无当前问题索引",
                        "formatted_message": "错误：未提供问题索引，且会话中无当前问题索引"
                    }
            
            # 检查会话中是否已保存了当前题目集合
            saved_questions = session.get("current_questions", None)
            
            # 检查用户是否传入了questions
            user_questions = None
            
            if hasattr(config, 'questions') and config.questions and len(config.questions) > 0:
                logger.info(f"检测到用户传入questions，数量: {len(config.questions) if isinstance(config.questions, list) else '未知'}")
                index_logger.info(f"检测到用户传入questions，数量: {len(config.questions) if isinstance(config.questions, list) else '未知'}")
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
            elif saved_questions is not None:
                questions_to_use = saved_questions
                index_logger.info("使用会话中保存的问题集")
            else:
                index_logger.error("没有找到可用的问题集")
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": "请提供questions参数或先调用get_next_batch",
                    "formatted_message": "错误：请提供questions参数或先调用get_next_batch"
                }
            
            # 记录当前使用的问题集信息
            index_logger.info(f"使用的问题集长度: {len(questions_to_use)}, 类型: {type(questions_to_use)}")
            
            # 记录当前会话中的关键数据
            index_logger.info(f"提交答案前会话状态: {json.dumps({k: v for k, v in session.items() if k != 'current_questions'}, default=str)}")
            
            # 获取全局索引对应的问题
            global_index = int(question_index)
            logger.info(f"处理全局索引: {global_index}")
            index_logger.info(f"处理全局索引: {global_index}")
            
            # 获取批次索引和范围
            batch_size = getattr(config, 'batch_size', 10)
            current_batch_index = session.get("current_batch_index", 0)
            start_index = session.get("batch_start_index", 0)
            end_index = session.get("batch_end_index", 0)
            
            # 转换全局索引为本地索引
            local_index = global_index - start_index
            
            # 添加额外的验证日志
            index_logger.info(f"索引转换: global_index={global_index}, start_index={start_index}, local_index={local_index}, questions_count={len(questions_to_use)}, quiz_id={quiz_id}")
            
            # 验证本地索引的合法性
            if local_index < 0 or local_index >= len(questions_to_use):
                logger.error(f"本地索引超出问题集范围: global_index={global_index}, local_index={local_index}, questions_count={len(questions_to_use)}, quiz_id={quiz_id}")
                index_logger.error(f"本地索引超出问题集范围: global_index={global_index}, local_index={local_index}, questions_count={len(questions_to_use)}, quiz_id={quiz_id}")
                
                # 添加更详细的错误信息
                shuffled_questions = session.get("shuffled_questions", [])
                current_questions = session.get("current_questions", [])
                
                detailed_error = (f"索引超出范围: global_index={global_index}, local_index={local_index}, "
                                 f"questions_to_use长度={len(questions_to_use)}, "
                                 f"当前批次范围={start_index}-{end_index}, "
                                 f"current_questions长度={len(current_questions)}, "
                                 f"shuffled_questions长度={len(shuffled_questions)}")
                
                index_logger.error(detailed_error)
                
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": detailed_error,
                    "formatted_message": f"错误：{detailed_error}"
                }
            
            # 获取问题
            question = questions_to_use[local_index]  # 使用本地索引访问当前批次的问题
            logger.info(f"当前批次索引: {current_batch_index}")
            index_logger.info(f"当前批次索引: {current_batch_index}, local_index={local_index}, global_index={global_index}")
            
            # 检查问题结构 - 增加对字符串类型的兼容处理
            if isinstance(question, str):
                # 如果问题是字符串类型，将其转换为标准字典格式
                question = {
                    "word": question,
                    "options": [],  # 空选项列表
                    "correct_index": 0  # 默认正确答案索引
                }
                logger.info(f"将字符串类型问题 '{question['word']}' 转换为字典格式, quiz_id={quiz_id}")
                index_logger.info(f"将字符串类型问题 '{question['word']}' 转换为字典格式, quiz_id={quiz_id}")
            elif not isinstance(question, dict) or 'word' not in question or 'options' not in question or 'correct_index' not in question:
                logger.error(f"问题结构不完整: {question}")
                index_logger.error(f"问题结构不完整: {json.dumps(question, ensure_ascii=False)}")
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": "问题结构不完整",
                    "formatted_message": "错误：问题结构不完整"
                }
            
            # 记录会话状态
            initial_status = session.get("status", "unknown")
            logger.info(f"提交答案前的会话状态: {initial_status}")
            index_logger.info(f"提交答案前的会话状态: {initial_status}")
            
            # 确保answer是整数类型
            try:
                answer_index = int(answer)
                index_logger.info(f"将答案解析为索引: {answer_index}")
            except (ValueError, TypeError) as e:
                logger.error(f"无法将答案解析为整数: {answer} - {str(e)}")
                index_logger.error(f"无法将答案解析为整数: {answer} - {str(e)}")
                return {
                    "quiz_id": quiz_id,
                    "status": "error",
                    "message": f"无法解析答案，请提供有效的选项索引或选项字母",
                    "formatted_message": f"错误：无法解析答案，请提供有效的选项索引或选项字母"
                }
            
            # 判断答案是否正确
            correct_index = question.get('correct_index', 0)
            is_correct = (answer_index == int(correct_index))
            index_logger.info(f"答案判断: 用户选择={answer_index}, 正确答案={correct_index}, 是否正确={is_correct}")
            
            # 记录答案到全局索引位置
            answers = session.get("answers", {})
            index_logger.info(f"提交答案前已有回答: {json.dumps(answers, default=str)}")
            
            # 记录这道题的答案
            answers[global_index] = {
                "selected_index": answer_index,
                "is_correct": is_correct
            }
            session["answers"] = answers
            index_logger.info(f"添加答案记录: global_index={global_index}, answer={answer_index}, is_correct={is_correct}")
            index_logger.info(f"提交答案后的answers: {json.dumps(answers, default=str)}")
            
            # 保存当前处理的问题索引到会话中
            session["current_question_index"] = global_index
            index_logger.info(f"将当前问题索引保存到会话: current_question_index={global_index}")
            
            # 更新正确答题数
            correct_count = sum(1 for ans in answers.values() if ans.get("is_correct", False))
            session["correct_count"] = correct_count
            index_logger.info(f"更新正确答题数: {correct_count}")
            
            # 如果答案错误，记录到错误答案列表
            if not is_correct:
                wrong_answers = session.get("wrong_answers", [])
                
                # 避免重复添加
                word_already_wrong = any(w == question['word'] for w in wrong_answers)
                
                if not word_already_wrong:
                    wrong_answers.append(question["word"])
                    session["wrong_answers"] = wrong_answers
                    logger.info(f"添加错误答案记录: {question['word']}")
                    index_logger.info(f"添加错误答案记录: {question['word']}")
            
            # 再次调用_update_session_state更新会话状态
            session = self._update_session_state(quiz_id, None, {
                "answers": answers,
                "correct_count": correct_count,
                "current_question_index": global_index
            })
            
            # 计算统计信息
            total_questions = session.get("total_questions", 0)  # 使用session中的total_questions
            total_batches = session.get("total_batches", 0)  # 使用session中的total_batches
            answered_count = len(answers)
            score_percentage = (correct_count / answered_count) * 100 if answered_count > 0 else 0
            
            # 记录更新后的会话状态摘要
            index_logger.info(f"提交答案后会话状态: 总题数={total_questions}, 已答题数={answered_count}, 正确数={correct_count}")
            transition_logger.info(f"答题状态: 已答={answered_count}/{total_questions}, 正确={correct_count}, 批次={current_batch_index}/{total_batches-1 if total_batches > 0 else 0}")
            
            # 生成反馈消息
            formatted_message = ""
            if is_correct:
                formatted_message = f"✅ 回答正确！\n\n"
            else:
                formatted_message = f"❌ 回答错误！正确答案是选项 {chr(65 + int(correct_index))}。\n\n"
            
            # 添加统计信息
            formatted_message += f"进度: {answered_count}/{total_questions} | 正确率: {round(score_percentage, 1)}%"
            
            # 简化：不再在此处获取下一个问题或判断批次完成状态
            # 让前端负责调用get_next_question来获取下一个问题
            
            # 将wrong_answers保持为数组类型，不转换为字符串
            wrong_answers_arr = session.get("wrong_answers", [])
            
            result = {
                "quiz_id": quiz_id,
                "status": "answer_recorded",  # 简化状态
                "message": "答案已记录",
                "is_correct": is_correct,
                "correct_answer": question["options"][correct_index] if 'options' in question and len(question.get("options", [])) > correct_index else "",
                "correct_index": correct_index,
                "global_index": global_index,
                "total_questions": total_questions,
                "answered_count": answered_count,
                "correct_count": correct_count,
                "score_percentage": round(score_percentage, 2),
                "current_batch": current_batch_index,
                "batch_index": current_batch_index,
                "batch_size": batch_size,
                "total_batches": total_batches,
                "formatted_message": formatted_message,
                "wrong_answers": wrong_answers_arr
                # 移除下一个问题相关的字段和状态转换字段
            }
            
            logger.info(f"返回结果: status={result['status']}, is_correct={result['is_correct']}, score={result['score_percentage']}%")
            index_logger.info(f"返回结果: status={result['status']}, is_correct={result['is_correct']}, score={result['score_percentage']}%")
            transition_logger.info(f"返回结果: status={result['status']}, global_index={result['global_index']}")
            index_logger.info(f"==================== 提交答案结束 ====================")
            transition_logger.info(f"==================== _submit_answer 结束 ====================")
            return result
            
        except Exception as e:
            logger.error(f"_submit_answer执行异常: {str(e)}")
            logger.error(traceback.format_exc())
            index_logger.error(f"_submit_answer执行异常: {str(e)}")
            index_logger.error(traceback.format_exc())
            transition_logger.error(f"_submit_answer执行异常: {str(e)}")
            transition_logger.error(traceback.format_exc())
            return {
                "quiz_id": quiz_id if 'quiz_id' in locals() else "unknown",
                "status": "error",
                "message": f"提交答案时出错: {str(e)}",
                "formatted_message": f"错误：{str(e)}"
            }
    
    def _update_session_state(self, quiz_id, config=None, update_data=None):
        """
        统一管理和更新会话状态，确保跨方法状态一致性
        
        参数:
            quiz_id: 测验ID
            config: 配置对象，可能包含batch_index、batch_size等参数
            update_data: 要更新的数据字典
            
        返回:
            更新后的会话对象
        """
        try:
            # 记录调用信息
            caller_frame = traceback.extract_stack()[-2]
            caller_function = caller_frame.name
            logger.info(f"_update_session_state被{caller_function}调用，quiz_id={quiz_id}")
            transition_logger.info(f"_update_session_state被{caller_function}调用，quiz_id={quiz_id}")
            
            # 获取会话 - 确保使用同一个字典
            if quiz_id not in self.__class__.sessions:
                logger.error(f"未找到会话ID: {quiz_id}")
                transition_logger.error(f"未找到会话ID: {quiz_id}")
                return None
                
            # 获取会话对象的引用
            session = self.__class__.sessions[quiz_id]
            logger.info(f"获取会话成功，会话对象地址: id(session)={id(session)}, quiz_id={quiz_id}")
            transition_logger.info(f"获取会话成功，会话对象地址: id(session)={id(session)}, quiz_id={quiz_id}")
            
            # 确保兼容性，同步_quiz_sessions和sessions
            self.__class__._quiz_sessions = self.__class__.sessions
            
            # 处理config中的数据
            if config:
                # 从config中获取题目集合
                if hasattr(config, 'questions') and config.questions:
                    questions = config.questions
                    if isinstance(questions, str):
                        try:
                            questions = json.loads(questions)
                        except:
                            questions = None
                    
                    if questions:
                        session["current_questions"] = questions
                        logger.info(f"从config更新题目集，数量: {len(questions)}, 题目对象地址: id(questions)={id(questions)}, quiz_id={quiz_id}")
                        transition_logger.info(f"从config更新题目集，数量: {len(questions)}, quiz_id={quiz_id}")
                
                # 获取批次相关参数 - 只在初始设置时更新batch_size，后续保持一致
                if "batch_size" not in session and hasattr(config, 'batch_size'):
                    session["batch_size"] = config.batch_size
                    logger.info(f"首次设置批次大小: batch_size={config.batch_size}, quiz_id={quiz_id}")
                    transition_logger.info(f"首次设置批次大小: batch_size={config.batch_size}, quiz_id={quiz_id}")
                # 如果config中的batch_size与会话中的不同，记录但不更新
                elif hasattr(config, 'batch_size') and session.get("batch_size") != config.batch_size:
                    logger.warning(f"忽略config中的batch_size={config.batch_size}，保持会话中的batch_size={session.get('batch_size')}, quiz_id={quiz_id}")
                    transition_logger.warning(f"忽略config中的batch_size={config.batch_size}，保持会话中的batch_size={session.get('batch_size')}, quiz_id={quiz_id}")
                
                if hasattr(config, 'batch_index') and config.batch_index is not None:
                    session["current_batch_index"] = config.batch_index
            
            # 处理update_data中的数据
            if update_data:
                for key, value in update_data.items():
                    # 特殊处理batch_size，确保批次大小一致性
                    if key == "batch_size" and "batch_size" in session:
                        logger.warning(f"忽略update_data中的batch_size={value}，保持会话中的batch_size={session.get('batch_size')}, quiz_id={quiz_id}")
                        transition_logger.warning(f"忽略update_data中的batch_size={value}，保持会话中的batch_size={session.get('batch_size')}, quiz_id={quiz_id}")
                        continue
                    
                    session[key] = value
                    logger.info(f"更新会话状态: {key}={value}, quiz_id={quiz_id}")
                    transition_logger.info(f"更新会话状态: {key}={value}, quiz_id={quiz_id}")
            
            # 获取shuffled_questions和current_questions，优先使用shuffled_questions计算总题目数
            shuffled_questions = session.get("shuffled_questions", [])
            current_questions = session.get("current_questions", [])
            
            # 记录会话中题目集合的情况
            transition_logger.info(f"会话状态摘要: shuffled_questions长度={len(shuffled_questions)}, current_questions长度={len(current_questions)}, quiz_id={quiz_id}")
            
            # 确保batch_size存在，并记录当前使用的批次大小
            if "batch_size" not in session:
                session["batch_size"] = 5  # 默认批次大小
                logger.info(f"设置默认批次大小: batch_size=5, quiz_id={quiz_id}")
                transition_logger.info(f"设置默认批次大小: batch_size=5, quiz_id={quiz_id}")
            
            batch_size = session["batch_size"]
            transition_logger.info(f"当前使用的批次大小: batch_size={batch_size}, quiz_id={quiz_id}")
            
            # 计算并更新批次相关信息 - 关键修改：优先使用shuffled_questions计算总题目数和批次数
            if shuffled_questions and len(shuffled_questions) > 0:
                total_questions = len(shuffled_questions)
                transition_logger.info(f"使用shuffled_questions计算题目总数: total_questions={total_questions}, quiz_id={quiz_id}")
            else:
                # 如果shuffled_questions为空，则使用current_questions
                total_questions = len(current_questions)
                transition_logger.info(f"shuffled_questions为空，使用current_questions计算题目总数: total_questions={total_questions}, quiz_id={quiz_id}")
            
            total_batches = max(1, (total_questions + batch_size - 1) // batch_size) if total_questions > 0 else 0
            
            # 添加更详细的日志以便调试
            transition_logger.info(f"题目总数计算: total_questions={total_questions}")
            transition_logger.info(f"批次总数计算: max(1, ({total_questions} + {batch_size} - 1) // {batch_size}) = {total_batches}")
            
            # 更新基本状态参数
            session["total_questions"] = total_questions
            session["total_batches"] = total_batches
            
            # 获取当前批次索引
            if "current_batch_index" not in session:
                session["current_batch_index"] = 0
            
            current_batch_index = session["current_batch_index"]
            
            # 确保批次索引有效 - 只在明显无效（小于0或者远大于total_batches）时重置
            if current_batch_index < 0 or (total_batches > 0 and current_batch_index >= total_batches * 2):
                logger.warning(f"批次索引严重超出范围: {current_batch_index}，重置为0, quiz_id={quiz_id}")
                transition_logger.warning(f"批次索引严重超出范围: {current_batch_index}，重置为0, quiz_id={quiz_id}")
                current_batch_index = 0
                session["current_batch_index"] = 0
            # 当current_batch_index刚好等于total_batches时，可能是最后一个批次，不要重置
            elif total_batches > 0 and current_batch_index > 0 and current_batch_index >= total_batches:
                # 仅记录日志，不重置批次索引
                logger.info(f"批次索引超出正常范围但在合理范围内: {current_batch_index}/{total_batches}, quiz_id={quiz_id}")
                transition_logger.info(f"批次索引超出正常范围但在合理范围内: {current_batch_index}/{total_batches}, quiz_id={quiz_id}")
            
            # 计算当前批次范围 - 关键修改：确保范围基于total_questions
            start_index = current_batch_index * batch_size
            end_index = min(start_index + batch_size, total_questions)
            
            # 记录完整的批次范围计算过程
            transition_logger.info(f"批次范围计算: start_index = {current_batch_index} * {batch_size} = {start_index}, quiz_id={quiz_id}")
            transition_logger.info(f"批次范围计算: end_index = min({start_index} + {batch_size}, {total_questions}) = {end_index}, quiz_id={quiz_id}")
            
            # 更新批次范围信息
            session["batch_start_index"] = start_index
            session["batch_end_index"] = end_index
            
            # 计算已回答状态
            answers = session.get("answers", {})
            answered_indices = []
            for k in answers.keys():
                try:
                    answered_indices.append(int(k))
                except:
                    pass
            
            # 更新回答状态
            answered_count = len(answered_indices)
            session["answered_count"] = answered_count
            
            # 计算当前批次状态
            batch_answered_count = sum(1 for i in range(start_index, end_index) if i in answered_indices)
            batch_total = end_index - start_index
            batch_fully_answered = batch_answered_count >= batch_total
            
            # 更新批次完成状态
            session["batch_fully_answered"] = batch_fully_answered
            session["batch_answered_count"] = batch_answered_count
            session["batch_total"] = batch_total
            
            # 检查是否所有题目都已回答 - 关键修改：确保基于total_questions
            all_answered = answered_count >= total_questions
            session["all_answered"] = all_answered
            
            # 保存会话 - 确保其他方法能看到更改
            self.__class__.sessions[quiz_id] = session
            # 确保兼容性
            self.__class__._quiz_sessions = self.__class__.sessions
            
            # 记录关键状态
            logger.info(f"会话状态更新完成: quiz_id={quiz_id}")
            logger.info(f"题目数据: total_questions={total_questions}, current_questions={id(session.get('current_questions'))}, quiz_id={quiz_id}")
            logger.info(f"批次信息: batch_index={current_batch_index}/{total_batches-1 if total_batches > 0 else 0}, range={start_index}-{end_index}, quiz_id={quiz_id}")
            logger.info(f"答题进度: answered={answered_count}/{total_questions}, batch_answered={batch_answered_count}/{batch_total}, quiz_id={quiz_id}")
            logger.info(f"状态标记: all_answered={all_answered}, batch_fully_answered={batch_fully_answered}, quiz_id={quiz_id}")
            
            transition_logger.info(f"会话状态更新完成: quiz_id={quiz_id}")
            transition_logger.info(f"题目数据: total_questions={total_questions}, quiz_id={quiz_id}")
            transition_logger.info(f"批次信息: batch_index={current_batch_index}/{total_batches-1 if total_batches > 0 else 0}, range={start_index}-{end_index}, quiz_id={quiz_id}")
            transition_logger.info(f"答题进度: answered={answered_count}/{total_questions}, batch_answered={batch_answered_count}/{batch_total}, quiz_id={quiz_id}")
            transition_logger.info(f"状态标记: all_answered={all_answered}, batch_fully_answered={batch_fully_answered}, quiz_id={quiz_id}")
            
            return session
        except Exception as e:
            logger.error(f"更新会话状态异常: {str(e)}, quiz_id={quiz_id}")
            logger.error(traceback.format_exc())
            transition_logger.error(f"更新会话状态异常: {str(e)}, quiz_id={quiz_id}")
            transition_logger.error(traceback.format_exc())
            return None
    
    def debug_session(self, quiz_id):
        """
        打印会话中的关键状态信息，用于调试。
        
        参数:
            quiz_id: 测验ID
        """
        if not quiz_id or quiz_id not in self.__class__.sessions:
            transition_logger.error(f"debug_session: 未找到会话ID: {quiz_id}")
            return
            
        session = self.__class__.sessions[quiz_id]
        
        # 常规状态
        transition_logger.info(f"============ 会话调试信息 quiz_id={quiz_id} ============")
        transition_logger.info(f"会话对象地址: id={id(session)}")
        transition_logger.info(f"会话状态: status={session.get('status', '未知')}")
        
        # 题目信息
        questions = session.get("current_questions", [])
        original_questions = session.get("original_questions", [])
        shuffled_questions = session.get("shuffled_questions", [])
        
        transition_logger.info(f"题目数量: current={len(questions)}, original={len(original_questions)}, shuffled={len(shuffled_questions)}")
        
        # 批次信息
        batch_size = session.get("batch_size", 0)
        total_questions = session.get("total_questions", 0)
        total_batches = session.get("total_batches", 0)
        current_batch_index = session.get("current_batch_index", 0)
        next_batch_index = session.get("next_batch_index", None)
        
        transition_logger.info(f"批次配置: batch_size={batch_size}, current_index={current_batch_index}, next_index={next_batch_index}")
        transition_logger.info(f"批次计算: total_questions={total_questions}, total_batches={total_batches}")
        
        # 批次范围
        start_index = session.get("batch_start_index", 0)
        end_index = session.get("batch_end_index", 0)
        
        transition_logger.info(f"当前批次范围: [{start_index}-{end_index})")
        
        # 答题状态
        answers = session.get("answers", {})
        answered_count = session.get("answered_count", 0)
        correct_count = session.get("correct_count", 0)
        current_question_index = session.get("current_question_index", -1)
        
        transition_logger.info(f"答题状态: answered={answered_count}/{total_questions}, correct={correct_count}, current_question={current_question_index}")
        transition_logger.info(f"已回答题目索引: {list(answers.keys())}")
        
        # 批次状态
        batch_answered_count = session.get("batch_answered_count", 0)
        batch_total = session.get("batch_total", 0)
        batch_fully_answered = session.get("batch_fully_answered", False)
        all_answered = session.get("all_answered", False)
        
        transition_logger.info(f"批次完成状态: batch_answered={batch_answered_count}/{batch_total}, fully_answered={batch_fully_answered}")
        transition_logger.info(f"全部完成状态: all_answered={all_answered}")
        transition_logger.info(f"============ 会话调试信息结束 ============")