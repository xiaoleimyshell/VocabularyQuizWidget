#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
完整测验流程测试脚本

此脚本模拟用户完整的测验流程，包括:
1. 获取题目批次
2. 开始测验
3. 回答所有问题
4. 验证状态转换
5. 结束测验

重点测试所有问题回答后的状态转换和 _get_next_question 方法中的逻辑
"""

import os
import sys
import logging
import pickle
import random
from pathlib import Path
from easydict import EasyDict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_full_quiz.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("test_full_quiz")

# 确保可以导入模块
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from vocabulary_quiz import VocabularyQuizWidget
except ImportError:
    logger.error("无法导入 VocabularyQuizWidget，请确保在正确的目录中运行")
    sys.exit(1)

def create_sample_questions(num_questions=10):
    """创建示例问题"""
    words = ["apple", "book", "computer", "dog", "elephant", "flower", "guitar", 
             "house", "internet", "jacket", "keyboard", "lemon", "mountain", 
             "notebook", "orange", "piano", "queen", "rabbit", "street", "tree"]
    
    translations = {
        "apple": ["苹果", "橙子", "香蕉", "葡萄"],
        "book": ["书", "杂志", "报纸", "笔记本"],
        "computer": ["电脑", "手机", "平板", "电视"],
        "dog": ["狗", "猫", "兔子", "鸟"],
        "elephant": ["大象", "老虎", "狮子", "熊"],
        "flower": ["花", "树", "草", "叶子"],
        "guitar": ["吉他", "钢琴", "小提琴", "鼓"],
        "house": ["房子", "公寓", "大厦", "别墅"],
        "internet": ["互联网", "电视", "广播", "报纸"],
        "jacket": ["夹克", "衬衫", "裤子", "帽子"],
        "keyboard": ["键盘", "鼠标", "显示器", "打印机"],
        "lemon": ["柠檬", "苹果", "香蕉", "橙子"],
        "mountain": ["山", "海", "河", "湖"],
        "notebook": ["笔记本", "书", "杂志", "报纸"],
        "orange": ["橙子", "苹果", "香蕉", "葡萄"],
        "piano": ["钢琴", "吉他", "小提琴", "鼓"],
        "queen": ["女王", "国王", "总统", "首相"],
        "rabbit": ["兔子", "狗", "猫", "鸟"],
        "street": ["街道", "道路", "高速公路", "桥梁"],
        "tree": ["树", "花", "草", "叶子"]
    }
    
    selected_words = words[:min(num_questions, len(words))]
    questions = []
    
    for word in selected_words:
        options = translations.get(word, [f"{word}的翻译", "错误选项1", "错误选项2", "错误选项3"])
        questions.append({
            "word": word,
            "options": options,
            "correct_index": 0  # 假设正确答案总是第一个选项
        })
    
    return questions

def test_full_quiz_flow():
    """测试完整测验流程"""
    logger.info("开始完整测验流程测试")
    
    # 创建小部件实例
    widget = VocabularyQuizWidget()
    
    # 创建示例问题
    num_questions = 5  # 可以调整为任意数量
    batch_size = 3  # 每批返回的问题数量
    questions = create_sample_questions(num_questions)
    
    logger.info(f"创建了 {len(questions)} 个示例问题，批次大小为 {batch_size}")
    
    # 1. 获取第一批题目
    config = EasyDict({
        "operation": "get_batch",
        "questions": questions,
        "batch_size": batch_size,
        "batch_index": 0,
        "quiz_id": None
    })
    
    result = widget.execute({}, config)
    quiz_id = result["quiz_id"]
    logger.info(f"获取批次结果: {result}")
    
    # 2. 开始测验
    config = EasyDict({
        "operation": "start_quiz",
        "quiz_id": quiz_id
    })
    
    result = widget.execute({}, config)
    logger.info(f"开始测验结果: {result}")
    
    # 3. 获取第一个问题
    config = EasyDict({
        "operation": "get_next_question",
        "quiz_id": quiz_id
    })
    
    result = widget.execute({}, config)
    logger.info(f"获取第一个问题结果: {result}")
    
    # 记录当前会话状态
    if hasattr(widget, "_quiz_sessions") and quiz_id in widget._quiz_sessions:
        session_before = pickle.dumps(widget._quiz_sessions[quiz_id])
        logger.info(f"当前会话状态: {widget._quiz_sessions[quiz_id]}")
    
    # 4. 回答所有问题
    for i in range(num_questions):
        # 获取下一个问题
        config = EasyDict({
            "operation": "get_next_question",
            "quiz_id": quiz_id
        })
        
        question_result = widget.execute({}, config)
        
        # 检查是否应该转换状态
        if question_result.get("should_transition", False):
            logger.info(f"问题 {i+1}/{num_questions}: 检测到状态转换 -> {question_result.get('transition_to', '')}")
            break
            
        question_index = question_result.get("question_index")
        
        # 提交答案 (随机选择，有50%几率选择正确答案)
        correct_answer = 0  # 示例问题中正确答案总是索引0
        selected_index = 0 if random.random() > 0.5 else random.randint(1, 3)
        
        config = EasyDict({
            "operation": "submit_answer",
            "quiz_id": quiz_id,
            "question_index": question_index,
            "selected_index": selected_index
        })
        
        answer_result = widget.execute({}, config)
        logger.info(f"问题 {i+1}/{num_questions}: 回答结果: {answer_result}")
    
    # 记录测验后的会话状态
    if hasattr(widget, "_quiz_sessions") and quiz_id in widget._quiz_sessions:
        session_after = pickle.dumps(widget._quiz_sessions[quiz_id])
        logger.info(f"回答后会话状态: {widget._quiz_sessions[quiz_id]}")
        
        # 检查会话状态变化
        if session_before != session_after:
            logger.info("会话状态已更新")
    
    # 5. 再次尝试获取问题，应返回状态转换信息
    config = EasyDict({
        "operation": "get_next_question",
        "quiz_id": quiz_id
    })
    
    result = widget.execute({}, config)
    logger.info(f"回答所有问题后再次获取问题: {result}")
    
    # 验证转换状态
    if result.get("should_transition", False):
        logger.info(f"验证通过: 检测到状态转换 -> {result.get('transition_to', '')}")
    else:
        logger.error(f"验证失败: 未检测到状态转换")
    
    # 6. 结束测验
    config = EasyDict({
        "operation": "end_quiz",
        "quiz_id": quiz_id
    })
    
    result = widget.execute({}, config)
    logger.info(f"结束测验结果: {result}")
    
    # 7. 验证结果
    logger.info("测验完成统计:")
    logger.info(f"总问题数: {result.get('total_questions', 0)}")
    logger.info(f"已回答数: {result.get('answered_count', 0)}")
    logger.info(f"正确数: {result.get('correct_count', 0)}")
    logger.info(f"分数百分比: {result.get('score_percentage', 0)}%")
    
    # 检验是否所有问题都已回答
    if result.get('answered_count', 0) == num_questions:
        logger.info("测试通过: 所有问题已回答")
    else:
        logger.error(f"测试失败: 未回答所有问题, 期望 {num_questions}, 实际 {result.get('answered_count', 0)}")
    
    return True

if __name__ == "__main__":
    try:
        test_full_quiz_flow()
        logger.info("测试完成")
    except Exception as e:
        logger.exception(f"测试过程中发生错误: {e}") 