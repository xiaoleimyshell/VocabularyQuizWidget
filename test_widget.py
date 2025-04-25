#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试VocabularyQuizWidget功能的脚本
此脚本模拟多批次测验流程，验证批次转换是否正确
"""

import os
import sys
import json
import logging
from easydict import EasyDict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WidgetTest")

# 导入Widget类
try:
    from vocabulary_quiz import VocabularyQuizWidget
except ImportError:
    from .vocabulary_quiz import VocabularyQuizWidget

def create_test_data(word_count=20):
    """创建测试数据"""
    words = []
    for i in range(word_count):
        words.append({
            "word": f"测试词汇{i+1}",
            "meaning": f"测试释义{i+1}",
            "phonetic": f"phonetic{i+1}",
            "example": f"This is an example {i+1}."
        })
    return words

def test_widget():
    """测试Widget功能"""
    # 创建测试数据
    words = create_test_data(20)
    
    # 创建Widget实例
    widget = VocabularyQuizWidget()
    
    # 保存原始会话保存方法
    original_dump_sessions = VocabularyQuizWidget.dump_sessions
    VocabularyQuizWidget.dump_sessions = lambda: None  # 空函数，避免写入文件
    
    # 初始化测验
    logger.info("1. 初始化测验")
    config = EasyDict({
        "title": "测试词汇测验",
        "words": words,
        "quiz_mode": "multiple_choice",
        "batch_size": 5  # 每批5个问题
    })
    
    init_result = widget.execute({}, config)
    quiz_id = init_result.get("quiz_id")
    
    logger.info(f"测验已初始化，ID: {quiz_id}")
    logger.info(f"初始化结果: {init_result}")
    
    # 模拟回答第一批次问题
    logger.info("\n2. 开始回答第一批次问题")
    batch_index = 0
    
    for i in range(5):  # 第一批次有5个问题
        # 获取问题
        get_question_config = EasyDict({
            "quiz_id": quiz_id,
            "batch_index": batch_index,
            "batch_size": 5
        })
        
        question_result = widget._get_next_question(get_question_config)
        question_index = question_result.get("question_index")
        
        logger.info(f"问题{i+1}: 索引={question_index}")
        
        # 回答问题（总是选择正确答案）
        submit_config = EasyDict({
            "quiz_id": quiz_id,
            "question_index": question_index,
            "answer": 0  # 假设0是正确答案
        })
        
        submit_result = widget._submit_answer(submit_config)
        logger.info(f"回答结果: {submit_result.get('message')}")
    
    # 第一批次完成后，获取下一个问题应触发批次转换
    logger.info("\n3. 第一批次完成，检查批次转换")
    next_question_config = EasyDict({
        "quiz_id": quiz_id,
        "batch_index": batch_index,
        "batch_size": 5
    })
    
    transition_result = widget._get_next_question(next_question_config)
    logger.info(f"批次转换结果: {transition_result}")
    
    # 验证批次转换是否正确
    should_transition = transition_result.get("should_transition", False)
    transition_to = transition_result.get("transition_to", None)
    next_batch = transition_result.get("next_batch_index", None)
    
    assert should_transition == True, "应该转换到下一批次"
    assert transition_to == "get_next_batch", f"应该转换到get_next_batch，但实际是{transition_to}"
    assert next_batch == 1, f"下一批次应该是1，但实际是{next_batch}"
    
    logger.info("✅ 第一批次转换正确")
    
    # 模拟回答第二批次问题
    logger.info("\n4. 开始回答第二批次问题")
    batch_index = 1
    
    for i in range(5):  # 第二批次有5个问题
        # 获取问题
        get_question_config = EasyDict({
            "quiz_id": quiz_id,
            "batch_index": batch_index,
            "batch_size": 5
        })
        
        question_result = widget._get_next_question(get_question_config)
        question_index = question_result.get("question_index")
        
        logger.info(f"问题{i+1}: 索引={question_index}")
        
        # 回答问题（总是选择正确答案）
        submit_config = EasyDict({
            "quiz_id": quiz_id,
            "question_index": question_index,
            "answer": 0  # 假设0是正确答案
        })
        
        submit_result = widget._submit_answer(submit_config)
        logger.info(f"回答结果: {submit_result.get('message')}")
    
    # 第二批次完成后，获取下一个问题应触发批次转换
    logger.info("\n5. 第二批次完成，检查批次转换")
    next_question_config = EasyDict({
        "quiz_id": quiz_id,
        "batch_index": batch_index,
        "batch_size": 5
    })
    
    transition_result = widget._get_next_question(next_question_config)
    logger.info(f"批次转换结果: {transition_result}")
    
    # 验证批次转换是否正确
    should_transition = transition_result.get("should_transition", False)
    transition_to = transition_result.get("transition_to", None)
    next_batch = transition_result.get("next_batch_index", None)
    
    assert should_transition == True, "应该转换到下一批次"
    assert transition_to == "get_next_batch", f"应该转换到get_next_batch，但实际是{transition_to}"
    assert next_batch == 2, f"下一批次应该是2，但实际是{next_batch}"
    
    logger.info("✅ 第二批次转换正确")
    
    # 模拟完成所有剩余批次
    logger.info("\n6. 完成剩余批次")
    
    # 第三批次
    batch_index = 2
    for i in range(5):
        get_question_config = EasyDict({
            "quiz_id": quiz_id,
            "batch_index": batch_index,
            "batch_size": 5
        })
        
        question_result = widget._get_next_question(get_question_config)
        question_index = question_result.get("question_index")
        
        # 回答问题
        submit_config = EasyDict({
            "quiz_id": quiz_id,
            "question_index": question_index,
            "answer": 0
        })
        
        widget._submit_answer(submit_config)
    
    # 第四批次（最后一批）
    batch_index = 3
    for i in range(5):
        get_question_config = EasyDict({
            "quiz_id": quiz_id,
            "batch_index": batch_index,
            "batch_size": 5
        })
        
        question_result = widget._get_next_question(get_question_config)
        question_index = question_result.get("question_index")
        
        # 回答问题
        submit_config = EasyDict({
            "quiz_id": quiz_id,
            "question_index": question_index,
            "answer": 0
        })
        
        widget._submit_answer(submit_config)
    
    # 最后一批次完成后，应该结束测验
    logger.info("\n7. 所有批次完成，检查测验结束状态")
    next_question_config = EasyDict({
        "quiz_id": quiz_id,
        "batch_index": batch_index,
        "batch_size": 5
    })
    
    final_result = widget._get_next_question(next_question_config)
    logger.info(f"最终测验结果: {final_result}")
    
    # 验证测验结束状态是否正确
    should_transition = final_result.get("should_transition", False)
    transition_to = final_result.get("transition_to", None)
    status = final_result.get("status", None)
    
    assert should_transition == True, "应该转换到quiz_completed"
    assert transition_to == "quiz_completed", f"应该转换到quiz_completed，但实际是{transition_to}"
    assert status == "completed", f"测验状态应该是completed，但实际是{status}"
    
    logger.info("✅ 测验成功完成")
    
    # 恢复原始会话保存方法
    VocabularyQuizWidget.dump_sessions = original_dump_sessions
    
    return True

if __name__ == "__main__":
    try:
        test_widget()
        logger.info("\n🎉 所有测试通过")
    except AssertionError as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {str(e)}")
        sys.exit(1) 