#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批次转换测试脚本
用于测试VocabularyQuizWidget在多批次场景下的transition_to设置逻辑
"""

import os
import sys
import json
import logging
import pickle
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('batch_transition_test')

# 测试会话数据
def create_test_session(quiz_id, num_questions=15, batch_size=5):
    """创建测试会话数据"""
    questions = []
    for i in range(num_questions):
        questions.append({
            "word": f"word_{i}",
            "options": [f"option_{i}_1", f"option_{i}_2", f"option_{i}_3", f"option_{i}_4"],
            "correct_index": 0
        })
    
    return {
        "quiz_id": quiz_id,
        "timestamp": datetime.now().isoformat(),
        "status": "in_progress",
        "questions": questions,
        "shuffled_questions": questions.copy(),  # 为简单起见，不打乱顺序
        "current_questions": questions[:batch_size],  # 初始批次
        "current_batch_index": 0,
        "current_question_index": None,
        "batch_size": batch_size,
        "answers": {},
        "correct_count": 0,
        "wrong_answers": []
    }

def get_next_question_mock(session, batch_index, num_answered):
    """模拟_get_next_question方法的核心逻辑"""
    logger.info(f"模拟_get_next_question: 批次索引={batch_index}, 已回答数量={num_answered}")
    
    # 获取核心数据
    batch_size = session["batch_size"]
    questions = session["shuffled_questions"]
    
    # 模拟回答题目
    for i in range(min(num_answered, len(questions))):
        session["answers"][i] = 0  # 假设选择第一个选项
        session["correct_count"] += 1  # 假设所有回答都是正确的
    
    # 计算批次范围
    start_index = batch_index * batch_size
    end_index = min(start_index + batch_size, len(questions))
    questions_to_use = questions[start_index:end_index]
    
    # 计算重要的数值
    total_questions = len(questions_to_use)
    total_original_questions = len(questions)
    total_batches = (total_questions + batch_size - 1) // batch_size
    total_original_batches = (total_original_questions + batch_size - 1) // batch_size
    
    logger.info(f"批次计算: batch_index={batch_index}, batch_size={batch_size}, start={start_index}, end={end_index}")
    logger.info(f"当前批次题目数: {total_questions}, 当前批次数: {total_batches}")
    logger.info(f"原始题目数: {total_original_questions}, 原始总批次: {total_original_batches}")
    
    # 获取已回答的问题索引
    answered_indices = set(session["answers"].keys())
    logger.info(f"已回答题目索引: {sorted(list(answered_indices))}")
    logger.info(f"答题进度: {len(answered_indices)}/{total_original_questions}")
    
    # 首先检查是否所有原始题目都已回答
    all_original_questions_answered = len(answered_indices) >= total_original_questions
    if all_original_questions_answered:
        logger.info(f"所有原始题目已回答: {len(answered_indices)}/{total_original_questions}")
        return {
            "should_transition": True,
            "transition_to": "quiz_completed",
            "message": "所有题目都已回答完成"
        }
    
    # 检查当前批次是否已全部回答
    current_batch_answered = True
    for i in range(start_index, end_index):
        if i not in answered_indices:
            current_batch_answered = False
            logger.info(f"当前批次中索引 {i} 未被回答")
            break
    
    logger.info(f"当前批次{batch_index}是否已全部回答: {current_batch_answered}")
    
    # 如果当前批次已全部回答，检查是否有下一批
    if current_batch_answered:
        logger.info(f"批次{batch_index}全部回答完毕，检查是否有下一批")
        
        # 判断是否还有下一批
        has_next_batch = (batch_index + 1) < total_original_batches
        
        if has_next_batch:
            next_batch_index = batch_index + 1
            logger.info(f"有下一批次: {next_batch_index}")
            logger.info(f"下一批范围: {next_batch_index*batch_size}-{min((next_batch_index+1)*batch_size, total_original_questions)}")
            return {
                "should_transition": True,
                "transition_to": "get_next_batch",
                "next_batch_index": next_batch_index,
                "message": f"当前批次已完成，请使用批次索引{next_batch_index}获取下一批"
            }
        else:
            logger.info(f"没有下一批次，当前为最后一批")
            return {
                "should_transition": True,
                "transition_to": "quiz_completed",
                "message": "所有问题都已回答完成"
            }
    
    # 如果当前批次未全部回答，找出第一个未回答的问题
    next_question_index = None
    for i in range(start_index, end_index):
        if i not in answered_indices:
            next_question_index = i
            break
    
    if next_question_index is not None:
        logger.info(f"找到下一个未回答题目，索引: {next_question_index}")
        return {
            "should_transition": False,
            "transition_to": None,
            "question_index": next_question_index,
            "message": "继续回答当前批次中的问题"
        }
    else:
        logger.info("未找到未回答的题目，这是一个意外情况")
        return {
            "should_transition": False,
            "transition_to": None,
            "message": "未找到未回答的题目，请检查逻辑"
        }

def run_batch_transition_tests():
    """运行多个批次转换测试用例"""
    test_cases = [
        # 单批次场景
        {"quiz_id": "test1", "num_questions": 5, "batch_size": 5, "batch_index": 0, "num_answered": 5},  # 刚好一个批次，全部回答
        {"quiz_id": "test2", "num_questions": 5, "batch_size": 5, "batch_index": 0, "num_answered": 3},  # 一个批次，部分回答
        
        # 多批次场景
        {"quiz_id": "test3", "num_questions": 7, "batch_size": 5, "batch_index": 0, "num_answered": 5},  # 第一批全部回答，应转到下一批
        {"quiz_id": "test4", "num_questions": 7, "batch_size": 5, "batch_index": 1, "num_answered": 7},  # 第二批全部回答，应完成测验
        {"quiz_id": "test5", "num_questions": 12, "batch_size": 5, "batch_index": 0, "num_answered": 5},  # 第一批全部回答，应转到下一批
        {"quiz_id": "test6", "num_questions": 12, "batch_size": 5, "batch_index": 1, "num_answered": 10},  # 第二批全部回答，应转到第三批
        {"quiz_id": "test7", "num_questions": 12, "batch_size": 5, "batch_index": 2, "num_answered": 12},  # 第三批全部回答，应完成测验
        
        # 特殊场景
        {"quiz_id": "test8", "num_questions": 7, "batch_size": 3, "batch_index": 1, "num_answered": 5},  # 第二批部分回答
        {"quiz_id": "test9", "num_questions": 11, "batch_size": 5, "batch_index": 1, "num_answered": 7},  # 第二批部分回答
    ]
    
    for i, case in enumerate(test_cases):
        logger.info(f"\n================== 测试用例 {i+1} ==================")
        logger.info(f"配置: 题目数={case['num_questions']}, 批次大小={case['batch_size']}, 批次索引={case['batch_index']}, 已回答={case['num_answered']}")
        
        # 创建测试会话
        session = create_test_session(case["quiz_id"], case["num_questions"], case["batch_size"])
        
        # 模拟 _get_next_question 逻辑
        result = get_next_question_mock(session, case["batch_index"], case["num_answered"])
        
        # 输出结果
        logger.info(f"测试结果: {result}")
        
        # 计算预期结果
        batch_size = case["batch_size"]
        total_questions = case["num_questions"]
        batch_index = case["batch_index"]
        num_answered = case["num_answered"]
        total_batches = (total_questions + batch_size - 1) // batch_size
        
        # 验证结果
        expected_result = None
        if num_answered >= total_questions:
            expected_result = "quiz_completed"
        elif num_answered >= (batch_index + 1) * batch_size:
            # 当前批次已全部回答
            if batch_index + 1 < total_batches:
                expected_result = "get_next_batch"
            else:
                expected_result = "quiz_completed"
        else:
            # 当前批次未全部回答
            expected_result = None
        
        is_correct = (expected_result == result.get("transition_to"))
        logger.info(f"验证: 预期={expected_result}, 实际={result.get('transition_to')}, 结果={'正确' if is_correct else '错误'}")
        
        if not is_correct:
            logger.error(f"测试用例 {i+1} 失败! 期望转换到 {expected_result}，但得到了 {result.get('transition_to')}")
    
    logger.info("\n================== 测试完成 ==================")

if __name__ == "__main__":
    logger.info("开始批次转换测试")
    run_batch_transition_tests() 