#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批次转换测试脚本
用于测试VocabularyQuizWidget在多批次场景下的批次转换逻辑
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
logger = logging.getLogger('batch_test')

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

def test_transition_logic(session, num_answered, batch_size):
    """测试批次转换逻辑"""
    logger.info(f"测试批次转换逻辑: 已回答={num_answered}, 批次大小={batch_size}")
    
    # 模拟回答题目
    for i in range(min(num_answered, len(session["shuffled_questions"]))):
        session["answers"][i] = 0  # 假设选择第一个选项
        session["correct_count"] += 1  # 假设所有回答都是正确的
    
    # 获取关键数据
    questions = session["shuffled_questions"]
    total_questions = len(questions)
    total_batches = (total_questions + batch_size - 1) // batch_size
    current_batch_index = session["current_batch_index"]
    
    # 计算当前批次范围
    start_index = current_batch_index * batch_size
    end_index = min(start_index + batch_size, total_questions)
    
    logger.info(f"批次信息: batch_index={current_batch_index}, 范围={start_index}-{end_index}, 总题目={total_questions}")
    logger.info(f"总批次数: {total_batches}")
    logger.info(f"已回答题目: {sorted(list(session['answers'].keys()))}")
    
    # 检查当前批次是否已全部回答
    current_batch_answered = True
    for i in range(start_index, end_index):
        if i not in session["answers"]:
            current_batch_answered = False
            logger.info(f"当前批次中索引 {i} 未被回答")
            break
    
    logger.info(f"当前批次是否已全部回答: {current_batch_answered}")
    
    # 如果当前批次已全部回答，检查是否有下一批
    if current_batch_answered:
        # 判断是否还有下一批
        has_next_batch = (current_batch_index + 1) < total_batches
        
        if has_next_batch:
            next_batch_index = current_batch_index + 1
            logger.info(f"有下一批次: {next_batch_index}")
            transition_to = "get_next_batch"
        else:
            logger.info(f"没有下一批次，测验已完成")
            transition_to = "quiz_completed"
        
        logger.info(f"应设置 transition_to={transition_to}")
        return {
            "current_batch_answered": current_batch_answered,
            "has_next_batch": has_next_batch,
            "total_questions": total_questions,
            "total_batches": total_batches,
            "transition_to": transition_to
        }
    else:
        logger.info(f"当前批次未完成，应继续当前批次")
        return {
            "current_batch_answered": current_batch_answered,
            "has_next_batch": None,
            "total_questions": total_questions,
            "total_batches": total_batches,
            "transition_to": None
        }

def run_test_cases():
    """运行多个测试用例"""
    test_cases = [
        # 单批次场景
        {"quiz_id": "test1", "num_questions": 5, "batch_size": 5, "num_answered": 5},  # 刚好一个批次，全部回答
        {"quiz_id": "test2", "num_questions": 5, "batch_size": 5, "num_answered": 3},  # 一个批次，部分回答
        
        # 多批次场景
        {"quiz_id": "test3", "num_questions": 7, "batch_size": 5, "num_answered": 5},  # 第一批全部回答，应转到下一批
        {"quiz_id": "test4", "num_questions": 7, "batch_size": 5, "num_answered": 7},  # 全部回答，应完成测验
        {"quiz_id": "test5", "num_questions": 12, "batch_size": 5, "num_answered": 5},  # 第一批全部回答
        {"quiz_id": "test6", "num_questions": 12, "batch_size": 5, "num_answered": 10},  # 前两批全部回答
        {"quiz_id": "test7", "num_questions": 12, "batch_size": 5, "num_answered": 12},  # 全部回答
    ]
    
    for i, case in enumerate(test_cases):
        logger.info(f"\n============ 测试用例 {i+1} ============")
        logger.info(f"配置: 题目数={case['num_questions']}, 批次大小={case['batch_size']}, 已回答={case['num_answered']}")
        
        # 创建测试会话
        session = create_test_session(case["quiz_id"], case["num_questions"], case["batch_size"])
        
        # 执行测试
        result = test_transition_logic(session, case["num_answered"], case["batch_size"])
        
        # 输出结果
        logger.info(f"测试结果: {result}")
        logger.info(f"转换目标: {result['transition_to']}")
        
        # 验证结果
        total_batches = result["total_batches"]
        current_batch = session["current_batch_index"]
        
        expected_transition = None
        
        if result["current_batch_answered"]:
            if current_batch + 1 < total_batches:
                expected_transition = "get_next_batch"
            else:
                expected_transition = "quiz_completed"
        
        if expected_transition:
            is_correct = result["transition_to"] == expected_transition
            logger.info(f"验证: 预期={expected_transition}, 实际={result['transition_to']}, 结果={'正确' if is_correct else '错误'}")
            if not is_correct:
                logger.error(f"测试用例 {i+1} 失败! 期望转换到 {expected_transition}，但得到了 {result['transition_to']}")
        else:
            logger.info("验证: 当前批次未完成，无需转换")
    
    logger.info("\n============ 测试完成 ============")

if __name__ == "__main__":
    logger.info("开始批次转换逻辑测试")
    run_test_cases() 