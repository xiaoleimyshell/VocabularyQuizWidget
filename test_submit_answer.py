#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
答案提交功能测试脚本

此脚本模拟单词测验小部件中_submit_answer方法的功能，测试：
1. 用户答案的处理
2. 答案正确性的判定
3. 对答题记录的跟踪
4. 状态转换的判断
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_submit_answer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("test_submit_answer")

def convert_option_to_index(option):
    """将选项转换为索引
    
    支持以下格式:
    - A/B/C/D 字母
    - "选项A"/"选项B"/"选项C"/"选项D"
    - 数字字符串 "0"/"1"/"2"/"3"
    - 数字索引 0/1/2/3
    
    返回:
    - 整数索引 (0-3)
    - None 如果无法转换
    """
    # 如果是None，直接返回None
    if option is None:
        return None
        
    # 如果是整数，直接返回
    if isinstance(option, int):
        return option
        
    # 如果是字符串，进行转换
    if isinstance(option, str):
        # 去除空格并转为大写
        v = option.strip().upper()
        
        # 检查是否为'选项X'格式
        if v.startswith('选项') and len(v) > 2:
            v = v[2:]  # 提取字母部分
        
        # 如果是A-D字母，转换为0-3的索引
        if v in ['A', 'B', 'C', 'D']:
            return ord(v) - ord('A')  # A->0, B->1, C->2, D->3
            
        # 如果是数字字符串，转换为整数
        if v.isdigit():
            return int(v)
            
    # 其他情况无法转换
    logger.warning(f"无法将选项 '{option}' 转换为索引")
    return None

def submit_answer(session, quiz_id, current_question, answer_option, time_spent=None):
    """模拟_submit_answer方法的功能
    
    参数:
        session: 模拟会话状态
        quiz_id: 测验ID
        current_question: 当前问题数据
        answer_option: 用户提交的答案选项(A/B/C/D或索引)
        time_spent: 答题花费的时间(秒)
        
    返回:
        dict: 包含处理结果的字典
    """
    logger.info(f"提交答案: quiz_id={quiz_id}, answer_option={answer_option}")
    
    # 1. 验证测验ID和当前问题
    if quiz_id not in session:
        logger.error(f"无效的测验ID: {quiz_id}")
        return {"error": f"无效的测验ID: {quiz_id}"}
    
    quiz_data = session[quiz_id]
    
    if not current_question:
        logger.error("没有当前问题数据")
        return {"error": "没有当前问题数据"}
    
    # 2. 将用户答案转换为索引
    selected_index = convert_option_to_index(answer_option)
    if selected_index is None:
        logger.error(f"无效的答案选项: {answer_option}")
        return {"error": f"无效的答案选项: {answer_option}"}
    
    # 3. 获取当前问题数据
    question_id = current_question.get("id")
    correct_index = current_question.get("correct_index")
    word = current_question.get("word", "")
    options = current_question.get("options", [])
    
    # 4. 判断答案正确性
    is_correct = selected_index == correct_index
    
    # 5. 记录答题结果
    timestamp = datetime.now().isoformat()
    
    answer_record = {
        "question_id": question_id,
        "selected_index": selected_index,
        "correct_index": correct_index,
        "is_correct": is_correct,
        "timestamp": timestamp,
        "time_spent": time_spent if time_spent is not None else 0
    }
    
    # 6. 更新会话中的答题记录
    if "answer_records" not in quiz_data:
        quiz_data["answer_records"] = []
    
    quiz_data["answer_records"].append(answer_record)
    
    # 7. 获取正确和错误的答案列表
    correct_answers = [r for r in quiz_data["answer_records"] if r["is_correct"]]
    wrong_answers = [r for r in quiz_data["answer_records"] if not r["is_correct"]]
    
    # 8. 计算完成状态
    total_questions = len(quiz_data.get("questions", []))
    answered_count = len(quiz_data["answer_records"])
    correct_count = len(correct_answers)
    completion_percentage = int(answered_count / total_questions * 100) if total_questions > 0 else 0
    
    # 9. 判断是否需要状态转换
    all_answered = answered_count >= total_questions
    should_transition = all_answered
    transition_to = "quiz_completed" if all_answered else None
    
    # 10. 构建响应
    result = {
        "success": True,
        "word": word,
        "options": options,
        "selected_index": selected_index,
        "correct_index": correct_index,
        "is_correct": is_correct,
        "question_id": question_id,
        "stats": {
            "total_questions": total_questions,
            "answered_count": answered_count,
            "correct_count": correct_count,
            "wrong_count": len(wrong_answers),
            "completion_percentage": completion_percentage
        },
        "should_transition": should_transition,
        "transition_to": transition_to
    }
    
    logger.info(f"答案提交结果: 正确={is_correct}, 进度={completion_percentage}%, 状态转换={should_transition}")
    return result

def test_submit_answer_basic():
    """测试基本的答案提交功能"""
    logger.info("开始测试基本答案提交功能")
    
    # 创建模拟会话和测验数据
    session = {}
    quiz_id = "test_quiz_1"
    
    # 创建10个测试问题
    questions = []
    for i in range(10):
        questions.append({
            "id": f"q{i}",
            "word": f"word_{i}",
            "options": [f"选项{j}_问题{i}" for j in range(4)],
            "correct_index": i % 4  # 正确答案轮换
        })
    
    # 初始化测验数据
    session[quiz_id] = {
        "questions": questions,
        "current_question_index": 0,
        "answer_records": []
    }
    
    # 测试提交正确答案
    current_question = questions[0]
    correct_option = chr(ord('A') + current_question["correct_index"])
    
    logger.info(f"测试正确答案: 问题={current_question['word']}, 正确选项={correct_option}")
    result = submit_answer(session, quiz_id, current_question, correct_option, 5)
    
    assert result["success"] == True
    assert result["is_correct"] == True
    assert result["stats"]["correct_count"] == 1
    assert result["stats"]["answered_count"] == 1
    
    # 测试提交错误答案
    current_question = questions[1]
    wrong_index = (current_question["correct_index"] + 1) % 4
    wrong_option = chr(ord('A') + wrong_index)
    
    logger.info(f"测试错误答案: 问题={current_question['word']}, 错误选项={wrong_option}")
    result = submit_answer(session, quiz_id, current_question, wrong_option, 3)
    
    assert result["success"] == True
    assert result["is_correct"] == False
    assert result["stats"]["correct_count"] == 1
    assert result["stats"]["wrong_count"] == 1
    assert result["stats"]["answered_count"] == 2
    
    logger.info("基本答案提交测试通过")

def test_submit_all_answers():
    """测试提交所有答案并触发状态转换"""
    logger.info("开始测试提交所有答案")
    
    # 创建模拟会话和测验数据
    session = {}
    quiz_id = "test_quiz_2"
    
    # 创建5个测试问题
    questions = []
    for i in range(5):
        questions.append({
            "id": f"q{i}",
            "word": f"word_{i}",
            "options": [f"选项{j}_问题{i}" for j in range(4)],
            "correct_index": i % 4  # 正确答案轮换
        })
    
    # 初始化测验数据
    session[quiz_id] = {
        "questions": questions,
        "current_question_index": 0
    }
    
    # 依次回答所有问题
    for i, question in enumerate(questions):
        # 随机选择正确或错误答案
        use_correct = i % 2 == 0  # 偶数题目答对，奇数题目答错
        
        if use_correct:
            option_index = question["correct_index"]
        else:
            option_index = (question["correct_index"] + 1) % 4
            
        option = chr(ord('A') + option_index)
        
        logger.info(f"回答问题 {i+1}/{len(questions)}: {question['word']} -> {option} ({'正确' if use_correct else '错误'})")
        result = submit_answer(session, quiz_id, question, option, i + 2)
        
        # 记录统计信息
        logger.info(f"  进度: {result['stats']['completion_percentage']}%, 正确: {result['stats']['correct_count']}/{result['stats']['answered_count']}")
        
        # 检查最后一个问题是否触发状态转换
        if i == len(questions) - 1:
            assert result["should_transition"] == True
            assert result["transition_to"] == "quiz_completed"
            logger.info("  检测到状态转换: quiz_completed")
        else:
            assert result["should_transition"] == False
            assert result["transition_to"] is None
    
    # 最终统计
    final_stats = result["stats"]
    logger.info(f"测验完成: 总题数={final_stats['total_questions']}, 已答={final_stats['answered_count']}, 正确={final_stats['correct_count']}")
    
    # 检查是否所有问题都已回答
    assert final_stats["answered_count"] == final_stats["total_questions"]
    
    logger.info("全部答案提交测试通过")

if __name__ == "__main__":
    try:
        test_submit_answer_basic()
        print("\n" + "-"*50 + "\n")
        test_submit_all_answers()
        logger.info("所有测试完成")
    except AssertionError as e:
        logger.error(f"测试断言失败: {e}")
    except Exception as e:
        logger.exception(f"测试过程中发生错误: {e}") 