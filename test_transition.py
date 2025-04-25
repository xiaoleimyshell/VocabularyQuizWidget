#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批次转换测试脚本
用于测试VocabularyQuizWidget批次转换逻辑是否正确
"""

import sys
import json
import logging
from easydict import EasyDict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BatchTransitionTest")

# 导入Widget类
try:
    from vocabulary_quiz import VocabularyQuizWidget
except ImportError:
    from .vocabulary_quiz import VocabularyQuizWidget

class MockSession:
    """模拟会话对象"""
    def __init__(self, questions=None, batch_size=5, answers=None):
        self.questions = questions or []
        self.batch_size = batch_size
        self.answers = answers or {}
        self.correct_count = 0
        
    def to_dict(self):
        return {
            "original_questions": self.questions,
            "shuffled_questions": self.questions,  # 简化测试，不进行打乱
            "answers": self.answers,
            "correct_count": self.correct_count,
            "status": "in_progress"
        }

def create_mock_questions(count):
    """创建模拟测试题"""
    questions = []
    for i in range(count):
        questions.append({
            "word": f"测试词汇{i+1}",
            "options": [f"选项A{i+1}", f"选项B{i+1}", f"选项C{i+1}", f"选项D{i+1}"],
            "correct_answer": 0
        })
    return questions

def test_batch_transition(total_questions=12, batch_size=5):
    """测试批次转换逻辑"""
    # 创建模拟问题
    questions = create_mock_questions(total_questions)
    widget = VocabularyQuizWidget()
    
    # 保存原始会话保存方法
    original_dump_sessions = VocabularyQuizWidget.dump_sessions
    
    # 替换为空方法，防止测试过程中写入文件
    VocabularyQuizWidget.dump_sessions = lambda: None
    
    # 创建模拟会话
    mock_session = MockSession(questions=questions, batch_size=batch_size)
    quiz_id = "test_batch_transition"
    VocabularyQuizWidget.sessions = {quiz_id: mock_session.to_dict()}
    
    # 计算总批次数
    total_batches = (total_questions + batch_size - 1) // batch_size
    
    logger.info(f"开始测试批次转换，总题目数: {total_questions}，批次大小: {batch_size}，总批次数: {total_batches}")
    
    # 测试每个批次的转换
    for batch_idx in range(total_batches):
        # 当前批次的题目范围
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_questions)
        batch_questions = questions[start_idx:end_idx]
        
        logger.info(f"\n===================== 测试批次 {batch_idx+1}/{total_batches} =====================")
        logger.info(f"批次范围: {start_idx}-{end_idx}，题目数: {len(batch_questions)}")
        
        # 设置当前批次所有题目都已回答
        session = VocabularyQuizWidget.sessions[quiz_id]
        for q_idx in range(start_idx, end_idx):
            session["answers"][str(q_idx)] = {"answer": 0, "is_correct": True}
        
        # 获取下一个问题，实际上因为所有题目都已回答，所以会触发批次转换
        config = EasyDict({
            "quiz_id": quiz_id,
            "batch_index": batch_idx,
            "batch_size": batch_size
        })
        
        result = widget._get_next_question(config)
        
        # 检查转换状态
        logger.info(f"批次{batch_idx+1}完成后状态: should_transition={result.get('should_transition', False)}, transition_to={result.get('transition_to', None)}")
        
        # 验证转换结果
        if batch_idx < total_batches - 1:
            # 如果不是最后一个批次，应该转到下一个批次
            assert result.get("should_transition", False) == True, f"批次{batch_idx+1}应该转换到下一批次，但没有转换"
            assert result.get("transition_to") == "get_next_batch", f"批次{batch_idx+1}转换目标应该是get_next_batch，但实际是{result.get('transition_to')}"
            assert result.get("next_batch_index") == batch_idx + 1, f"下一批次索引应该是{batch_idx+1}，但实际是{result.get('next_batch_index')}"
            logger.info(f"✅ 批次{batch_idx+1}成功转换到下一批次{batch_idx+2}")
        else:
            # 如果是最后一个批次，应该结束测验
            assert result.get("should_transition", False) == True, f"最后批次{batch_idx+1}应该转换到quiz_completed，但没有转换"
            assert result.get("transition_to") == "quiz_completed", f"最后批次{batch_idx+1}转换目标应该是quiz_completed，但实际是{result.get('transition_to')}"
            logger.info(f"✅ 最后批次{batch_idx+1}成功转换到quiz_completed")
    
    # 恢复原始会话保存方法
    VocabularyQuizWidget.dump_sessions = original_dump_sessions
    
    logger.info("\n✅ 所有批次转换测试通过")
    return True

if __name__ == "__main__":
    # 可以通过命令行参数指定总题目数和批次大小
    total_questions = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    test_batch_transition(total_questions, batch_size) 