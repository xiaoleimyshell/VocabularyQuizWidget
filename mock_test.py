#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VocabularyQuizWidget 独立测试脚本
使用模拟类替代框架依赖
"""

import uuid
import random
from typing import List, Dict, Any, Optional

# 模拟BaseWidget和WIDGETS
class BaseWidget:
    class InputsSchema:
        pass
    class OutputsSchema:
        pass

class MockRegistry:
    def register_module(self):
        def decorator(cls):
            return cls
        return decorator

WIDGETS = MockRegistry()

# 模拟Field类
class Field:
    def __init__(self, default=None, description=None):
        self.default = default
        self.description = description

# 导入VocabularyQuizWidget的代码
exec("""
class VocabularyQuizWidget(BaseWidget):
    \"\"\"单词测验小部件，实现单词测验功能\"\"\"
    CATEGORY = "教育/语言学习"
    NAME = "单词测验"
    
    class InputsSchema(BaseWidget.InputsSchema):
        questions = Field(
            default=[],
            description="题目列表，格式为[{'word': '单词', 'options': ['选项1', '选项2', '选项3', '选项4'], 'correct_index': 0}, ...]"
        )
        batch_size = Field(
            default=15,
            description="每批返回的题目数量"
        )
        batch_index = Field(
            default=0,
            description="批次索引，从0开始"
        )
        quiz_id = Field(
            default=None,
            description="考试ID，如果提供则使用之前乱序的题目列表，否则重新乱序"
        )
        operation = Field(
            default="get_batch",
            description="操作类型：get_batch（获取批次）, start_quiz（开始测验）, submit_answer（提交答案）, end_quiz（结束测验）"
        )
        selected_index = Field(
            default=None,
            description="用户选择的选项索引，用于submit_answer操作"
        )
        question_index = Field(
            default=None,
            description="当前问题的索引，用于submit_answer操作"
        )
    
    class OutputsSchema(BaseWidget.OutputsSchema):
        quiz_id = Field(description="考试唯一标识")
        status = Field(description="操作状态：success, error, 或考试状态：not_started, in_progress, completed")
        message = Field(description="状态消息，尤其是错误信息")
        words = Field(description="当前批次的单词，逗号分隔")
        total_questions = Field(description="总题目数")
        answered_count = Field(description="已回答题目数")
        correct_count = Field(description="正确题目数")
        wrong_answers = Field(
            description="错误回答详情 [{'word': '单词', 'selected': '选择的选项', 'correct': '正确选项'}, ...]"
        )
        score_percentage = Field(description="得分百分比")
        current_batch = Field(description="当前批次索引")
        total_batches = Field(description="总批次数")
        
    # 用于存储测验状态的字典，键为quiz_id
    _quiz_sessions = {}
    
    def execute(self, environ, config):
        \"\"\"执行小部件的主要入口方法\"\"\"
        operation = config.operation
        
        if operation == "get_batch":
            return self._get_batch_words(config)
        elif operation == "start_quiz":
            return self._start_quiz(config)
        elif operation == "submit_answer":
            return self._submit_answer(config)
        elif operation == "end_quiz":
            return self._end_quiz(config)
        else:
            return {
                "quiz_id": str(uuid.uuid4()),
                "status": "error",
                "message": f"不支持的操作: {operation}"
            }
    
    def _get_batch_words(self, config):
        \"\"\"获取一批乱序的单词\"\"\"
        questions = config.questions
        batch_size = config.batch_size
        batch_index = config.batch_index
        quiz_id = config.quiz_id
        
        if not questions and not quiz_id:
            return {
                "quiz_id": str(uuid.uuid4()),
                "status": "error",
                "message": "题目列表为空且未提供有效的quiz_id"
            }
        
        # 如果提供了quiz_id，尝试获取会话
        if quiz_id and quiz_id in self._quiz_sessions:
            session = self._quiz_sessions[quiz_id]
            shuffled_questions = session["shuffled_questions"]
        else:
            # 创建新会话并乱序题目
            quiz_id = str(uuid.uuid4())
            shuffled_questions = questions.copy()
            random.shuffle(shuffled_questions)
            
            self._quiz_sessions[quiz_id] = {
                "shuffled_questions": shuffled_questions,
                "status": "not_started",
                "answers": {},  # 用户回答的题目
                "correct_count": 0,
                "wrong_answers": []
            }
        
        # 计算总批次数和验证批次索引
        total_questions = len(shuffled_questions)
        total_batches = (total_questions + batch_size - 1) // batch_size
        
        if batch_index < 0 or batch_index >= total_batches:
            return {
                "quiz_id": quiz_id,
                "status": "error",
                "message": f"批次索引无效，有效范围: 0-{total_batches-1}"
            }
        
        # 获取当前批次的题目
        start_index = batch_index * batch_size
        end_index = min(start_index + batch_size, total_questions)
        current_batch_questions = shuffled_questions[start_index:end_index]
        
        # 提取单词并用逗号连接
        words = ",".join([q["word"] for q in current_batch_questions])
        
        return {
            "quiz_id": quiz_id,
            "status": "success",
            "words": words,
            "current_batch": batch_index,
            "total_batches": total_batches,
            "total_questions": total_questions
        }
    
    def _start_quiz(self, config):
        \"\"\"开始一个新的测验会话\"\"\"
        quiz_id = config.quiz_id
        
        # 验证quiz_id是否有效
        if not quiz_id or quiz_id not in self._quiz_sessions:
            return {
                "quiz_id": quiz_id or str(uuid.uuid4()),
                "status": "error",
                "message": "未找到有效的测验会话"
            }
        
        # 更新会话状态
        session = self._quiz_sessions[quiz_id]
        session["status"] = "in_progress"
        
        # 重置答案记录
        session["answers"] = {}
        session["correct_count"] = 0
        session["wrong_answers"] = []
        
        return {
            "quiz_id": quiz_id,
            "status": "in_progress",
            "total_questions": len(session["shuffled_questions"]),
            "answered_count": 0,
            "correct_count": 0,
            "score_percentage": 0.0
        }
    
    def _submit_answer(self, config):
        \"\"\"提交答案并更新测验状态\"\"\"
        quiz_id = config.quiz_id
        question_index = config.question_index
        selected_index = config.selected_index
        
        # 验证参数
        if not quiz_id or quiz_id not in self._quiz_sessions:
            return {
                "quiz_id": quiz_id or str(uuid.uuid4()),
                "status": "error",
                "message": "未找到有效的测验会话"
            }
        
        if question_index is None or selected_index is None:
            return {
                "quiz_id": quiz_id,
                "status": "error",
                "message": "问题索引和选择索引不能为空"
            }
        
        # 获取会话和问题
        session = self._quiz_sessions[quiz_id]
        if session["status"] != "in_progress":
            return {
                "quiz_id": quiz_id,
                "status": "error",
                "message": f"测验当前状态为 {session['status']}，无法提交答案"
            }
        
        questions = session["shuffled_questions"]
        if question_index < 0 or question_index >= len(questions):
            return {
                "quiz_id": quiz_id,
                "status": "error",
                "message": f"问题索引无效: {question_index}"
            }
        
        question = questions[question_index]
        correct_index = question["correct_index"]
        
        # 记录答案
        is_correct = selected_index == correct_index
        session["answers"][question_index] = {
            "selected_index": selected_index,
            "is_correct": is_correct
        }
        
        # 更新统计信息
        if is_correct:
            session["correct_count"] += 1
        else:
            session["wrong_answers"].append({
                "word": question["word"],
                "selected": question["options"][selected_index],
                "correct": question["options"][correct_index]
            })
        
        # 计算进度
        answered_count = len(session["answers"])
        total_questions = len(questions)
        score_percentage = (session["correct_count"] / answered_count) * 100 if answered_count > 0 else 0
        
        # 如果所有题目都已回答，自动结束测验
        if answered_count >= total_questions:
            session["status"] = "completed"
        
        return {
            "quiz_id": quiz_id,
            "status": session["status"],
            "answered_count": answered_count,
            "correct_count": session["correct_count"],
            "score_percentage": round(score_percentage, 2),
            "total_questions": total_questions,
            "is_correct": is_correct
        }
    
    def _end_quiz(self, config):
        \"\"\"结束测验并返回最终结果\"\"\"
        quiz_id = config.quiz_id
        
        # 验证quiz_id是否有效
        if not quiz_id or quiz_id not in self._quiz_sessions:
            return {
                "quiz_id": quiz_id or str(uuid.uuid4()),
                "status": "error",
                "message": "未找到有效的测验会话"
            }
        
        # 获取会话
        session = self._quiz_sessions[quiz_id]
        session["status"] = "completed"
        
        # 计算统计信息
        total_questions = len(session["shuffled_questions"])
        answered_count = len(session["answers"])
        correct_count = session["correct_count"]
        score_percentage = (correct_count / answered_count) * 100 if answered_count > 0 else 0
        
        return {
            "quiz_id": quiz_id,
            "status": "completed",
            "total_questions": total_questions,
            "answered_count": answered_count,
            "correct_count": correct_count,
            "score_percentage": round(score_percentage, 2),
            "wrong_answers": session["wrong_answers"]
        }
""")

# 创建EasyDict类
class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        super(EasyDict, self).__init__(**d, **kwargs)
        for key, value in self.items():
            self.__setattr__(key, value)

def main():
    """主测试函数"""
    print("=== VocabularyQuizWidget 独立测试 ===")
    
    # 创建小部件实例
    widget = VocabularyQuizWidget()
    
    # 准备测试数据
    sample_questions = [
        {
            "word": "apple",
            "options": ["苹果", "橙子", "香蕉", "葡萄"],
            "correct_index": 0
        },
        {
            "word": "book",
            "options": ["书", "杂志", "报纸", "笔记本"],
            "correct_index": 0
        },
        {
            "word": "computer",
            "options": ["电脑", "手机", "平板", "电视"],
            "correct_index": 0
        },
        {
            "word": "dog",
            "options": ["狗", "猫", "兔子", "鸟"],
            "correct_index": 0
        },
        {
            "word": "elephant",
            "options": ["大象", "老虎", "狮子", "熊"],
            "correct_index": 0
        }
    ]
    
    # 1. 测试获取批次
    print("\n1. 测试获取批次")
    config = EasyDict({
        "operation": "get_batch",
        "questions": sample_questions,
        "batch_size": 2,
        "batch_index": 0,
        "quiz_id": None
    })
    
    result = widget.execute({}, config)
    print(f"获取批次结果: {result}")
    quiz_id = result["quiz_id"]
    
    # 2. 测试使用相同quiz_id获取另一个批次
    print("\n2. 测试使用相同quiz_id获取另一个批次")
    config2 = EasyDict({
        "operation": "get_batch",
        "questions": [],  # 不需要再传题目
        "batch_size": 2,
        "batch_index": 1,
        "quiz_id": quiz_id
    })
    
    result2 = widget.execute({}, config2)
    print(f"获取第二批次结果: {result2}")
    
    # 3. 测试开始测验
    print("\n3. 测试开始测验")
    config3 = EasyDict({
        "operation": "start_quiz",
        "quiz_id": quiz_id
    })
    
    result3 = widget.execute({}, config3)
    print(f"开始测验结果: {result3}")
    
    # 4. 测试提交正确答案
    print("\n4. 测试提交正确答案")
    config4 = EasyDict({
        "operation": "submit_answer",
        "quiz_id": quiz_id,
        "question_index": 0,
        "selected_index": 0  # 正确答案
    })
    
    result4 = widget.execute({}, config4)
    print(f"提交正确答案结果: {result4}")
    
    # 5. 测试提交错误答案
    print("\n5. 测试提交错误答案")
    config5 = EasyDict({
        "operation": "submit_answer",
        "quiz_id": quiz_id,
        "question_index": 1,
        "selected_index": 1  # 错误答案
    })
    
    result5 = widget.execute({}, config5)
    print(f"提交错误答案结果: {result5}")
    
    # 6. 测试结束测验
    print("\n6. 测试结束测验")
    config6 = EasyDict({
        "operation": "end_quiz",
        "quiz_id": quiz_id
    })
    
    result6 = widget.execute({}, config6)
    print(f"结束测验结果: {result6}")
    
    # 7. 测试错误情况
    print("\n7. 测试错误情况 - 无效批次索引")
    config7 = EasyDict({
        "operation": "get_batch",
        "questions": sample_questions,
        "batch_size": 2,
        "batch_index": 10,  # 超出范围
        "quiz_id": quiz_id
    })
    
    result7 = widget.execute({}, config7)
    print(f"无效批次索引结果: {result7}")

if __name__ == "__main__":
    main() 