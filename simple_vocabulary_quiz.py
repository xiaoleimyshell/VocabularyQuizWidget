#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化版词汇测验小部件

专注于选项解析和答案提交功能的简化版词汇测验小部件
"""

import logging
import os
import random
from typing import Dict, List, Optional, Any, Union
from pydantic import Field

from proconfig.widgets.base import WIDGETS, BaseWidget

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simple_vocabulary_quiz.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("simple_vocabulary_quiz")

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
        
    # 如果是整数，直接返回（确保在0-3范围内）
    if isinstance(option, int):
        if 0 <= option <= 3:
            return option
        else:
            logger.warning(f"数字索引超出范围: {option}")
            return None
        
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
            num = int(v)
            if 0 <= num <= 3:
                return num
            else:
                logger.warning(f"数字字符串超出范围: {v}")
                return None
            
    # 其他情况无法转换
    logger.warning(f"无法将选项 '{option}' 转换为索引")
    return None

@WIDGETS.register_module()
class SimpleVocabularyQuiz(BaseWidget):
    """简化版词汇测验小部件"""
    
    CATEGORY = "Custom Widgets/Vocabulary Quiz"
    NAME = "简化版词汇测验"
    
    class InputsSchema(BaseWidget.InputsSchema):
        submit_answer: Optional[str] = Field(None, description="提交的答案，格式为选项A/B/C/D或A/B/C/D或0/1/2/3")
        reset_quiz: bool = Field(False, description="是否重置测验")
    
    class OutputsSchema(BaseWidget.OutputsSchema):
        current_question: Dict[str, Any]
        is_correct: Optional[bool] = None
        correct_option: Optional[str] = None
        selected_option: Optional[str] = None
        statistics: Dict[str, Any]
        
    def __init__(self):
        super().__init__()
        self.questions = self._generate_sample_questions()
        self.current_index = 0
        self.answers = {}  # 记录用户的答案
        
    def _generate_sample_questions(self, count=5) -> List[Dict[str, Any]]:
        """生成示例问题"""
        sample_words = ["苹果", "香蕉", "橙子", "葡萄", "西瓜", "菠萝", "草莓", "樱桃", "梨", "桃子"]
        sample_translations = ["apple", "banana", "orange", "grape", "watermelon", "pineapple", "strawberry", "cherry", "pear", "peach"]
        
        questions = []
        for i in range(min(count, len(sample_words))):
            # 正确答案索引
            correct_index = 0
            
            # 生成选项 (正确答案总是第一个)
            options = [sample_translations[i]]
            
            # 添加3个错误选项
            wrong_options = [t for j, t in enumerate(sample_translations) if j != i]
            random.shuffle(wrong_options)
            options.extend(wrong_options[:3])
            
            # 打乱选项顺序
            correct_option = options[0]
            random.shuffle(options)
            correct_index = options.index(correct_option)
            
            questions.append({
                "id": f"q{i+1}",
                "word": sample_words[i],
                "options": options,
                "correct_index": correct_index,
                "difficulty": "简单",
            })
            
        return questions
    
    def _get_question(self, index):
        """获取指定索引的问题"""
        if 0 <= index < len(self.questions):
            return self.questions[index]
        return None
    
    def _get_current_question(self):
        """获取当前问题"""
        return self._get_question(self.current_index)
    
    def _get_statistics(self):
        """获取测验统计信息"""
        total_questions = len(self.questions)
        answered_questions = len(self.answers)
        correct_answers = sum(1 for a in self.answers.values() if a["is_correct"])
        
        return {
            "total_questions": total_questions,
            "answered_questions": answered_questions,
            "correct_answers": correct_answers,
            "completion_percentage": int(answered_questions / total_questions * 100) if total_questions > 0 else 0,
            "accuracy": int(correct_answers / answered_questions * 100) if answered_questions > 0 else 0,
            "remaining_questions": total_questions - answered_questions,
        }
    
    def _format_option_display(self, index):
        """将索引格式化为显示选项 (A/B/C/D)"""
        if 0 <= index <= 3:
            return chr(ord('A') + index)
        return str(index)
        
    def _submit_answer(self, answer_option):
        """提交答案并记录结果"""
        current_question = self._get_current_question()
        if not current_question:
            logger.warning("没有当前问题，无法提交答案")
            return None, None, None
            
        # 转换用户答案为索引
        selected_index = convert_option_to_index(answer_option)
        if selected_index is None:
            logger.warning(f"无法解析答案选项: {answer_option}")
            return False, self._format_option_display(current_question["correct_index"]), None
            
        # 检查答案是否正确
        is_correct = selected_index == current_question["correct_index"]
        
        # 记录答案
        self.answers[current_question["id"]] = {
            "question_id": current_question["id"],
            "word": current_question["word"],
            "selected_index": selected_index,
            "correct_index": current_question["correct_index"],
            "is_correct": is_correct,
        }
        
        # 移动到下一个问题
        if self.current_index < len(self.questions) - 1:
            self.current_index += 1
        
        return (
            is_correct, 
            self._format_option_display(current_question["correct_index"]),
            self._format_option_display(selected_index)
        )
    
    def _reset_quiz(self):
        """重置测验"""
        self.current_index = 0
        self.answers = {}
        random.shuffle(self.questions)
        logger.info("测验已重置")
    
    def execute(self, environ, config):
        """执行小部件逻辑"""
        logger.info(f"执行简化版词汇测验，配置: {config}")
        
        # 重置测验
        if config.reset_quiz:
            self._reset_quiz()
            
        is_correct = None
        correct_option = None
        selected_option = None
        
        # 提交答案
        if config.submit_answer:
            is_correct, correct_option, selected_option = self._submit_answer(config.submit_answer)
            
        # 获取当前问题
        current_question = self._get_current_question()
        
        # 转换为UI友好格式
        ui_question = None
        if current_question:
            ui_question = {
                "id": current_question["id"],
                "word": current_question["word"],
                "options": current_question["options"],
                "difficulty": current_question["difficulty"],
                "option_labels": ["A", "B", "C", "D"][:len(current_question["options"])],
            }
            
        # 获取统计信息
        statistics = self._get_statistics()
            
        # 返回结果
        return {
            "current_question": ui_question,
            "is_correct": is_correct,
            "correct_option": correct_option,
            "selected_option": selected_option,
            "statistics": statistics
        }
        
# 测试代码
if __name__ == "__main__":
    print("=" * 50)
    print("简化版词汇测验测试")
    print("=" * 50)
    
    # 创建小部件实例
    quiz = SimpleVocabularyQuiz()
    
    # 显示初始问题
    config = type('obj', (object,), {
        'submit_answer': None,
        'reset_quiz': False
    })
    
    result = quiz.execute({}, config)
    current_q = result["current_question"]
    
    print(f"\n当前问题: {current_q['word']}")
    for i, (label, option) in enumerate(zip(current_q["option_labels"], current_q["options"])):
        print(f"  {label}. {option}")
    
    # 提交正确答案
    correct_index = quiz.questions[0]["correct_index"]
    correct_answer = chr(ord('A') + correct_index)
    
    config.submit_answer = correct_answer
    result = quiz.execute({}, config)
    
    print(f"\n提交答案: {correct_answer}")
    print(f"是否正确: {'是' if result['is_correct'] else '否'}")
    print(f"正确选项: {result['correct_option']}")
    print(f"选择的选项: {result['selected_option']}")
    
    # 显示下一个问题
    current_q = result["current_question"]
    print(f"\n下一个问题: {current_q['word']}")
    for i, (label, option) in enumerate(zip(current_q["option_labels"], current_q["options"])):
        print(f"  {label}. {option}")
    
    # 提交错误答案
    wrong_index = (quiz.questions[1]["correct_index"] + 1) % 4
    wrong_answer = chr(ord('A') + wrong_index)
    
    config.submit_answer = wrong_answer
    result = quiz.execute({}, config)
    
    print(f"\n提交答案: {wrong_answer}")
    print(f"是否正确: {'是' if result['is_correct'] else '否'}")
    print(f"正确选项: {result['correct_option']}")
    print(f"选择的选项: {result['selected_option']}")
    
    # 显示统计信息
    stats = result["statistics"]
    print("\n测验统计:")
    print(f"  总问题数: {stats['total_questions']}")
    print(f"  已回答: {stats['answered_questions']}")
    print(f"  正确答案: {stats['correct_answers']}")
    print(f"  完成百分比: {stats['completion_percentage']}%")
    print(f"  正确率: {stats['accuracy']}%")
    print(f"  剩余问题: {stats['remaining_questions']}")
    
    # 重置测验
    config.reset_quiz = True
    config.submit_answer = None
    result = quiz.execute({}, config)
    
    print("\n测验已重置")
    stats = result["statistics"]
    print(f"  已回答: {stats['answered_questions']}")
    print(f"  正确答案: {stats['correct_answers']}")
    
    print("\n" + "=" * 50)
    print("测试完成") 