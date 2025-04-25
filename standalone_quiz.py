#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
独立版词汇测验

这是一个可以独立运行的词汇测验脚本，不依赖于ShellAgent的架构。
适合用于本地测试和开发。
"""

import logging
import os
import random
import pickle
from typing import Dict, List, Optional, Any, Union

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("standalone_quiz.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("standalone_quiz")

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

class StandaloneVocabularyQuiz:
    """独立版词汇测验"""
    
    def __init__(self):
        """初始化测验"""
        self.questions = self._generate_sample_questions()
        self.current_index = 0
        self.answers = {}  # 记录用户的答案
        logger.info(f"初始化测验，共 {len(self.questions)} 个问题")
        
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
        
    def submit_answer(self, answer_option):
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
    
    def reset_quiz(self):
        """重置测验"""
        self.current_index = 0
        self.answers = {}
        random.shuffle(self.questions)
        logger.info("测验已重置")
    
    def get_current_question_ui(self):
        """获取当前问题的UI友好格式"""
        current_question = self._get_current_question()
        if not current_question:
            return None
            
        return {
            "id": current_question["id"],
            "word": current_question["word"],
            "options": current_question["options"],
            "difficulty": current_question["difficulty"],
            "option_labels": ["A", "B", "C", "D"][:len(current_question["options"])],
        }
        
    def save_session(self, filename="quiz_session.pkl"):
        """保存会话状态"""
        session_data = {
            "current_index": self.current_index,
            "answers": self.answers,
            "questions": self.questions
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(session_data, f)
        logger.info(f"会话已保存到 {filename}")
        
    def load_session(self, filename="quiz_session.pkl"):
        """加载会话状态"""
        if not os.path.exists(filename):
            logger.warning(f"会话文件 {filename} 不存在")
            return False
            
        try:
            with open(filename, 'rb') as f:
                session_data = pickle.load(f)
                
            self.current_index = session_data.get("current_index", 0)
            self.answers = session_data.get("answers", {})
            self.questions = session_data.get("questions", self._generate_sample_questions())
            
            logger.info(f"从 {filename} 加载了会话")
            return True
        except Exception as e:
            logger.error(f"加载会话时出错: {e}")
            return False

# 交互式测试
def interactive_test():
    """交互式测试独立版词汇测验"""
    quiz = StandaloneVocabularyQuiz()
    
    print("=" * 50)
    print("词汇测验 - 交互式测试")
    print("=" * 50)
    print("命令: [A-D]=回答, s=保存, l=加载, r=重置, q=退出")
    
    while True:
        # 显示当前问题
        question = quiz.get_current_question_ui()
        if not question:
            print("\n没有更多问题")
            break
            
        print(f"\n问题: {question['word']}")
        for label, option in zip(question["option_labels"], question["options"]):
            print(f"  {label}. {option}")
            
        # 获取用户输入
        user_input = input("\n请选择 (A/B/C/D 或 q 退出): ").strip()
        
        # 处理命令
        if user_input.lower() == 'q':
            print("退出测验")
            break
        elif user_input.lower() == 's':
            quiz.save_session()
            print("会话已保存")
            continue
        elif user_input.lower() == 'l':
            if quiz.load_session():
                print("会话已加载")
            else:
                print("加载会话失败")
            continue
        elif user_input.lower() == 'r':
            quiz.reset_quiz()
            print("测验已重置")
            continue
            
        # 提交答案
        is_correct, correct_option, selected_option = quiz.submit_answer(user_input)
        
        # 显示结果
        if is_correct is None:
            print("无法处理答案")
            continue
            
        print(f"回答: {selected_option}")
        print(f"正确答案: {correct_option}")
        print(f"结果: {'正确' if is_correct else '错误'}")
        
        # 显示统计信息
        stats = quiz._get_statistics()
        print(f"进度: {stats['answered_questions']}/{stats['total_questions']} " + 
              f"({stats['completion_percentage']}%), 正确率: {stats['accuracy']}%")
    
    # 显示最终统计
    stats = quiz._get_statistics()
    print("\n测验结束")
    print(f"总问题数: {stats['total_questions']}")
    print(f"已回答: {stats['answered_questions']}")
    print(f"正确答案: {stats['correct_answers']}")
    print(f"正确率: {stats['accuracy']}%")
    
# 自动测试
def automated_test():
    """自动测试独立版词汇测验"""
    quiz = StandaloneVocabularyQuiz()
    
    print("=" * 50)
    print("词汇测验 - 自动测试")
    print("=" * 50)
    
    for _ in range(len(quiz.questions) + 1):  # +1 测试边界情况
        question = quiz.get_current_question_ui()
        if not question:
            print("\n没有更多问题")
            break
            
        print(f"\n问题 {quiz.current_index+1}: {question['word']}")
        
        # 随机选择一个答案
        answer_index = random.randint(0, 3)
        answer_option = chr(ord('A') + answer_index)
        
        # 提交答案
        is_correct, correct_option, selected_option = quiz.submit_answer(answer_option)
        
        # 显示结果
        print(f"选择: {selected_option}")
        print(f"正确答案: {correct_option}")
        print(f"结果: {'正确' if is_correct else '错误'}")
    
    # 显示最终统计
    stats = quiz._get_statistics()
    print("\n测验结束")
    print(f"总问题数: {stats['total_questions']}")
    print(f"已回答: {stats['answered_questions']}")
    print(f"正确答案: {stats['correct_answers']}")
    print(f"正确率: {stats['accuracy']}%")
    
# 选项解析测试
def option_parsing_test():
    """测试选项解析功能"""
    print("=" * 50)
    print("选项解析测试")
    print("=" * 50)
    
    test_options = [
        "A", "B", "C", "D",                     # 字母选项
        "a", "b", "c", "d",                     # 小写字母
        "选项A", "选项B", "选项C", "选项D",     # "选项X"格式
        "0", "1", "2", "3",                     # 数字字符串
        0, 1, 2, 3,                             # 数字索引
        " A ", " 选项B ", " 2 ",                # 带空格
        "E", "5", -1, 4, "选项E", "选项5",      # 无效选项
        None, "", "   "                         # 特殊情况
    ]
    
    for option in test_options:
        result = convert_option_to_index(option)
        print(f"选项: {option}, 转换结果: {result}")

# 主函数
if __name__ == "__main__":
    print("=" * 50)
    print("独立版词汇测验")
    print("=" * 50)
    print("请选择测试模式:")
    print("1. 交互式测试")
    print("2. 自动测试")
    print("3. 选项解析测试")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == "1":
        interactive_test()
    elif choice == "2":
        automated_test()
    elif choice == "3":
        option_parsing_test()
    else:
        print("无效选择，退出") 