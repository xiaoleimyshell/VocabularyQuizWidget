#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
选项解析测试脚本

此脚本测试单词测验小部件中的选项解析功能，包括：
1. 各种格式选项字符串转换为索引
2. 模拟用户提交答案的功能
"""

import logging
import os
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_option_parsing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("test_option_parsing")

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

def test_option_conversion():
    """测试各种选项格式的转换"""
    logger.info("开始测试选项转换功能")
    
    # 测试用例：(输入, 预期输出)
    test_cases = [
        # 字母选项
        ("A", 0),
        ("B", 1),
        ("C", 2),
        ("D", 3),
        ("a", 0),
        ("b", 1),
        ("c", 2),
        ("d", 3),
        
        # 带空格的字母选项
        (" A ", 0),
        (" B ", 1),
        
        # "选项X"格式
        ("选项A", 0),
        ("选项B", 1),
        ("选项C", 2),
        ("选项D", 3),
        
        # 数字字符串
        ("0", 0),
        ("1", 1),
        ("2", 2),
        ("3", 3),
        
        # 数字索引
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        
        # 无效输入
        ("E", None),
        ("选项E", None),
        ("4", None),
        (4, None),
        ("abc", None),
        ("", None),
        (None, None),
        ("选择A", None)
    ]
    
    # 执行测试
    for i, (input_option, expected_output) in enumerate(test_cases):
        result = convert_option_to_index(input_option)
        is_correct = result == expected_output
        
        status = "✓" if is_correct else "✗"
        logger.info(f"测试 {i+1}: 输入='{input_option}' → 输出={result} → 预期={expected_output} → {status}")
        
        if not is_correct:
            logger.error(f"测试失败: 输入='{input_option}', 预期={expected_output}, 实际={result}")
    
    logger.info("选项转换测试完成")

def test_answer_checking():
    """测试答案检查功能"""
    logger.info("开始测试答案检查功能")
    
    # 模拟问题数据
    question = {
        "id": "q1",
        "word": "测试单词",
        "options": ["选项A", "选项B", "选项C", "选项D"],
        "correct_index": 2  # 正确答案是C
    }
    
    # 测试用例：不同格式的用户输入
    user_inputs = [
        ("C", True),              # 正确答案，字母格式
        ("A", False),             # 错误答案，字母格式
        ("选项C", True),          # 正确答案，"选项X"格式
        ("选项B", False),         # 错误答案，"选项X"格式
        ("2", True),              # 正确答案，数字字符串
        ("1", False),             # 错误答案，数字字符串
        (2, True),                # 正确答案，数字索引
        (0, False),               # 错误答案，数字索引
    ]
    
    # 执行测试
    for i, (user_input, expected_correct) in enumerate(user_inputs):
        # 转换用户输入为索引
        selected_index = convert_option_to_index(user_input)
        
        # 检查答案是否正确
        is_correct = selected_index == question["correct_index"]
        match_expected = is_correct == expected_correct
        
        status = "✓" if match_expected else "✗"
        logger.info(f"答案测试 {i+1}: 输入='{user_input}' → 索引={selected_index} → 正确={is_correct} → 预期={expected_correct} → {status}")
        
        if not match_expected:
            logger.error(f"答案检查测试失败: 输入='{user_input}', 预期正确性={expected_correct}, 实际正确性={is_correct}")
    
    logger.info("答案检查测试完成")

if __name__ == "__main__":
    try:
        print("=" * 50)
        print("选项解析测试")
        print("=" * 50)
        
        # 执行选项转换测试
        test_option_conversion()
        
        print("\n" + "-"*50 + "\n")
        
        # 执行答案检查测试
        test_answer_checking()
        
        print("\n" + "=" * 50)
        logger.info("所有测试完成")
        print("测试完成，详细日志请查看 test_option_parsing.log")
        
    except Exception as e:
        logger.exception(f"测试过程中发生错误: {e}")
        print(f"测试失败: {e}") 