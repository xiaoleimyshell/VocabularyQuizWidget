#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
选项转换测试脚本

此脚本测试单词测验小部件中选项字符串转换为索引的功能。
包括：
1. 测试 A/B/C/D 格式的选项转换
2. 测试"选项A"/"选项B"/"选项C"/"选项D"格式的选项转换
3. 测试数字字符串转换
4. 测试直接数字索引
"""

import os
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_option_conversion.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("test_option_conversion")

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

def test_option_conversion():
    """测试选项转换功能"""
    logger.info("开始测试选项转换功能")
    
    test_cases = [
        # 字母格式
        ('A', 0),
        ('B', 1),
        ('C', 2),
        ('D', 3),
        ('a', 0),
        ('b', 1),
        ('c', 2),
        ('d', 3),
        
        # "选项X"格式
        ('选项A', 0),
        ('选项B', 1),
        ('选项C', 2),
        ('选项D', 3),
        ('选项a', 0),
        ('选项b', 1),
        ('选项c', 2),
        ('选项d', 3),
        
        # 数字字符串
        ('0', 0),
        ('1', 1),
        ('2', 2),
        ('3', 3),
        
        # 整数
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        
        # 带空格的情况
        (' A ', 0),
        (' 选项B ', 1),
        (' 2 ', 2),
        
        # 无效输入
        ('E', None),
        ('选项E', None),
        ('4', 4),  # 注意：数字字符串会被转为整数，但可能超出有效范围
        (4, 4),    # 整数会直接返回，但可能超出有效范围
        ('abc', None),
        ('', None),
        (None, None)
    ]
    
    for option, expected in test_cases:
        result = convert_option_to_index(option)
        status = "✓" if result == expected else "✗"
        logger.info(f"{status} 选项: '{option}', 期望: {expected}, 结果: {result}")
        
        if result != expected:
            logger.error(f"选项转换错误: '{option}' -> {result}, 期望 {expected}")
    
    logger.info("选项转换测试完成")

def test_answer_checking():
    """测试答案检查功能"""
    logger.info("开始测试答案检查功能")
    
    # 模拟一个问题
    question = {
        "word": "apple",
        "options": ["苹果", "橙子", "香蕉", "葡萄"],
        "correct_index": 0
    }
    
    test_inputs = [
        'A', 'B', 'C', 'D',
        '选项A', '选项B', '选项C', '选项D',
        0, 1, 2, 3,
        '0', '1', '2', '3'
    ]
    
    correct_answer_index = question["correct_index"]
    
    for user_input in test_inputs:
        # 转换用户输入为索引
        selected_index = convert_option_to_index(user_input)
        
        # 检查是否正确
        is_correct = selected_index == correct_answer_index
        
        # 记录结果
        status = "✓" if is_correct else "✗"
        logger.info(f"{status} 用户输入: '{user_input}', 转换为: {selected_index}, 正确答案: {correct_answer_index}")
        
        # 如果是正确的，添加更多信息
        if is_correct:
            logger.info(f"   正确! {question['word']} 的正确选项是 {question['options'][correct_answer_index]}")
        else:
            logger.info(f"   错误! {question['word']} 的正确选项是 {question['options'][correct_answer_index]}, 但用户选择了 {question['options'][selected_index] if 0 <= selected_index < len(question['options']) else '无效选项'}")
    
    logger.info("答案检查测试完成")

if __name__ == "__main__":
    try:
        test_option_conversion()
        print("\n" + "-"*50 + "\n")
        test_answer_checking()
        logger.info("所有测试完成")
    except Exception as e:
        logger.exception(f"测试过程中发生错误: {e}") 