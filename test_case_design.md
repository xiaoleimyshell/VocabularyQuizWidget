# VocabularyQuizWidget 测试用例设计

## 测试目标

对 VocabularyQuizWidget 小部件进行全面测试，确保其按照需求规格正确实现单词测验功能，包括批量获取单词、开始测验、提交答案和结束测验等核心功能。

## 测试环境

- Python 版本：3.7+
- 依赖项：pydantic, typing

## 测试数据准备

### 示例题目列表

```python
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
    # ... 可以添加更多测试数据
]
```

## 测试用例

### 1. 基本功能测试

#### 1.1 批量获取单词测试

**测试目标**：验证 `get_batch` 操作能够正确返回指定批次的单词。

**测试步骤**：
1. 创建 VocabularyQuizWidget 实例
2. 准备包含至少30个单词的题目列表
3. 调用 `execute` 方法，传入 `operation="get_batch"`, `batch_size=10`, `batch_index=0`
4. 检查返回值中的 `quiz_id`, `status`, `words`, `current_batch`, `total_batches`

**预期结果**：
- `status` 应为 "success"
- `words` 应是逗号分隔的单词列表，包含10个单词
- `current_batch` 应为 0
- `total_batches` 应为 3（假设共30个单词）

#### 1.2 状态保持测试

**测试目标**：验证使用同一 `quiz_id` 获取不同批次时，单词顺序一致。

**测试步骤**：
1. 获取第一批单词，记录返回的 `quiz_id`
2. 使用相同的 `quiz_id` 获取第二批单词
3. 确认两次获取的单词没有重复

**预期结果**：
- 第二次请求成功，返回第二批单词
- 第一批和第二批单词没有重叠

### 2. 测验流程测试

#### 2.1 完整测验流程

**测试目标**：验证完整的测验流程，包括获取单词、开始测验、提交答案和结束测验。

**测试步骤**：
1. 获取单词批次，记录 `quiz_id`
2. 调用 `start_quiz` 操作
3. 为每个问题提交答案（部分正确，部分错误）
4. 调用 `end_quiz` 操作
5. 检查最终结果统计

**预期结果**：
- 测验状态从 "not_started" 变为 "in_progress" 再变为 "completed"
- 正确题数、错误题数、得分百分比计算正确
- 错误单词列表包含所有答错的单词

### 3. 边界条件测试

#### 3.1 空题目列表

**测试目标**：验证处理空题目列表的情况。

**测试步骤**：
1. 调用 `execute` 方法，传入空题目列表
2. 检查返回值

**预期结果**：
- `status` 应为 "error"
- `message` 应包含提示信息

#### 3.2 无效批次索引

**测试目标**：验证处理超出范围的批次索引。

**测试步骤**：
1. 使用有效题目列表获取 `quiz_id`
2. 使用超出范围的 `batch_index` 调用 `get_batch`
3. 检查返回值

**预期结果**：
- `status` 应为 "error"
- `message` 应包含提示批次索引无效的信息

#### 3.3 无效问题索引

**测试目标**：验证处理超出范围的问题索引。

**测试步骤**：
1. 开始一个测验
2. 提交答案时使用超出范围的 `question_index`
3. 检查返回值

**预期结果**：
- `status` 应为 "error"
- `message` 应包含问题索引无效的信息

### 4. 异常流程测试

#### 4.1 无效操作类型

**测试目标**：验证处理无效的操作类型。

**测试步骤**：
1. 调用 `execute` 方法，传入不存在的操作类型
2. 检查返回值

**预期结果**：
- `status` 应为 "error"
- `message` 应包含不支持的操作类型的信息

#### 4.2 无效会话ID

**测试目标**：验证处理无效的 `quiz_id`。

**测试步骤**：
1. 调用 `start_quiz` 操作，传入不存在的 `quiz_id`
2. 检查返回值

**预期结果**：
- `status` 应为 "error"
- `message` 应包含未找到有效会话的信息

## 测试代码示例

```python
import unittest
from vocabulary_quiz import VocabularyQuizWidget

class TestVocabularyQuizWidget(unittest.TestCase):
    def setUp(self):
        self.widget = VocabularyQuizWidget()
        self.sample_questions = [
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
            # ... 更多测试数据
        ]
    
    def test_get_batch(self):
        """测试获取单词批次"""
        # 配置
        config = type('obj', (object,), {
            'operation': 'get_batch',
            'questions': self.sample_questions,
            'batch_size': 2,
            'batch_index': 0,
            'quiz_id': None
        })
        
        # 执行
        result = self.widget.execute({}, config)
        
        # 验证
        self.assertEqual(result['status'], 'success')
        self.assertIsNotNone(result['quiz_id'])
        self.assertEqual(result['current_batch'], 0)
        words = result['words'].split(',')
        self.assertEqual(len(words), 2)
    
    def test_quiz_flow(self):
        """测试完整测验流程"""
        # 1. 获取批次
        config1 = type('obj', (object,), {
            'operation': 'get_batch',
            'questions': self.sample_questions,
            'batch_size': 3,
            'batch_index': 0,
            'quiz_id': None
        })
        result1 = self.widget.execute({}, config1)
        quiz_id = result1['quiz_id']
        
        # 2. 开始测验
        config2 = type('obj', (object,), {
            'operation': 'start_quiz',
            'quiz_id': quiz_id
        })
        result2 = self.widget.execute({}, config2)
        self.assertEqual(result2['status'], 'in_progress')
        
        # 3. 提交答案
        config3 = type('obj', (object,), {
            'operation': 'submit_answer',
            'quiz_id': quiz_id,
            'question_index': 0,
            'selected_index': 0  # 正确答案
        })
        result3 = self.widget.execute({}, config3)
        self.assertTrue(result3['is_correct'])
        
        # 4. 结束测验
        config4 = type('obj', (object,), {
            'operation': 'end_quiz',
            'quiz_id': quiz_id
        })
        result4 = self.widget.execute({}, config4)
        self.assertEqual(result4['status'], 'completed')
        self.assertEqual(result4['correct_count'], 1)
```

## 性能测试

### 1. 大量题目测试

**测试目标**：验证处理大量题目的性能。

**测试步骤**：
1. 准备1000个单词的题目列表
2. 测量获取不同批次的响应时间
3. 确保响应时间在可接受范围内

**预期结果**：
- 响应时间应在合理范围内（例如<500ms）
- 内存使用不应显著增加

## 兼容性测试

### 1. 不同输入格式兼容性

**测试目标**：验证处理不同格式输入的兼容性。

**测试步骤**：
1. 使用不同格式的输入数据（例如缺少某些字段，或者类型不匹配）
2. 观察小部件的处理行为

**预期结果**：
- 应优雅地处理格式问题
- 提供清晰的错误消息

## 数据安全测试

### 1. 会话隔离测试

**测试目标**：验证不同测验会话之间的数据隔离。

**测试步骤**：
1. 创建多个测验会话
2. 在各会话中提交不同的答案
3. 验证会话间数据不会相互干扰

**预期结果**：
- 每个测验会话数据应该完全隔离
- 不同会话的操作不会相互影响 