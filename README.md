# VocabularyQuizWidget

一个用于实现单词测验功能的自定义小部件，支持用户导入单词题库，进行随机乱序，并支持分批获取题目进行测验。

## 功能特点

- 支持导入预设的单词题库（包含单词、选项和正确答案）
- 自动对题目列表进行随机排序
- 支持分批次获取单词，便于分段测验
- 跟踪考试状态：未开始、进行中、已完成
- 记录用户答题情况并提供详细的成绩统计
- 提供错误单词列表，便于复习

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本用法

1. **获取一批单词**

```python
from vocabulary_quiz import VocabularyQuizWidget

# 创建小部件实例
widget = VocabularyQuizWidget()

# 准备题目列表
questions = [
    {
        "word": "apple",
        "options": ["苹果", "橙子", "香蕉", "葡萄"],
        "correct_index": 0
    },
    # ... 更多题目
]

# 获取第一批单词
config = {
    "operation": "get_batch",
    "questions": questions,
    "batch_size": 15,
    "batch_index": 0
}
result = widget.execute({}, config)

# 获取quiz_id和单词列表
quiz_id = result["quiz_id"]
words = result["words"]  # 逗号分隔的单词字符串
```

2. **开始测验**

```python
# 使用之前获取的quiz_id开始测验
config = {
    "operation": "start_quiz",
    "quiz_id": quiz_id
}
result = widget.execute({}, config)
```

3. **提交答案**

```python
# 提交用户答案
config = {
    "operation": "submit_answer",
    "quiz_id": quiz_id,
    "question_index": 0,  # 题目索引
    "selected_index": 2   # 用户选择的选项索引
}
result = widget.execute({}, config)

# 检查答案是否正确
is_correct = result["is_correct"]
```

4. **结束测验**

```python
# 结束测验并获取统计结果
config = {
    "operation": "end_quiz",
    "quiz_id": quiz_id
}
result = widget.execute({}, config)

# 获取测验统计信息
total_questions = result["total_questions"]
correct_count = result["correct_count"]
score_percentage = result["score_percentage"]
wrong_answers = result["wrong_answers"]
```

## 接口说明

### 输入参数

| 参数 | 类型 | 说明 |
|------|------|------|
| operation | string | 操作类型：get_batch, start_quiz, submit_answer, end_quiz |
| questions | list | 题目列表，仅在get_batch操作且未提供quiz_id时需要 |
| batch_size | int | 每批返回的题目数量，默认15 |
| batch_index | int | 批次索引，从0开始 |
| quiz_id | string | 考试ID，用于状态管理 |
| question_index | int | 当前问题的索引，用于submit_answer操作 |
| selected_index | int | 用户选择的选项索引，用于submit_answer操作 |

### 返回数据

返回数据根据操作类型不同而变化，但通常包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| quiz_id | string | 考试唯一标识 |
| status | string | 操作状态或考试状态 |
| message | string | 状态消息，尤其是错误信息 |
| words | string | 当前批次的单词，逗号分隔 |
| total_questions | int | 总题目数 |
| answered_count | int | 已回答题目数 |
| correct_count | int | 正确题目数 |
| score_percentage | float | 得分百分比 |
| wrong_answers | list | 错误回答详情 |

## 错误处理

小部件会处理各种可能的错误情况，并返回带有 `status="error"` 和适当错误消息的响应。常见错误包括：

- 题目列表为空
- 批次索引超出范围
- 问题索引无效
- quiz_id无效
- 操作类型不支持