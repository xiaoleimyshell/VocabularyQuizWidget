# VocabularyQuizWidget 代码设计文档

## 整体架构

VocabularyQuizWidget 是一个用于实现单词测验功能的自定义小部件。该小部件允许用户导入预设的单词题库，进行随机乱序，并支持分批获取题目进行测验。整个测验过程中会跟踪用户的回答，并在测验结束后提供详细的成绩统计。

## 核心类

### VocabularyQuizWidget

继承自 `BaseWidget`，是整个小部件的主类，负责处理所有操作请求并维护测验状态。

#### 主要属性

- `CATEGORY`: 小部件分类，显示在UI中的分类路径
- `NAME`: 小部件名称，显示在UI中
- `_quiz_sessions`: 静态字典，用于存储所有活跃的测验会话状态

#### 输入模式 (InputsSchema)

- `questions`: 题目列表，包含单词、选项和正确答案索引
- `batch_size`: 每批返回的题目数量，默认为15
- `batch_index`: 批次索引，从0开始
- `quiz_id`: 考试ID，用于状态管理
- `operation`: 操作类型，如获取批次、开始测验、提交答案等
- `selected_index`: 用户选择的选项索引
- `question_index`: 当前问题的索引

#### 输出模式 (OutputsSchema)

- `quiz_id`: 考试唯一标识
- `status`: 操作状态或考试状态
- `message`: 状态消息，尤其是错误信息
- `words`: 当前批次的单词，逗号分隔
- `total_questions`: 总题目数
- `answered_count`: 已回答题目数
- `correct_count`: 正确题目数
- `wrong_answers`: 错误回答详情
- `score_percentage`: 得分百分比
- `current_batch`: 当前批次索引
- `total_batches`: 总批次数

## 主要方法

### execute(environ, config)

小部件的主要入口方法，根据 `operation` 参数调用相应的内部方法。

### _get_batch_words(config)

获取一批乱序的单词，主要功能包括：
1. 验证输入参数
2. 如果有quiz_id，使用已有会话；否则创建新会话并乱序题目
3. 计算总批次数和验证批次索引
4. 获取当前批次的题目并提取单词
5. 返回单词列表和相关信息

### _start_quiz(config)

开始一个新的测验会话，功能包括：
1. 验证quiz_id是否有效
2. 更新会话状态为"in_progress"
3. 重置答案记录
4. 返回初始状态信息

### _submit_answer(config)

提交答案并更新测验状态，功能包括：
1. 验证参数
2. 检查会话状态
3. 验证问题索引
4. 记录答案并判断正误
5. 更新统计信息
6. 检查是否所有题目都已回答，如果是则自动结束测验
7. 返回最新状态信息

### _end_quiz(config)

结束测验并返回最终结果，功能包括：
1. 验证quiz_id是否有效
2. 将会话状态设置为"completed"
3. 计算统计信息
4. 返回完整的测验结果

## 数据结构

### 测验会话 (_quiz_sessions)

每个测验会话是一个字典，包含以下字段：
- `shuffled_questions`: 乱序后的题目列表
- `status`: 测验状态，如"not_started"、"in_progress"、"completed"
- `answers`: 用户回答记录，键为问题索引，值为答案详情
- `correct_count`: 正确答案数
- `wrong_answers`: 错误回答详情列表

## 状态管理

小部件使用 `_quiz_sessions` 字典在内存中维护测验状态，通过 `quiz_id` 唯一标识符关联每个测验会话。测验状态包括：

- `not_started`: 未开始，初始化会话后的状态
- `in_progress`: 进行中，开始测验后的状态
- `completed`: 已完成，测验结束后的状态

在实际部署中，可以考虑使用数据库或文件系统来持久化会话状态，以防止服务重启导致数据丢失。

## 错误处理

小部件对各种可能的错误情况进行了处理，包括：
- 题目列表为空
- quiz_id无效
- 批次索引超出范围
- 问题索引无效
- 会话状态不正确

所有错误都会返回适当的错误消息，并设置status为"error"。

## 扩展性考虑

当前实现使用内存字典存储会话状态，适用于开发和测试。在生产环境中，可以考虑：
1. 使用数据库持久化会话状态
2. 添加会话超时机制
3. 支持更多的题目类型
4. 添加用户认证机制 