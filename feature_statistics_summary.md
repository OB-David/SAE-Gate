# Feature 统计方法简述

目标：在相同 token 总量下，比较哪些 SAE feature 在 PTSD 文本中更常出现，而在 EmpatheticDialogues 和 DailyDialog 中较少出现。

## 流程

1. 对完整 conversation 做 tokenizer。
2. 每个数据集抽到相同 token 总量，例如 `BALANCED_TOKENS_PER_DATASET = 40000`。
3. 按 `WINDOW_TOKEN_LENGTH = 512` 切成计算窗口送入模型。
4. 对每个 token，记录 SAE TopK features。
5. 按 dataset 汇总每个 feature 的出现次数和频率。
6. 比较 PTSD 与两个 control 数据集的 feature 频率差异。

这里的 512-token window 只用于模型计算，不是统计单位。

## 活跃度定义

当前的“活跃”定义是：

```text
某个 feature 出现在一个 token 的 SAE TopK feature 列表中
= 该 feature 在这个 token 上活跃一次
```

注意：每个 token 在所有 feature 维度上都有数值，但这里只统计 TopK 中最显著的 feature。因此这是 **TopK 活跃频率**，不是“非零激活频率”。

主要指标：

- `activation_count`：该 feature 出现在多少个 token 的 TopK 中。
- `dataset_token_count`：该 dataset 实际统计的 token 总数。
- `activation_frequency_per_1k_tokens`：每 1000 个 token 中，该 feature 平均出现多少次。

公式：

```text
activation_frequency_per_1k_tokens =
activation_count / dataset_token_count * 1000
```

## Token 解释

每个 feature 还保留：

- `top_content_tokens`：最常激活该 feature 的内容 token 及次数。

这用于帮助判断该 feature 可能对应什么词、语义或表达模式。

## PTSD-Specific 比较

核心比较列：

```text
activation_frequency_diff_ptsd_minus_control =
PTSD 每 1000 token 出现次数
- control 每 1000 token 出现次数
```

如果一个 feature 在 PTSD 中频率高，而在两个 control 中频率低，它就更可能是 PTSD-specific feature。

## 指标合理性

这个指标适合做第一轮 feature screening：简单、可比，并且控制了文本长度差异。

但它有两个限制：

- 它只看 feature 是否进入 TopK，不等于绝对激活值超过某个阈值。
- 它主要衡量“出现频率”，不直接衡量“激活强度”。

因此可以同时参考：

- `activation_value_sum`
- `activation_value_max`
- `mean_activation_value_when_active`
