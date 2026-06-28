# 概率论与收益率 Probability and Returns

> 赢和输的差别，在于对数收益率和算术收益率的差别

## 学习目标

- 理解随机变量的期望与方差，并用纯 Python 手动计算
- 区分简单收益率与对数收益率，知道各自的适用场景
- 掌握对数收益率的时间可加性，理解为什么这在金融中至关重要
- 将日频统计量正确年化（均值×252，标准差×√252）
- 用 akshare 下载沪深300真实数据并验证上述公式
- 能用 pandas/numpy 一行代码完成收益率序列计算

---

## 问题背景

### 两个交易员的故事

李明和张红各自管理一个账户，每人初始 100 万元。

第一年，李明赚了 50%，账户变成 150 万。第二年亏了 50%，账户变成 75 万。
他向客户汇报："我两年的平均收益率是 (50% + (-50%)) / 2 = 0%，持平。"

客户很困惑：明明是亏了 25 万，怎么说"持平"？

这就是**算术平均收益率**的陷阱——它不能反映实际的财富变化。

### 没有正确工具会怎样

如果用简单收益率做多期复合计算：
- 连续两天各涨 5%，终值 = 1.05 × 1.05 = 1.1025，总涨 10.25%
- 但用简单收益率相加：5% + 5% = 10%，差了 0.25%

在短期交易中误差不大，但在回测数年历史数据时，这个误差会累积到显著水平，
导致策略评估失真，最终做出错误决策。

### 解决方案：对数收益率

对数收益率（log return）天然满足时间可加性：

$$\ln(P_3/P_1) = \ln(P_2/P_1) + \ln(P_3/P_2)$$

无论多少期，直接相加就得到总收益率的对数值。这是量化金融中处理历史数据的标准做法。

---

## 核心概念

### 1. 随机变量与概率分布

一个**随机变量** X 是对某个不确定结果的数值表示。

股票明天的涨跌幅就是一个随机变量。假设某股票明天的可能结果如下：

| 结果 | 概率 P | 收益率 X |
|------|--------|----------|
| 大涨 | 0.20 | +5% |
| 小涨 | 0.35 | +2% |
| 平盘 | 0.15 | 0% |
| 小跌 | 0.20 | -2% |
| 大跌 | 0.10 | -5% |
| **合计** | **1.00** | |

**期望（Expected Value）**：加权平均结果

$$\begin{aligned}
\mathbb{E}[X] &= \sum x_i \cdot P(x_i) \\
     &= 0.05 \cdot 0.20 + 0.02 \cdot 0.35 + 0.00 \cdot 0.15 + (-0.02) \cdot 0.20 + (-0.05) \cdot 0.10 \\
     &= 0.01 + 0.007 + 0 - 0.004 - 0.005 \\
     &= 0.008
\end{aligned}$$

（即平均期望收益率 0.8%）

**方差（Variance）**：偏离均值的平方的期望，衡量"不确定性"

$$\operatorname{Var}[X] = \mathbb{E}[(X - \mu)^2] = \sum (x_i - \mu)^2 \cdot P(x_i)$$

**标准差（Standard Deviation）**：方差的平方根，与 X 同量纲

$$\sigma = \sqrt{\operatorname{Var}[X]}$$

在金融中，标准差就是**波动率**（Volatility），是风险的核心度量。

---

### 2. 简单收益率 vs 对数收益率

#### 简单收益率（Arithmetic Return / Simple Return）

$$\begin{aligned}
r_t &= (P_t - P_{t-1}) / P_{t-1} \\
    &= P_t / P_{t-1} - 1
\end{aligned}$$

**直觉**：买入后，价格从 $P_{t-1}$ 涨到 $P_t$，赚了多少百分比。

**优点**：直觉友好，便于向非专业人士解释。

**缺点**：
- 不能跨时期直接相加（5% + 5% ≠ 实际复合收益）
- 下界是 -100%，分布不对称

#### 对数收益率（Log Return / Continuously Compounded Return）

$$\begin{aligned}
r_t &= \ln(P_t / P_{t-1}) \\
    &= \ln(P_t) - \ln(P_{t-1})
\end{aligned}$$

**直觉**：如果把价格取对数，对数收益率就是对数价格序列的一阶差分。

**优点**：
- 时间可加性：总对数收益率 = 各期对数收益率之和
- 理论上取值范围为 (-∞, +∞)，便于统计建模
- 与连续复利直接对应

**缺点**：
- 直觉不如简单收益率直接（但两者在小幅度时近似相等）

#### 两者关系

$$\begin{aligned}
r_{\log} &= \ln(1 + r_{\text{simple}}) \\
r_{\text{simple}} &= e^{r_{\log}} - 1
\end{aligned}$$

当 $r_{\text{simple}}$ 很小时：$r_{\log} \approx r_{\text{simple}}$

（泰勒展开：$\ln(1+x) \approx x - x^2/2 + \dots$）

---

### 3. 时间可加性的意义

假设三天价格为 P1=100, P2=105, P3=110。

**简单收益率求总收益**：

$$\begin{aligned}
r_1 &= (105-100)/100 = 5\% \\
r_2 &= (110-105)/105 = 4.76\% \\
r_{\text{total}} &= (110-100)/100 = 10\%
\end{aligned}$$

（注意：$10\%$ 不等于 $5\% + 4.76\% = 9.76\%$）

**对数收益率求总收益**：

$$\begin{aligned}
r_1 &= \ln(105/100) = \ln(1.05) = 0.04879 \\
r_2 &= \ln(110/105) = \ln(1.04762) = 0.04652 \\
r_{\text{total}} &= r_1 + r_2 = 0.09531
\end{aligned}$$

验证：$\ln(110/100) = \ln(1.1) = 0.09531$，完全相等。

这就是对数收益率在量化中被广泛使用的核心原因：**多期收益率可以直接相加，无需复合计算**。

---

### 4. 年化：从日频到年频

A股一年有 **252 个交易日**（去掉周末和法定节假日后的近似值）。

#### 年化收益率

日均对数收益率：$\mu_{\text{daily}}$

年化收益率：$\mu_{\text{annual}} = \mu_{\text{daily}} \times 252$

（注意：这里是乘法，因为对数收益率可加）

#### 年化波动率

波动率的年化遵循**平方根法则**（来自随机游走假设）：

日波动率：$\sigma_{\text{daily}}$

年化波动率：$\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$

（标准差按时间的平方根缩放）

**为什么是√252 而不是 252？**

如果每日收益率是独立同分布的，$n$ 天总收益率的方差 $= n \times$ 日方差。
因此 $n$ 天总收益率的标准差 $= \sigma_{\text{daily}} \times \sqrt{n}$。

取 $n = 252$，即一年的交易日数。

$$\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252} \approx \sigma_{\text{daily}} \times 15.87$$

---

## 动手实现

### 步骤 1：实现基础统计函数（只用 math 标准库）

```python
import math

def expected_value(values: list, probs: list) -> float:
    """
    计算离散随机变量的期望值
    E[X] = Σ x_i * P(x_i)
    """
    if abs(sum(probs) - 1.0) > 1e-9:
        raise ValueError(f"概率之和必须为1，当前为 {sum(probs)}")
    return sum(x * p for x, p in zip(values, probs))


def variance(values: list, probs: list) -> float:
    """
    计算离散随机变量的方差
    Var[X] = E[(X - μ)²] = Σ (x_i - μ)² * P(x_i)
    """
    mu = expected_value(values, probs)
    return sum((x - mu) ** 2 * p for x, p in zip(values, probs))


def std_dev(values: list, probs: list) -> float:
    """标准差 = √方差"""
    return math.sqrt(variance(values, probs))
```

### 步骤 2：实现收益率计算函数

```python
def log_return(p1: float, p2: float) -> float:
    """
    计算单期对数收益率
    r = ln(P2 / P1)
    """
    if p1 <= 0 or p2 <= 0:
        raise ValueError("价格必须为正数")
    return math.log(p2 / p1)


def simple_return(p1: float, p2: float) -> float:
    """
    计算单期简单收益率
    r = (P2 - P1) / P1
    """
    if p1 <= 0:
        raise ValueError("基准价格必须为正数")
    return (p2 - p1) / p1
```

### 步骤 3：实现年化函数

```python
TRADING_DAYS = 252  # A股年交易日

def annualise_return(daily_mean: float) -> float:
    """
    将日均对数收益率年化
    μ_annual = μ_daily × 252
    """
    return daily_mean * TRADING_DAYS


def annualise_vol(daily_std: float) -> float:
    """
    将日波动率年化（平方根法则）
    σ_annual = σ_daily × √252
    """
    return daily_std * math.sqrt(TRADING_DAYS)
```

### 步骤 4：验证时间可加性

```python
# 验证：ln(P3/P1) = ln(P2/P1) + ln(P3/P2)
P1, P2, P3 = 100.0, 105.0, 110.0

r1 = log_return(P1, P2)
r2 = log_return(P2, P3)
r_total_sum = r1 + r2
r_total_direct = log_return(P1, P3)

print(f"分段对数收益率之和：{r_total_sum:.6f}")
print(f"直接计算总对数收益率：{r_total_direct:.6f}")
print(f"是否相等：{abs(r_total_sum - r_total_direct) < 1e-10}")
```

---

## 使用库

### 用 numpy + pandas 实现等价计算

```python
import numpy as np
import pandas as pd
import akshare as ak

# 下载沪深300指数数据（前复权）
df = ak.index_zh_a_hist(symbol="000300", period="daily",
                         start_date="20230101", end_date="20231231",
                         adjust="qfq")
df = df.rename(columns={"日期": "date", "收盘": "close"})
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

close = df["close"]

# 计算对数收益率
log_returns = np.log(close / close.shift(1)).dropna()

# 计算简单收益率（对比用）
simple_returns = close.pct_change().dropna()

# 年化统计
ann_return = log_returns.mean() * 252
ann_vol = log_returns.std() * np.sqrt(252)

print(f"[pandas/numpy] 年化收益率：{ann_return:.4f} ({ann_return*100:.2f}%)")
print(f"[pandas/numpy] 年化波动率：{ann_vol:.4f} ({ann_vol*100:.2f}%)")

# 对比简单收益率与对数收益率的差异
print(f"\n对数收益率均值：{log_returns.mean():.6f}")
print(f"简单收益率均值：{simple_returns.mean():.6f}")
print(f"差异（小幅度时近似相等）：{abs(log_returns.mean() - simple_returns.mean()):.8f}")
```

### 对比手动实现与库的结果

```python
# 手动计算（用步骤1-3的函数）
log_ret_list = [log_return(close.iloc[i-1], close.iloc[i])
                for i in range(1, len(close))]

manual_mean = expected_value(log_ret_list, [1/len(log_ret_list)] * len(log_ret_list))
manual_var = variance(log_ret_list, [1/len(log_ret_list)] * len(log_ret_list))
manual_std = std_dev(log_ret_list, [1/len(log_ret_list)] * len(log_ret_list))

print(f"\n[手动实现] 日均对数收益率：{manual_mean:.6f}")
print(f"[numpy]   日均对数收益率：{log_returns.mean():.6f}")
print(f"[手动实现] 年化收益率：{annualise_return(manual_mean)*100:.2f}%")
print(f"[手动实现] 年化波动率：{annualise_vol(manual_std)*100:.2f}%")
```

---

## 练习题

1. **[简单]** 某股票三天的价格为 100、102、99。分别用简单收益率和对数收益率计算两天的总收益率，并验证对数收益率的可加性。

2. **[中等]** 一个策略的日胜率是 55%，日均盈利 1.2%，日均亏损 -0.8%（假设每天只有涨跌两种结果）。用 `expected_value` 和 `variance` 函数计算该策略的日期望收益率和日波动率，并将其年化。

3. **[困难]** 下载过去3年的沪深300数据，计算滚动20日年化波动率（即每天用过去20个交易日的对数收益率标准差×√252），并找出波动率最高的5个交易日。思考：这些高波动日发生在什么市场背景下？

---

## 关键术语

| 术语 | 常见误解 | 正确含义 |
|------|----------|----------|
| 简单收益率 | 可以多期直接相加 | 多期复合需要乘积，不能直接相加 |
| 对数收益率 | 复杂难理解，实际用途少 | 量化金融标准，多期可加，便于统计 |
| 年化波动率 | 日波动率 × 252 | 日波动率 × √252（平方根法则） |
| 期望值 | 等于最可能的结果 | 概率加权平均，可能是不可能真实发生的值 |
| 方差 | 和标准差可以互换使用 | 方差与 X 量纲平方，标准差与 X 同量纲 |
| 交易日 | 一年365天或250天 | A股约252个交易日（年化标准值） |

---

## 延伸阅读

- *Options, Futures, and Other Derivatives* — John Hull (第1章) — 对数正态分布与连续复利的经典推导，是量化金融的基础教材
- *Active Portfolio Management* — Grinold & Kahn — 第2章系统介绍了收益率的统计性质，是组合管理的标准参考
- [Investopedia: Logarithmic Returns](https://www.investopedia.com/articles/investing/070815/calculating-logarithmic-returns.asp) — 适合入门的直观解释，包含实际案例
- *Quantitative Risk Management* — McNeil, Frey, Embrechts — 第2章深入讨论金融收益率的统计分布，适合进阶学习
