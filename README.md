<img width="517" height="457" alt="imgs_ruler_for_rule_logo" src="https://github.com/user-attachments/assets/0c50dd4c-3fa6-455d-acf2-f329480b59d8" />

# jupyter_rule_finder-Ruler 
# — AntiRisk Lab

**Jupyter 交互式风控规则挖掘与策略组合监控工具(Ruler)**

> *"Data defines the rule; strategy governs the risk."*

***

## 项目背景

在信贷风控业务中，策略分析师需要在海量申请数据中挖掘有效的规则（如 `debt_ratio > 0.6`、`credit_score <= 580`），并将多条规则组合成可落地的风控策略。这一过程通常涉及：

1. **规则发现** — 从数十个特征中高效提取具有区分度的规则
2. **规则筛选** — 按 Lift / Hit Rate 对规则进行分类和评估
3. **策略组合** — 将多条规则合并，评估组合后的通过率、坏样本命中率、Lift 等指标
4. **随机方向** — 随机组合各类策略形成随机策略池，用统计循环采用的方式观察最优策略组合方向
5. **策略监控** — 对策略上线后的效果进行持续可视化追踪

传统的做法依赖 Excel + SQL 多次查询，流程割裂、效率低下。**Ruler** 将这四个环节整合进 Jupyter Notebook 的单一交互界面，让策略分析师在同一页面完成从规则生成到监控分析的全流程闭环。

***

## 项目特色

### 1. 全流程闭环

在一个 Jupyter Notebook Cell 内初始化后，即可在同一界面完成规则挖掘 → 规则筛选 → 策略集成 → 策略监控四种模式的无缝切换。

### 2. 交互式 Widget 界面

基于 `ipywidgets` 构建纯 Widget UI，所有参数滑块、下拉选单、按钮均为实时交互，无需重跑 Cell。

### 3. 多算法规则生成

- **Decision Tree** — 单棵树规则提取，可控制树深与最小叶子样本比
- **Random Forest** — 多棵树集成规则挖掘，可控制树数量、深度和特征采样比
- **Demo Rules** — 基于统计分位数的模拟规则生成，用于快速原型验证

### 4. 三档规则分类

所有规则按 Lift 自动分为 High / Mid / Low 三档，阈值可拖拽调整，支持多选并添加到策略组合。

<img width="1036" height="760" alt="493a8888-b8b0-440c-9911-fa8f10d7a9eb" src="https://github.com/user-attachments/assets/e3f8be37-4e27-4f04-8c04-8a4ec2394d8b" />

### 5. 策略集成算法

- **Greedy (Lift Priority)** — 贪心策略，优先选取高 Lift 规则，通过命中占比上限和 Lift 提升阈值控制过度命中
- **Random Path** — 随机组合搜索，在组合空间中采样 50\~200 条路径，返回 Lift Top 5 且满足命中占比的规则集

<img width="1040" height="695" alt="d0ac7461-3dc0-42e4-842c-2bf8853bd6b2" src="https://github.com/user-attachments/assets/6567412a-739b-485b-891e-8ac241e05dc7" />

### 6. 四方象限监控图

基于新旧策略的通过/拒绝交叉矩阵，构建四个象限的对比分析：

- **主 y 轴**: 各象限中新旧策略各自的通过占比（Through Ratio per Strategy）
- **副 y 轴**: 各象限中新旧策略的 Bad Rate 对比（散点 + 差值连线）
- 直观展示策略替换后的 Swap In / Swap Out 效果

### 7. 时间序列可视化

- Pass Rate 与 Bad Rate 双轴时间序列图（主轴 0%-100%，副轴动态范围）
- Pass Rate vs Bad Rate 散点图（时间渐变色，越近期越深）
- 简洁配色

<img width="1055" height="830" alt="ba0b0a46-4f79-4360-8db5-e952c215af49" src="https://github.com/user-attachments/assets/3f19dcab-b1a9-47fb-aef9-4efea98da6d1" />

***

## 快速开始

### 安装

```bash
cd jupyter_ruler
pip install -e .
```

依赖项：

- `ipywidgets >= 7.6`
- `plotly >= 4.0`
- `pandas >= 1.0`
- `numpy`
- `scikit-learn >= 0.24`

### 基础示例

打开 `example.ipynb` 并依次执行以下 Cell 即可启动交互界面。

#### Step 0 — 导入模块

```python
import pandas as pd
import numpy as np
from jupyter_ruler import RuleStrategyApp
```

#### Step 1 — 构造信贷申请数据

模拟一份包含 **5,000 条申请记录**、覆盖 **用户行为 + 财务 + 时间维度** 的完整数据集：

```python
np.random.seed(0)
n = 5000
dates = pd.date_range('2023-01-01', '2024-12-31', periods=n)
df = pd.DataFrame({
    # 用户行为特征
    'query_count': np.random.randint(1, 50, n),      # 查询次数
    'overdue_count': np.random.randint(0, 10, n),    # 逾期次数
    'account_count': np.random.randint(1, 20, n),    # 账户数量
    'page_views': np.random.randint(5, 200, n),      # 浏览页面数
    'click_count': np.random.randint(10, 500, n),    # 点击数
    'product_count': np.random.randint(1, 15, n),    # 产品数量
    'credit_history': np.random.randint(0, 30, n),   # 历史授信次数
    # 财务特征
    'income': np.random.normal(8000, 3000, n).clip(2000, 20000),
    'debt_ratio': np.random.uniform(0.1, 0.8, n),    # 负债率
    'credit_score': np.random.randint(300, 850, n),  # 信用评分
    # 时间
    'apply_time': dates,
})
```

#### Step 2 — 衍生业务标签

构建风控场景中的关键标签字段：

```python
# 客群评级 A > B > C > D
df['customer_rating'] = np.random.choice(['A','B','C','D'], n,
    p=[0.25, 0.35, 0.28, 0.12])

# 通过标签：1=通过，2=拒绝（目标通过率 70%）
df['pass_label'] = np.random.choice([1, 2], n, p=[0.70, 0.30])

# 转人工审核（约 5% 申请，通过率 60%）
df['manual_review'] = np.where(np.random.random(n) < 0.05,
    np.random.choice([1, 2], n, p=[0.6, 0.4]), np.nan)

# 目标变量：MOB3 30+ 或 MOB6 60+ 逾期（坏样本率约 5%）
df['overdue_mob3'] = np.random.choice([0, 1], n, p=[0.96, 0.04])
df['overdue_mob6'] = np.random.choice([0, 1], n, p=[0.98, 0.02])
df['target'] = (df['overdue_mob3'] | df['overdue_mob6']).astype(int)

# 产品信息
df['product_name'] = np.random.choice(
    ['信用贷','消费贷','经营贷','车贷','房贷'], n,
    p=[0.35, 0.25, 0.15, 0.15, 0.10])
df['term'] = np.random.choice([6,12,24,36,48,60], n)
df['loan_amount'] = np.random.randint(5000, 500000, n)
```

数据预览输出：

```
总申请样本: 5000
通过样本: 3520
通过率: 70.4%
坏样本率: 5.65%

转人工审核数: 223
转人工率: 4.5%

数据时间范围: 2023-01-01 至 2024-12-31
```

#### Step 3 — 启动交互界面

```python
feature_cols = [
    'query_count', 'overdue_count', 'account_count',
    'page_views', 'click_count', 'product_count', 'credit_history',
    'income', 'debt_ratio', 'credit_score'
]

app = RuleStrategyApp(
    df=df,
    feature_cols=feature_cols,
    target='target',
    date_col='apply_time'
)
app.display()
```

执行后将在 Jupyter Notebook 中渲染出包含四个功能模块的完整交互界面。

***

## 交互界面详解

执行 `app.display()` 后，界面从上到下分为四个区块，每个区块以圆角卡片形式呈现：

```
┌──────────────────────────────────────────────────┐
│            Ruler — AntiRisk Lab                  │
│  Interactive Rule Mining & Strategy Monitoring   │
├──────────────────────────────────────────────────┤
│  ❖ Dataset Information — 数据集信息              │
│  ┌──────────────────────────────────────────┐   │
│  │ [Field: ▼] [Dev Start: ▼] [Dev End: ▼]    │   │
│  │ [OOT Start: ▼] [Execute Split]            │   │
│  │                                            │   │
│  │  ┌────────┬───────┬──────┬────────┬─────┐ │   │
│  │  │Dataset │ Total │ Pass │Reject  │Bad  │ │   │
│  │  ├────────┼───────┼──────┼────────┼─────┤ │   │
│  │  │ Dev    │ 1,667 │1,192 │ 475    │5.2% │ │   │
│  │  │ Val    │ 1,667 │1,166 │ 501    │6.1% │ │   │
│  │  │ OOT    │ 1,666 │1,163 │ 503    │5.5% │ │   │
│  │  └────────┴───────┴──────┴────────┴─────┘ │   │
│  │                                            │   │
│  │  [柱状图: 月度 Volume]                      │   │
│  │  [折线图: 月度 Pass Rate + Bad Rate]        │   │
│  └──────────────────────────────────────────┘   │
├──────────────────────────────────────────────────┤
│  ♙ Rule Mining — 规则发现                        │
│  ┌──────────────────────────────────────────┐   │
│  │ [Min Leaf Ratio: ─●──] [Rule Depth: ─●─]  │   │
│  │ [N Trees: ──●───] [Feature Ratio: ─●──]   │   │
│  │ [🌲 Decision Tree]  [🌲 Random Forest]     │   │
│  │ ┌──────────────────────────────────────┐  │   │
│  │ │ ■ High Lift ≥ 2.5                    │  │   │
│  │ │ [H:3.2%] [L:4.52] overdue_count > 2  │  │   │
│  │ │ [H:4.1%] [L:3.81] debt_ratio > 0.6   │  │   │
│  │ │ ...                                   │  │   │
│  │ └──────────────────────────────────────┘  │   │
│  │ ┌─────────────────┬──────────────────┐   │   │
│  │ │ ◆ Mid (L≥1.8)   │ ▲ Low (L≥1.2)    │   │   │
│  │ └─────────────────┴──────────────────┘   │   │
│  │           [+ Add to Strategy]            │   │
│  └──────────────────────────────────────────┘   │
├──────────────────────────────────────────────────┤
│  ♖ Strategy Integration — 策略集成               │
│  ┌──────────────────────────────────────────┐   │
│  │ [Hit Ratio Cap: ─●──] [Greedy] [Random]  │   │
│  │ ┌────────────────┐ ┌───────────────────┐ │   │
│  │ │ Strategy Rules  │ │ Strategy Metrics  │ │   │
│  │ │ L:3.81 H:3.2%   │ │ Lift        3.81  │ │   │
│  │ │ debt_ratio>0.6  │ │ Pass Rate  12.5%  │ │   │
│  │ │                 │ │ Hit Bad Rate 21.5% │ │   │
│  │ └────────────────┘ └───────────────────┘ │   │
│  └──────────────────────────────────────────┘   │
├──────────────────────────────────────────────────┤
│  ◎ Strategy Monitor — 策略监控                   │
│  ┌──────────────────────────────────────────┐   │
│  │ [Sample: Dev ▼] [Freq: Week ▼] [Show]    │   │
│  │                                            │   │
│  │  ┌──────────────┐  ┌───────────────────┐  │   │
│  │  │ Pass Rate &   │  │ Pass Rate vs      │  │   │
│  │  │ Bad Rate      │  │ Bad Rate          │  │   │
│  │  │ (双轴时序图)   │  │ (时间渐变色散点)   │  │   │
│  │  ├──────────────┤  └───────────────────┘  │   │
│  │  │ Swap In/Out   │                        │   │
│  │  │ Quadrant      │    (预留区域)          │   │
│  │  │ Analysis      │                        │   │
│  │  │ (四方象限图)   │                        │   │
│  │  └──────────────┘                        │   │
│  └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

### 模块一：Dataset Information — 数据集信息

界面顶部显示应用标题和副标题，下方卡片内包含三个控制行：

**样本分割控件**：

- `Field` 下拉选单：切换 `apply_time`（时间分割）或 `customer_rating` / `pass_label`（标签分割）
- 四个时间/标签下拉选单：分别设定 Dev Start / Dev End / OOT Start 的范围
- `Execute Split` 按钮：执行分割，生成 Dev / Val / OOT 三个子集

**数据集统计表格**：

| Dataset | Total | Pass  | Reject | Pass Rate | Bad Rate | Manual Rate |
| ------- | ----- | ----- | ------ | --------- | -------- | ----------- |
| Dev     | 1,667 | 1,184 | 483    | 71.0%     | 5.52%    | 4.3%        |
| Val     | 1,667 | 1,193 | 474    | 71.6%     | 5.34%    | 4.5%        |
| OOT     | 1,666 | 1,171 | 495    | 70.3%     | 5.88%    | 4.6%        |

**数据集趋势图**：

- 柱状图：各月度的申请量 Volume（左侧 y 轴）
- 双折线：各月度的 Pass Rate（实线）和 Bad Rate（虚线）（右侧 y 轴）
- 三个数据集分别以蓝/橙/绿三色区分

### 模块二：Rule Mining — 规则发现

**算法参数**（四个滑块联动）：

- `Min Leaf Ratio`：最小叶子节点样本占比（0.1% \~ 10%），控制规则粒度
- `Rule Depth`：决策树最大深度（2 \~ 8），深度越大规则越精细
- `N Trees`：随机森林的树数量（10 \~ 500），仅在 Random Forest 时生效
- `Feature Ratio`：每棵树的特征采样比例（10% \~ 100%）

**操作流程**：

1. 点击 `Decision Tree` 按钮 → 进度条显示训练进度 → 生成约 20\~40 条规则
2. 点击 `Random Forest` 按钮 → 进度条逐步完成 → 生成约 15\~30 条去重规则

**规则分类示例**：

**■ High Lift** （Lift ≥ 2.5，用红色标注）：

```
[H:3.8%] [L:4.32] [V1] query_count > 42.50
[H:2.9%] [L:3.95] [V1] overdue_count > 5.00
[H:4.2%] [L:3.17] [V1] debt_ratio > 0.72
[H:3.1%] [L:2.84] [V1] credit_score <= 385.00
```

**◆ Mid Lift** （1.8 ≤ Lift < 2.5，用金色标注）：

```
[H:6.5%] [L:2.31] [V1] income <= 4200.00
[H:8.1%] [L:2.15] [V1] account_count > 12.00
```

**▲ Low Lift** （1.2 ≤ Lift < 1.8，用绿色标注）：

```
[H:12.3%] [L:1.55] [V1] page_views > 100.00
[H:15.0%] [L:1.42] [V1] click_count <= 120.00
```

每个规则显示 `[Hit Rate] [Lift] [Variable Count]` 前缀信息，鼠标悬停可查看完整规则描述。

**Lift 阈值调节**：

- 三个 FloatSlider（280px 宽）可拖拽调整 High / Mid / Low 的 Lift 分界阈值
- 拖动后三栏列表实时刷新

**添加到策略**：

- 在任意栏中多选规则 → 点击 `+ Add to Strategy` → 规则进入策略集成模块

### 模块三：Strategy Integration — 策略集成

**策略集成算法**：

**Greedy (Lift Priority)**：

- 参数：`Hit Ratio Cap`（命中占比上限，默认 35%）
- 算法逻辑：按 Lift 降序排列所有规则 → 逐一尝试加入策略 → 仅当加入后策略整体 Lift 有显著提升（满足 lift\_threshold 或 hit\_increment\_threshold）且命中占比未超过上限时保留
- 适用场景：需要高 Lift 策略且对命中占比有硬性约束

**Random Path**：

- 参数：`N Combinations`（随机组合次数，10 \~ 200）、`Random Hit Cap`（随机路径命中占比上限）
- 算法逻辑：在规则组合空间中随机采样 N 条路径 → 每条路径随机选择 k 条规则组合 → 过滤命中占比不满足条件的 → 按 Lift 排序返回 Top 5
- 适用场景：规则库较大时探索非贪心的更优组合

**策略指标展示**：

当策略中已添加规则后，右侧 `Strategy Metrics` 面板实时计算并展示：

```
┌──────────────────────┐
│ Strategy Metrics     │
├──────────────────────┤
│ Lift           2.97  │
│ Pass Rate     15.2%  │
│ Reject Rate   84.8%  │
│ Hit Ratio     15.2%  │
│ Hit Bad Rate  16.8%  │
│ Overall Bad    5.6%  │
└──────────────────────┘
```

**指标说明**：

| 指标               | 计算方式                             | 含义                  |
| ---------------- | -------------------------------- | ------------------- |
| Lift             | Bad Rate(Hit) / Overall Bad Rate | 策略对坏样本的捕捉倍数         |
| Pass Rate        | 通过数 / 总数                         | 策略通过率               |
| Reject Rate      | 拒绝数 / 总数                         | 策略拒绝率               |
| Hit Ratio        | 命中数 / 总数                         | 策略覆盖占比（同 Pass Rate） |
| Hit Bad Rate     | 命中样本中坏样本率                        | 策略命中的坏账浓度           |
| Overall Bad Rate | 整体坏样本率                           | 基线坏账率               |

### 模块四：Strategy Monitor — 策略监控

**控制面板**：

- `Sample` 下拉选单：切换 Dev / Val / OOT 样本集
- `Frequency` 下拉选单：聚合粒度 Day / Week / Month
- `Show Monitor` 按钮：生成四合一监控图表

**Chart 1 — Pass Rate & Bad Rate 时序图（左上）**：

- 主 y 轴（左）：Pass Rate，固定范围 0% \~ 100%
- 副 y 轴（右）：Bad Rate，动态范围（min-2% \~ max+2%，下限不低于 0%）
- 蓝色实线 = Base 策略 Pass Rate，蓝色虚线 = Base 策略 Bad Rate
- 橙色实线 = New 策略 Pass Rate，橙色虚线 = New 策略 Bad Rate
- 小圆点/菱形标记各数据点

**Chart 2 — Pass Rate vs Bad Rate 散点图（右上）**：

- x 轴：Pass Rate，y 轴：Bad Rate
- 每个点代表一个时间窗口（如一周）的策略表现
- **颜色渐变**：越靠近当前时点的点颜色越深（透明度 0.25 → 1.0）
  - 蓝色系 = Base 策略点，橙色系 = New 策略点
- y 轴动态范围：min-2% \~ max+2%（下限不低于 0%）
- 鼠标悬停显示具体 Pass Rate / Bad Rate 数值

**Chart 3 — Swap In/Out 四方象限分析图（左下）**：

基于新旧策略交叉矩阵（Base Pass/Reject × New Pass/Reject）构建四个象限：

```
       New Pass        New Reject
     ┌──────────┬──────────────┐
Base │ QⅠ       │ QⅡ           │
Pass │Both Pass │Base→Reject   │
     ├──────────┼──────────────┤
Base │ QⅢ       │ QⅣ           │
Rej. │New→Pass  │Both Reject   │
     └──────────┴──────────────┘
```

| 象限             | 含义      | 业务解读                     |
| -------------- | ------- | ------------------------ |
| QⅠ Both Pass   | 新旧策略都通过 | 不受策略变更影响的客群              |
| QⅡ Base→Reject | 原通过、新拒绝 | **Swap Out**：被新策略额外拦截的客群 |
| QⅢ New→Pass    | 原拒绝、新通过 | **Swap In**：被新策略额外放行的客群  |
| QⅣ Both Reject | 新旧策略都拒绝 | 始终被拦截的高风险客群              |

**主 y 轴**（柱状图）：各象限中新旧策略各自的 **通过占比**（Through Ratio per Strategy）

- 蓝色柱 = Base Strategy，橙色柱 = New Strategy
- 标签显示具体百分比

**副 y 轴**（散点 + 虚线）：各象限中新旧策略各自的 **Bad Rate**

- 蓝色圆点 = Base Bad Rate，橙色三角 = New Bad Rate
- 红色虚线连接同象限内新旧策略的 Bad Rate（Swap Differential）
- 范围：min-2% \~ max+2%（下限不低于 0%）

**典型分析场景**：

- QⅡ 的 New Bad Rate 应显著高于整体 Bad Rate（被拦截的是真坏样本）
- QⅢ 的 New Bad Rate 应接近整体 Bad Rate（放行的是好样本）
- 红色虚线越长，说明策略替换在该象限的效果差异越大

***

## 项目结构

```
jupyter_ruler/
├── jupyter_ruler/
│   ├── __init__.py          # 模块入口，导出 RuleStrategyApp
│   ├── app.py               # 主交互界面 (ipywidgets + Plotly)
│   │                        #   - 样本分割 & 数据集信息
│   │                        #   - 规则搜索 & 筛选
│   │                        #   - 策略集成 & 指标展示
│   │                        #   - 策略监控控制面板
│   ├── rule_mining.py       # 规则挖掘引擎
│   │                        #   - generate_demo_rules (统计模拟)
│   │                        #   - generate_decision_tree_rules
│   │                        #   - generate_random_forest_rules
│   │                        #   - classify_rules (三档分类)
│   │                        #   - deduplicate_rules (去重)
│   ├── strategy_builder.py  # 策略组合算法
│   │                        #   - greedy_lift_select (贪心Lift优先)
│   │                        #   - random_path_search (随机路径)
│   │                        #   - compute_strategy_stats (策略指标)
│   └── monitor.py           # 策略监控可视化
│                            #   - aggregate_stats (聚合统计)
│                            #   - swap_analysis (四方象限分析)
│                            #   - plot_strategy_monitor (监控图表)
├── example.ipynb            # 示例 Notebook
├── setup.py                 # 安装脚本
└── README.md                # 项目文档
```

***

## 未来开发规划

### 策略集成算法增强（高优先级）

当前 Greedy 和 Random Path 是两个独立算法，后续计划加入更多策略组合方法：

| 算法                          | 描述                                             | 适用场景           |
| --------------------------- | ---------------------------------------------- | -------------- |
| **Genetic Algorithm**       | 遗传算法搜索 Pareto 最优规则集，以 Lift 和 Hit Rate 作为多目标适应度 | 大规模规则库中寻找非支配解  |
| **Simulated Annealing**     | 模拟退火随机优化，从初始策略出发以概率接受劣解，逐步收敛                   | 规则数量多、搜索空间不规则  |
| **Beam Search**             | 束搜索，每步保留 Top-K 候选组合，平衡搜索深度与广度                  | 需要控制命中占比的渐进策略  |
| **Submodular Optimization** | 基于次模函数最大化，通过边际增益选择规则，有理论近似保证                   | 需要可证明性能边界的生产策略 |
| **Constraint Programming**  | 将多个约束（命中率 ≤ 35%、拒绝率 ≥ 20%）直接建模求解               | 多硬约束场景         |

### 规则挖掘增强

- **XGBoost / LightGBM 规则提取** — 基于 GBDT 的分裂路径提取非线性组合规则
- **关联规则挖掘 (Apriori / FP-Growth)** — 从多特征交互中发现频繁出现的特征组合模式
- **规则稳定性分析** — 跨时间窗口（Dev / Val / OOT）评估规则的 Lift 衰减

### 策略监控增强

- **KS / AUC 趋势追踪** — 在监控页增加 KS 曲线和 AUC 的时间序列
- **PSI 稳定性指标** — 自动计算各特征在新旧策略切换后的 PSI
- **策略自动预警** — 当通过率 / Bad Rate 偏离阈值时自动高亮提示
- **A/B Test 集成** — 支持多策略分流的线上实验效果对比

### 交互体验优化

- **规则拖拽排序** — 支持在策略列表中拖拽调整规则优先级
- **一键导出报告** — 将监控图表和策略指标导出为 PDF 报告
- **策略版本管理** — 保存和加载历史策略配置
- **多策略并行对比** — 支持同时加载 3+ 策略在监控图中对比

***

## License

MIT License

***

*Built with passion for risk control strategy analysts.*
