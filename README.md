# 自适应边界偏移 SMOTE（ABS-SMOTE）

本仓库给出了本科毕业论文:针对不平衡分类任务的过采样算法研究中所提出的
**自适应边界偏移 SMOTE（Adaptive Boundary-Shift SMOTE，ABS-SMOTE）**
算法的 Python 参考实现。

ABS-SMOTE 是一种面向不平衡分类问题的过采样方法。
与传统 SMOTE 算法相比，该方法在生成合成少数类样本时引入了
类别边界信息，通过自适应偏移机制使合成样本远离多数类决策边界，
从而缓解类别重叠问题，提高对边界样本的分类性能。

---

## 算法原理概述

ABS-SMOTE 在标准 SMOTE 的基础上进行了改进，其核心思想如下：

1. 在少数类样本内部搜索近邻样本；
2. 对每个少数类样本，计算其与最近多数类样本之间的距离，以刻画边界压力；
3. 采用 SMOTE 插值方式生成初始合成样本；
4. 沿远离多数类样本的安全方向，对合成样本施加自适应偏移。

通过上述过程，生成的合成样本能够更合理地分布于少数类空间，
并降低生成样本落入类别模糊区域的风险。

---

ABS-SMOTE/
├── sampling/
│ └── abs_smote.py ABS-SMOTE 算法核心实现/
├── requirements.txt 运行所需依赖库/
└── README.md 英文说明文档

# Adaptive Boundary-Shift SMOTE (ABS-SMOTE)

This repository provides the reference Python implementation of the
Adaptive Boundary-Shift SMOTE (ABS-SMOTE) algorithm proposed in the
undergraduate thesis.

ABS-SMOTE is an oversampling method designed for imbalanced classification
tasks. Compared with the standard SMOTE algorithm, ABS-SMOTE adaptively
shifts synthetic samples away from the majority class boundary, aiming to
reduce class overlap and improve the classification performance on
borderline minority samples.

---

## Algorithm Overview

The core idea of ABS-SMOTE is to incorporate boundary information into the
sample generation process. For each minority class sample, the algorithm:

1. Identifies nearest neighbors within the minority class.
2. Locates the nearest majority class sample to estimate boundary pressure.
3. Generates synthetic samples using standard SMOTE interpolation.
4. Applies an adaptive shift along a safe direction away from the majority
   class boundary.

This strategy allows synthetic samples to better represent the minority
class distribution while reducing the risk of generating samples in
ambiguous regions.

---
