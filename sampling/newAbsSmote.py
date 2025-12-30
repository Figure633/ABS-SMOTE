"""
改进版自适应边界偏移SMOTE (Improved Adaptive Boundary-Shift SMOTE)

本版本添加了边界偏移幅度约束，防止生成点远离真实数据流形，
避免成为"孤立噪声点"，从而解决样本多时性能下降的问题。

改进点：
1. 添加 beta 参数，控制偏移幅度的上界
2. 约束条件：||α· Δboundary || ≤ β· ||Xnn - Xi ||
3. 防止偏移过大，确保生成样本在合理范围内
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter


class ImprovedAdaptiveBoundaryShiftSMOTE:
    """
    改进版自适应边界偏移SMOTE (Improved Adaptive Boundary-Shift SMOTE)
    
    相比原版本，添加了边界偏移幅度约束，防止生成点远离真实数据流形。

    参数:
    -----------
    k : int, 可选 (默认=5)
        生成合成样本时使用的近邻数量

    n_samples : int 或 float, 可选 (默认=1.0)
        如果是int, 指定要生成的合成样本的确切数量
        如果是float, 指定相对于原始少数类样本的比例

    alpha : float, 可选 (默认=0.5)
        偏移强度参数，控制样本远离多数类的偏移程度

    beta : float, 可选 (默认=0.5)
        边界偏移幅度约束参数，控制偏移的上界
        约束条件：||α· Δboundary || ≤ β· ||Xnn - Xi ||
        beta 越小，偏移限制越严格，生成样本越保守
        beta 越大，允许更大的偏移，但可能生成远离流形的点

    epsilon : float, 可选 (默认=1e-8)
        小常数，避免除以零

    random_state : int 或 None, 可选 (默认=None)
        随机种子，用于结果复现
    """

    def __init__(self, k=5, n_samples=1.0, alpha=0.5, beta=0.5, epsilon=1e-8, random_state=None):
        self.k = k
        self.n_samples = n_samples
        self.alpha = alpha
        self.beta = beta  # 新增：边界偏移幅度约束参数
        self.epsilon = epsilon
        self.random_state = random_state

    def fit_resample(self, X, y):
        """
        重采样数据集

        参数:
        -----------
        X : array-like, shape (n_samples, n_features)
            输入样本

        y : array-like, shape (n_samples,)
            目标值

        返回:
        --------
        X_resampled : array-like, shape (n_samples_new, n_features)
            重采样后的输入样本

        y_resampled : array-like, shape (n_samples_new)
            重采样后的目标值
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 将输入转换为numpy数组
        X, y = np.array(X), np.array(y)

        # 找出少数类和多数类
        class_counts = Counter(y)
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)

        # 分离少数类和多数类实例
        X_min = X[y == minority_class]
        X_maj = X[y == majority_class]

        if len(X_min) == 0:
            return X.copy(), y.copy()

        # 计算要生成的样本数量
        if isinstance(self.n_samples, float):
            num_to_generate = int(self.n_samples * len(X_min))
        else:
            num_to_generate = self.n_samples

        if num_to_generate <= 0:
            return X.copy(), y.copy()

        # 为少数类样本找到最近邻(在少数类内部)
        nn_min = NearestNeighbors(n_neighbors=self.k + 1).fit(X_min)
        min_indices = nn_min.kneighbors(X_min, return_distance=False)[:, 1:]

        # 为每个少数类样本找到最近的多数类邻居
        nn_maj = NearestNeighbors(n_neighbors=1).fit(X_maj)
        maj_distances, maj_indices = nn_maj.kneighbors(X_min)
        maj_distances = maj_distances.flatten()

        # 计算每个少数类样本的边界压力
        boundary_pressure = 1.0 / (maj_distances + self.epsilon)

        # 找出从少数类样本到最近的多数类样本的方向
        nearest_maj_samples = np.array([X_maj[maj_indices[i][0]] for i in range(len(X_min))])
        safe_directions = X_min - nearest_maj_samples

        # 标准化安全方向
        norms = np.sqrt(np.sum(safe_directions ** 2, axis=1)).reshape(-1, 1)
        norms[norms == 0] = 1.0  # 避免除以零
        safe_directions = safe_directions / norms

        # 生成合成样本
        synthetic_samples = []
        synthetic_labels = []

        for _ in range(num_to_generate):
            # 随机选择一个少数类实例
            idx = np.random.randint(0, len(X_min))

            # 获取其邻居的索引
            nn_indices = min_indices[idx]

            # 随机选择其中一个邻居
            neighbor_idx = np.random.choice(nn_indices)

            # 获取原始样本及其邻居
            sample = X_min[idx]
            neighbor = X_min[neighbor_idx]

            # 标准SMOTE插值
            gap = np.random.random()
            std_sample = sample + gap * (neighbor - sample)

            # 基于边界压力计算自适应偏移
            shift = self.alpha * boundary_pressure[idx] * safe_directions[idx]

            # ============================================================
            # 新增：边界偏移幅度约束
            # 约束条件：||α· Δboundary || ≤ β· ||Xnn - Xi ||
            # 即：||shift|| ≤ beta * ||neighbor - sample||
            # ============================================================
            
            # 计算插值向量的范数 ||Xnn - Xi||
            interpolation_vector = neighbor - sample
            interpolation_norm = np.linalg.norm(interpolation_vector)
            
            # 计算偏移的范数 ||shift||
            shift_norm = np.linalg.norm(shift)
            
            # 计算允许的最大偏移范数
            max_shift_norm = self.beta * interpolation_norm
            
            # 如果偏移超过限制，按比例缩放
            if shift_norm > max_shift_norm:
                # 保持方向不变，只缩放幅度
                scale_factor = max_shift_norm / (shift_norm + self.epsilon)
                shift = shift * scale_factor
            
            # 最终的合成样本
            new_sample = std_sample + shift

            synthetic_samples.append(new_sample)
            synthetic_labels.append(minority_class)

        # 合并原始样本和合成样本
        X_resampled = np.vstack([X, np.array(synthetic_samples)])
        y_resampled = np.hstack([y, np.array(synthetic_labels)])

        return X_resampled, y_resampled

