# MovieLens 10M
# Chunk, 中心化, ALS-WR, Soft-Impute

import time
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz, csr_matrix
from scipy.sparse.linalg import svds

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 对所有观测评分集进行操作的块大小，防止内存溢出
CHUNK_SIZE = int(2_000_000)


def compute_global_user_item_means(train_csr):
    """
    中心化：计算全局平均值、用户平均值/偏置、电影平均值/偏置
    我们将计算：
        mu = 所有已观察到评分的平均值
        b_u = 用户 u 的评分平均值（i 属于 I(u)）减去 mu
        b_i = 电影 i 的评分平均值（u 属于 U(i)）减去 mu
    :return r_center = r - mu - b_u[u] - b_i[i]
    """
    rows, cols = train_csr.nonzero()
    ratings = train_csr.data.astype(np.float32)
    mu = float(np.mean(ratings))

    # 用户均值
    n_users = train_csr.shape[0]
    b_u = np.zeros(n_users, dtype=np.float32)
    counts_u = np.zeros(n_users, dtype=np.int32)
    for r, rating in zip(rows, ratings):
        b_u[r] += rating
        counts_u[r] += 1
    nonzero_u = counts_u > 0
    b_u[nonzero_u] = b_u[nonzero_u] / counts_u[nonzero_u] - mu

    # 电影均值
    n_items = train_csr.shape[1]
    b_i = np.zeros(n_items, dtype=np.float32)
    counts_i = np.zeros(n_items, dtype=np.int32)
    for c, rating in zip(cols, ratings):
        b_i[c] += rating
        counts_i[c] += 1
    nonzero_i = counts_i > 0
    b_i[nonzero_i] = b_i[nonzero_i] / counts_i[nonzero_i] - mu

    return mu, b_u, b_i, counts_u, counts_i


# ------------------------
# ALS (中心化， chunk化， WR加权正则)
# ------------------------
class PyTorchALS:
    """
    针对中心化后的评分残差进行因子分解，使用分块操作以避免内存溢出。
    传入原始训练稀疏矩阵
    """

    def __init__(self, rank=250, lambda_reg=0.02, max_iter=15, tol=1e-4, chunk_size=CHUNK_SIZE):
        self.rank = int(rank)
        self.lambda_reg = float(lambda_reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.chunk_size = int(chunk_size)

        # 中心化后的矩阵的学习因子
        self.U = None  # torch tensor [n_users, rank]
        self.V = None  # torch tensor [n_items, rank]
        # 和中心化有关的统计量
        self.global_mean = None
        self.user_bias = None  # numpy
        self.item_bias = None  # numpy

    def _initialize(self, n_users, n_items):
        self.U = torch.randn(n_users, self.rank, dtype=torch.float32, device=device) * 0.01
        self.V = torch.randn(n_items, self.rank, dtype=torch.float32, device=device) * 0.01

    def _chunked_preds(self, rows_t, cols_t):
        """对给定索引数组（torch tensor）分块计算预测"""
        n = len(rows_t)
        preds = torch.empty(n, dtype=torch.float32, device=device)
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            ru = rows_t[start:end]
            rv = cols_t[start:end]
            preds[start:end] = torch.sum(self.U[ru] * self.V[rv], dim=1)
        return preds

    def fit(self, train_csr):
        # 计算中心化的统计量
        mu, b_u, b_i, counts_u, counts_i = compute_global_user_item_means(train_csr)
        self.global_mean = float(mu)
        self.user_bias = b_u  # np array
        self.item_bias = b_i  # np array
        n_users, n_items = train_csr.shape
        self._initialize(n_users, n_items)

        # 准备观测数组
        rows, cols = train_csr.nonzero()
        ratings = train_csr.data.astype(np.float32)
        # 中心化评分 (r - mu - b_u - b_i)
        centered = ratings - (mu + b_u[rows] + b_i[cols])

        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        cols_t = torch.tensor(cols, dtype=torch.long, device=device)
        centered_t = torch.tensor(centered, dtype=torch.float32, device=device)

        prev_mse = float('inf')
        # ALS算法迭代
        for it in range(self.max_iter):
            # 更新用户
            unique_users = torch.unique(rows_t)
            for user in unique_users:
                mask = (rows_t == user)
                item_idx = cols_t[mask]
                y = centered_t[mask]  # 该用户对这些物品的中心化评分

                V_sub = self.V[item_idx]  # 提取这些物品的因子向量，k x r
                # eq: (V_sub^T V_sub + λ * |I(u)| I) u = V_sub^T y
                VTV = V_sub.T @ V_sub # 物品因子的协方差矩阵
                reg_scale = float(max(1, int((mask.sum().item()))))  # 正则化缩放因子：该用户的评分数量
                reg = self.lambda_reg * reg_scale * torch.eye(self.rank, device=device, dtype=torch.float32) # 正则化矩阵
                b = V_sub.T @ y
                # 求解线性方程得到当前用户的因子向量
                try:
                    self.U[user] = torch.linalg.solve(VTV + reg, b)
                except Exception:
                    self.U[user] = torch.linalg.pinv(VTV + reg) @ b

            # 更新电影
            unique_items = torch.unique(cols_t)
            for item in unique_items:
                mask = (cols_t == item)
                user_idx = rows_t[mask]
                y = centered_t[mask]

                U_sub = self.U[user_idx]  # k x r
                UTU = U_sub.T @ U_sub
                reg_scale = float(max(1, int((mask.sum().item()))))
                reg = self.lambda_reg * reg_scale * torch.eye(self.rank, device=device, dtype=torch.float32)
                b = U_sub.T @ y
                try:
                    self.V[item] = torch.linalg.solve(UTU + reg, b)
                except Exception:
                    self.V[item] = torch.linalg.pinv(UTU + reg) @ b

            # 计算中心化数据上的MSE
            preds = self._chunked_preds(rows_t, cols_t)
            mse = torch.mean((centered_t - preds) ** 2).item()
            print(f"[ALS] iter {it+1}/{self.max_iter}, centered MSE={mse:.6f}")
            if abs(prev_mse - mse) < self.tol:
                print(f"[ALS] converged at iter {it+1}")
                break
            prev_mse = mse

        return self

    def predict(self, user_idx, item_idx):
        uu = np.array(user_idx, dtype=np.int64)
        vv = np.array(item_idx, dtype=np.int64)
        n = len(uu)
        preds = np.empty(n, dtype=np.float32)

        # 分批处理
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            uu_chunk = torch.tensor(uu[start:end], dtype=torch.long, device=device)
            vv_chunk = torch.tensor(vv[start:end], dtype=torch.long, device=device)
            pred_chunk = torch.sum(self.U[uu_chunk] * self.V[vv_chunk], dim=1).cpu().numpy()
            preds[start:end] = pred_chunk

        # 添加偏置以还原中心化影响: mu + b_u + b_i
        preds = preds + self.global_mean + self.user_bias[uu] + self.item_bias[vv]
        preds = np.clip(preds, 1.0, 5.0)
        return preds


# ------------------------
# Soft-Impute (低秩svds，chunk化)
# ------------------------
class PyTorchSoftImpute:
    """
       对于大稀疏矩阵，采用一次性截断奇异值分解（svds）并对奇异值s进行软阈值处理
        - 计算截断奇异值分解 (U,s,Vt)
        - 收缩s := max(s - lambda, 0)
        - 存储(U_k, s_k, Vt_k)用于预测
       预测过程采用分块的行向点积： pred_ij = (U_k[i,:] * s_k) dot Vt_k[:, j]
    """

    def __init__(self, lambda_reg=1e-4, rank=250, chunk_size=CHUNK_SIZE):
        self.lambda_reg = float(lambda_reg)
        self.rank = int(rank)
        self.chunk_size = int(chunk_size)
        self.lowrank = None  # (U_k, s_k, Vt_k) - numpy on CPU
        # 中心化统计量
        self.mu = None
        self.b_u = None
        self.b_i = None

    def _svds(self, train_csr, k):
        # 使用scipy.sparse.linalg.svds，返回降序排序后的结果
        u, s, vt = svds(train_csr.astype(np.float64), k=k)
        idx = np.argsort(s)[::-1]
        return u[:, idx].astype(np.float32), s[idx].astype(np.float32), vt[idx, :].astype(np.float32)

    def fit(self, train_csr):
        # 1) 计算中心化数据 (mu, b_u, b_i)
        mu, b_u, b_i, _, _ = compute_global_user_item_means(train_csr)
        self.mu = mu
        self.b_u = b_u
        self.b_i = b_i

        # 2) 构建在CPU上的中心化稀疏矩阵作为svds的输入: r_center = r - mu - b_u[u] - b_i[v]
        rows, cols = train_csr.nonzero()
        vals = train_csr.data.astype(np.float32)
        centered_vals = vals - (mu + b_u[rows] + b_i[cols])
        # 构建带有中心化值的CSR矩阵
        A_center = csr_matrix((centered_vals, (rows, cols)), shape=train_csr.shape)

        # 3) 计算截断奇异值分解
        k = min(self.rank, min(train_csr.shape) - 1)
        if k <= 0:
            self.lowrank = (None, None, None)
            return self
        U, s, Vt = self._svds(A_center, k=k)

        # 4) 对s进行软阈值操作
        s_shrunk = np.maximum(s - self.lambda_reg, 0.0)
        nz = s_shrunk > 0
        if nz.sum() == 0:
            self.lowrank = (None, None, None)
            return self
        self.lowrank = (U[:, nz], s_shrunk[nz], Vt[nz, :])
        return self

    def predict(self, user_idx, item_idx):
        uu = np.array(user_idx, dtype=np.int64)
        vv = np.array(item_idx, dtype=np.int64)
        n = len(uu)
        preds = np.empty(n, dtype=np.float32)

        if self.lowrank is None or self.lowrank[0] is None:
            preds.fill(self.mu if self.mu is not None else 3.0)
            return np.clip(preds, 1.0, 5.0)

        U_k, s_k, Vt_k = self.lowrank  # U_k: m x r, Vt_k: r x n
        # 预计算V_k = Vt_k.T * s_k  -> 形状(n_items, r)
        V_k = (Vt_k.T * s_k.reshape(1, -1))

        # 分块点积
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            u_chunk = uu[start:end]
            v_chunk = vv[start:end]
            U_sub = U_k[u_chunk, :]    # chunk x r
            V_sub = V_k[v_chunk, :]    # chunk x r
            preds[start:end] = np.sum(U_sub * V_sub, axis=1)

        # 用偏置还原
        if self.mu is not None:
            preds = preds + self.mu + self.b_u[uu] + self.b_i[vv]

        return np.clip(preds, 1.0, 5.0)

# ------------------------
# 评估预测
# ------------------------
class MatrixCompletionEvaluator:
    def __init__(self, n_folds=5, chunk_size=CHUNK_SIZE):
        self.n_folds = int(n_folds)
        self.chunk_size = int(chunk_size)

    def evaluate(self, model_class, model_params, model_name=""):
        rmse_list = []
        times = []
        for fold in range(1, self.n_folds + 1):
            print("\n" + "=" * 60)
            print(f"Fold {fold} - {model_name}")
            print("=" * 60)
            try:
                train_matrix = load_npz(f"train_matrix_fold{fold}.npz").tocsr()
                test_df = pd.read_csv(f"test_set_fold{fold}.csv", engine='python')
                print(f"Train shape: {train_matrix.shape}, nnz={train_matrix.nnz}, test size={len(test_df)}")

                start = time.time()
                model = model_class(**model_params)
                model.fit(train_matrix)
                t = time.time() - start

                preds = model.predict(test_df['UserIdx'].values, test_df['MovieIdx'].values)
                test_df['Predicted'] = preds
                rmse = float(np.sqrt(np.mean((test_df['Rating'].values - preds) ** 2)))
                print(f"Fold {fold} RMSE: {rmse:.6f}")

                rmse_list.append(rmse)
                times.append(t)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in fold {fold}: {e}")
                import traceback
                traceback.print_exc()
                rmse_list.append(float('inf'))
                times.append(float('inf'))

        valid = [r for r in rmse_list if np.isfinite(r)]
        avg_rmse = float(np.mean(valid)) if valid else float('inf')
        avg_time = float(np.mean([tt for tt in times if np.isfinite(tt)])) if times else float('inf')
        return rmse_list, avg_rmse, avg_time


# ------------------------
# 运行两种实验算法
# ------------------------
def run_experiments():
    evaluator = MatrixCompletionEvaluator(n_folds=5, chunk_size=CHUNK_SIZE)

    results = []

    # Soft-Impute设置
    soft_configs = [
        {"name": "SoftImpute (svds λ=1e-4, r=250)", "params": {"lambda_reg": 1e-4, "rank": 250, "chunk_size": CHUNK_SIZE}},
        {"name": "SoftImpute (svds λ=5e-5, r=250)", "params": {"lambda_reg": 5e-5, "rank": 250, "chunk_size": CHUNK_SIZE}},
    ]

    for cfg in soft_configs:
        print("\n" + "-" * 60)
        print(f"Running {cfg['name']}")
        rmse_list, avg_rmse, avg_time = evaluator.evaluate(PyTorchSoftImpute, cfg['params'], cfg['name'])
        results.append({
            "Model": cfg['name'],
            "Avg RMSE": avg_rmse,
            "Avg Time (s)": avg_time,
            "Fold 1 RMSE": rmse_list[0] if len(rmse_list) > 0 else None,
            "Fold 2 RMSE": rmse_list[1] if len(rmse_list) > 1 else None,
            "Fold 3 RMSE": rmse_list[2] if len(rmse_list) > 2 else None,
            "Fold 4 RMSE": rmse_list[3] if len(rmse_list) > 3 else None,
            "Fold 5 RMSE": rmse_list[4] if len(rmse_list) > 4 else None,
        })


    # ALS设置
    als_configs = [
        {"name": "ALS (rank=250)", "params": {"rank": 250, "lambda_reg": 0.02, "max_iter": 30, "tol": 1e-4, "chunk_size": CHUNK_SIZE}},
        {"name": "ALS (rank=200)", "params": {"rank": 200, "lambda_reg": 0.02, "max_iter": 30, "tol": 1e-4, "chunk_size": CHUNK_SIZE}},
    ]

    for cfg in als_configs:
        print("\n" + "-" * 60)
        print(f"Running {cfg['name']}")
        rmse_list, avg_rmse, avg_time = evaluator.evaluate(PyTorchALS, cfg['params'], cfg['name'])
        results.append({
            "Model": cfg['name'],
            "Avg RMSE": avg_rmse,
            "Avg Time (s)": avg_time,
            "Fold 1 RMSE": rmse_list[0] if len(rmse_list) > 0 else None,
            "Fold 2 RMSE": rmse_list[1] if len(rmse_list) > 1 else None,
            "Fold 3 RMSE": rmse_list[2] if len(rmse_list) > 2 else None,
            "Fold 4 RMSE": rmse_list[3] if len(rmse_list) > 3 else None,
            "Fold 5 RMSE": rmse_list[4] if len(rmse_list) > 4 else None,
        })

    results_df = pd.DataFrame(results)
    print("\nFinal results:")
    print(results_df.to_string(index=False))
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"matrix_completion_results_final_{ts}.csv", index=False)
    print(f"Saved to matrix_completion_results_final_{ts}.csv")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    run_experiments()