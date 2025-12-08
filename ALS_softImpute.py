# MovieLens 10M
# Chunk, 中心化, ALS-WR, Soft-Impute

import time
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz, csr_matrix
from scipy.sparse.linalg import svds
import scipy.linalg as sla
from sklearn.utils.extmath import randomized_svd

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 对所有观测评分集进行操作的块大小，防止内存溢出
CHUNK_SIZE = int(2_000_000)

def compute_global_user_item_means(train_csr):
    """
    中心化：使用np.bincount计算全局平均值、用户平均值/偏置、电影平均值/偏置
    我们将计算：
        mu = 所有已观察到评分的平均值
        b_u = 用户 u 的评分平均值（i 属于 I(u)）减去 mu
        b_i = 电影 i 的评分平均值（u 属于 U(i)）减去 mu
    :return r_center = r - mu - b_u[u] - b_i[i]
    """
    rows, cols = train_csr.nonzero()
    ratings = train_csr.data.astype(np.float32)
    mu = float(ratings.mean())

    n_users, n_items = train_csr.shape
    # 用户
    sum_u = np.bincount(rows, weights=ratings, minlength=n_users).astype(np.float32)
    counts_u = np.bincount(rows, minlength=n_users).astype(np.int32)
    b_u = np.zeros(n_users, dtype=np.float32)
    mask_u = counts_u > 0
    b_u[mask_u] = sum_u[mask_u] / counts_u[mask_u] - mu
    # 电影
    sum_i = np.bincount(cols, weights=ratings, minlength=n_items).astype(np.float32)
    counts_i = np.bincount(cols, minlength=n_items).astype(np.int32)
    b_i = np.zeros(n_items, dtype=np.float32)
    mask_i = counts_i > 0
    b_i[mask_i] = sum_i[mask_i] / counts_i[mask_i] - mu

    return mu, b_u, b_i, counts_u, counts_i


# ------------------------
# ALS (中心化， chunk化， WR加权正则)
# ------------------------
class PyTorchALS:
    """
    ALS实现，中心化，用Cholesky解小的 r x r 线性系统
    针对中心化后的评分残差进行因子分解，使用分块操作以避免内存溢出
    传入原始训练稀疏矩阵
    """
    def __init__(self, rank=250, lambda_reg=0.02, max_iter=15, tol=1e-4, chunk_size=CHUNK_SIZE):
        self.rank = int(rank)
        self.lambda_reg = float(lambda_reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.chunk_size = int(chunk_size)
        # 因子
        self.U = None
        self.V = None
        # 中心化统计量（numpy）
        self.global_mean = None
        self.user_bias = None
        self.item_bias = None

    def _chunked_preds_torch(self, rows_t, cols_t):
        """torch版本分块预测"""
        n = len(rows_t)
        preds = torch.empty(n, dtype=torch.float32, device=device)
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            ru = rows_t[start:end]
            rv = cols_t[start:end]
            preds[start:end] = torch.sum(self.U[ru] * self.V[rv], dim=1)
        return preds

    def fit(self, train_csr):
        # 1) 计算中心化统计量
        mu, b_u, b_i, counts_u, counts_i = compute_global_user_item_means(train_csr)
        self.global_mean = float(mu)
        self.user_bias = b_u
        self.item_bias = b_i
        n_users, n_items = train_csr.shape

        # 2) 提取稀疏观测（numpy）
        rows, cols = train_csr.nonzero()
        ratings = train_csr.data.astype(np.float32)
        # centered = r - mu - b_u[rows] - b_i[cols]
        centered = ratings - (mu + b_u[rows] + b_i[cols])

        # 3) 构造 per-user / per-item 索引和观测值列表
        # 转成 numpy arrays
        user_items = [[] for _ in range(n_users)]
        user_vals = [[] for _ in range(n_users)]
        item_users = [[] for _ in range(n_items)]
        item_vals = [[] for _ in range(n_items)]

        for r, c, val in zip(rows, cols, centered):
            user_items[r].append(c)
            user_vals[r].append(val)
            item_users[c].append(r)
            item_vals[c].append(val)

        # 转成ndarray
        user_items = [np.array(l, dtype=np.int32) if l else np.array([], dtype=np.int32) for l in user_items]
        user_vals = [np.array(l, dtype=np.float32) if l else np.array([], dtype=np.float32) for l in user_vals]
        item_users = [np.array(l, dtype=np.int32) if l else np.array([], dtype=np.int32) for l in item_users]
        item_vals = [np.array(l, dtype=np.float32) if l else np.array([], dtype=np.float32) for l in item_vals]

        # 4) 初始化 U, V（在CPU上用numpy做求解）
        rng = np.random.RandomState(42)
        U_cpu = (rng.randn(n_users, self.rank).astype(np.float32) * 0.01)
        V_cpu = (rng.randn(n_items, self.rank).astype(np.float32) * 0.01)

        prev_mse = float('inf')

        # ALS 迭代，在CPU上用Cholesky解小系统
        for it in range(self.max_iter):
            # ---- 更新用户向量U ----
            for u in range(n_users):
                items = user_items[u]
                if items.size == 0:
                    U_cpu[u, :] = 0.0
                    continue
                V_sub = V_cpu[items, :] # k x r
                VTV = V_sub.T @ V_sub # r x r
                reg_scale = max(1, items.size)
                A = VTV + (self.lambda_reg * reg_scale) * np.eye(self.rank, dtype=np.float32)
                b = V_sub.T @ user_vals[u] # r,
                # 求解 Ax=b，优先用Cholesky
                try:
                    c, lower = sla.cho_factor(A, check_finite=False)
                    U_cpu[u, :] = sla.cho_solve((c, lower), b, check_finite=False)
                except Exception:
                    U_cpu[u, :] = np.linalg.solve(A, b)

            # ---- 更新电影向量V ----
            for i in range(n_items):
                users = item_users[i]
                if users.size == 0:
                    V_cpu[i, :] = 0.0
                    continue
                U_sub = U_cpu[users, :] # k x r
                UTU = U_sub.T @ U_sub
                reg_scale = max(1, users.size)
                A = UTU + (self.lambda_reg * reg_scale) * np.eye(self.rank, dtype=np.float32)
                b = U_sub.T @ item_vals[i]
                try:
                    c, lower = sla.cho_factor(A, check_finite=False)
                    V_cpu[i, :] = sla.cho_solve((c, lower), b, check_finite=False)
                except Exception:
                    V_cpu[i, :] = np.linalg.solve(A, b)

            # ---- 计算中心化后的MSE，使用分块numpy dot ----
            preds = np.empty(len(rows), dtype=np.float32)
            for start in range(0, len(rows), self.chunk_size):
                end = min(start + self.chunk_size, len(rows))
                ru = rows[start:end]
                rv = cols[start:end]
                preds[start:end] = np.sum(U_cpu[ru] * V_cpu[rv], axis=1)
            mse = float(np.mean((centered - preds) ** 2))
            print(f"[ALS] iter {it+1}/{self.max_iter}, centered MSE={mse:.6f}")

            if abs(prev_mse - mse) < self.tol:
                print(f"[ALS] converged at iter {it+1}")
                break
            prev_mse = mse

        # 训练结束，把U_cpu/V_cpu同步回torch用于predict
        self.U = torch.tensor(U_cpu, dtype=torch.float32, device=device)
        self.V = torch.tensor(V_cpu, dtype=torch.float32, device=device)

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

        # 恢复偏置
        preds = preds + self.global_mean + self.user_bias[uu] + self.item_bias[vv]
        return np.clip(preds, 1.0, 5.0)

# ------------------------
# Soft-Impute (低秩svds，chunk化)
# ------------------------
class PyTorchSoftImpute:
    """
    Soft-Impute实现
    使用randomized_svd优先加速，如果内存不足可回退到一次性稀疏svds
    """
    def __init__(self, lambda_reg=1e-4, rank=250, max_iter=50, tol=1e-4, randomized=True, chunk_size=CHUNK_SIZE):
        self.lambda_reg = float(lambda_reg)
        self.rank = int(rank)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.randomized = bool(randomized)
        self.chunk_size = int(chunk_size)
        self.lowrank = None  # (U_k, s_k, Vt_k) numpy
        # 中心化统计量
        self.mu = None
        self.b_u = None
        self.b_i = None

    def _truncated_svd(self, M, k):
        """
        封装：优先 randomized_svd（sklearn），若失败回退到scipy.sparse.linalg.svds
        M可以是稠密ndarray或scipy.sparse矩阵
        :return U (m x k), s (k,), Vt (k x n)
        """
        try:
            if self.randomized:
                # sklearn.randomized_svd能接受稀疏矩阵或ndarray
                U, s, Vt = randomized_svd(M, n_components=k, n_iter=7, random_state=42)
                return U.astype(np.float32), s.astype(np.float32), Vt.astype(np.float32)
            else:
                # fallback到svds
                from scipy.sparse.linalg import svds
                u, s, vt = svds(M.astype(np.float64), k=k)
                idx = np.argsort(s)[::-1]
                return u[:, idx].astype(np.float32), s[idx].astype(np.float32), vt[idx, :].astype(np.float32)
        except Exception as e:
            # 如果最初尝试randomized失败,回退尝试 svds
            try:
                from scipy.sparse.linalg import svds
                u, s, vt = svds(M.astype(np.float64), k=k)
                idx = np.argsort(s)[::-1]
                return u[:, idx].astype(np.float32), s[idx].astype(np.float32), vt[idx, :].astype(np.float32)
            except Exception:
                raise RuntimeError(f"SVD failed (randomized and svds). Error: {e}")

    def fit(self, train_csr):
        # 1) 计算中心化统计量
        mu, b_u, b_i, _, _ = compute_global_user_item_means(train_csr)
        self.mu = mu
        self.b_u = b_u
        self.b_i = b_i

        m, n = train_csr.shape
        k = min(self.rank, min(m, n) - 1)
        if k <= 0:
            self.lowrank = (None, None, None)
            return self

        # 2) 构造初始填充矩阵X
        rows, cols = train_csr.nonzero()
        vals = train_csr.data.astype(np.float32)
        centered_vals = vals - (mu + b_u[rows] + b_i[cols])

        # 尝试构建稠密矩阵
        try:
            X = np.zeros((m, n), dtype=np.float32)
            X[rows, cols] = centered_vals  # 已观察到的中心化值
        except Exception as e:
            raise MemoryError("构建稠密填充矩阵失败（内存不足）。在Fit前设置randomized=False并使用稀疏svds，或使用分块稀疏实现。") from e

        # 3) 迭代 Soft-Impute
        prev_frob = float('inf')
        for it in range(self.max_iter):
            # 截断的SVD
            U, s, Vt = self._truncated_svd(X, k=k)
            # 奇异值收缩
            s_shrunk = np.maximum(s - self.lambda_reg, 0.0)
            nz = s_shrunk > 0
            if nz.sum() == 0:
                # 所有奇异值被阈值化为0
                X_new = np.zeros_like(X)
            else:
                U_k = U[:, nz]          # m x r'
                s_k = s_shrunk[nz]      # r'
                Vt_k = Vt[nz, :]        # r' x n
                # 低秩重构U_k @ diag(s_k) @ Vt_k
                # 为效率先做V_k = (Vt_k.T * s_k)
                V_k = (Vt_k.T * s_k.reshape(1, -1))  # n x r'
                X_new = (U_k @ V_k.T)               # m x n
            # 填充观察条目：保持观测 entry 的值不变（或按原论文填充未观测）
            # 论文中做法是：对缺失条目用 low-rank 重构值填充，保留观测条目的真实值。
            # 我们这里令 X = reconstructed 对所有条目，对观测条目再用原始观测替换：
            X_filled = X_new
            X_filled[rows, cols] = centered_vals  # 保证观测位置为真实中心化值

            # 收敛检测：使用范数差
            frob_diff = np.linalg.norm(X_filled - X, ord='fro')
            print(f"[SoftImpute] iter {it+1}/{self.max_iter}, frob_diff={frob_diff:.6f}, kept_sv={nz.sum()}")
            X = X_filled
            if frob_diff < self.tol:
                print(f"[SoftImpute] converged at iter {it+1}")
                break

        # 保存最终低秩分解
        # 最后一次 SVD 得到 U_k, s_k, Vt_k （如果上面最后是 X_new 有 nz）
        if 'U_k' in locals() and nz.sum() > 0:
            self.lowrank = (U_k, s_k, Vt_k)
        else:
            self.lowrank = (None, None, None)
        return self

    def predict(self, user_idx, item_idx):
        uu = np.array(user_idx, dtype=np.int64)
        vv = np.array(item_idx, dtype=np.int64)
        n = len(uu)
        preds = np.empty(n, dtype=np.float32)

        if self.lowrank is None or self.lowrank[0] is None:
            preds.fill(self.mu if self.mu is not None else 3.0)
            return np.clip(preds, 1.0, 5.0)

        U_k, s_k, Vt_k = self.lowrank
        # 预计算 V_k = Vt_k.T * s_k  -> 形状(n_items, r)
        V_k = (Vt_k.T * s_k.reshape(1, -1))

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