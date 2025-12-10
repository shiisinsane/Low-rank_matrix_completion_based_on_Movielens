# MovieLens 10M
# Chunk, 中心化, ALS-WR, Soft-Impute
# 修改版：增加迭代历史记录(iter_history)、GPU同步与保存每折历史文件

import time
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz, csr_matrix
from scipy.sparse.linalg import svds
import scipy.linalg as sla
from sklearn.utils.extmath import randomized_svd
import os

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
        # 训练历史（每次迭代保存 elapsed_seconds, observed_centered_MSE）
        self.iter_history = []

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

        # 4) 初始化 U, V（优先使用截断SVD初始化，失败则回退到随机初始化）
        rng = np.random.RandomState(42)

        # 我们只用较小的rank做初始化（例如 min(rank, 100)）
        r_init = min(self.rank, 100)

        try:
            print("Performing truncated SVD initialization ...")

            # 构造中心化后的稀疏矩阵（仅对观测位置赋值）
            # 注意：不是构建稠密矩阵，而是直接对稀疏CSR使用svds（更稳）
            from scipy.sparse.linalg import svds

            # 计算rank r_init 的截断 SVD
            U0, s0, Vt0 = svds(train_csr.astype(np.float64), k=r_init)

            # svds 返回的奇异值是从小到大排序的，需要翻转
            s0 = s0[::-1].astype(np.float32)  # (r_init,)
            U0 = U0[:, ::-1].astype(np.float32)  # (n_users, r_init)
            Vt0 = Vt0[::-1, :].astype(np.float32)  # (r_init, n_items)

            # 按 ALS 的标准方式初始化：U = U0 * sqrt(s)，V = V0 * sqrt(s)
            sqrt_s = np.sqrt(s0)
            U_cpu = np.zeros((n_users, self.rank), dtype=np.float32)
            V_cpu = np.zeros((n_items, self.rank), dtype=np.float32)

            # 前 r_init 维用SVD初始化
            U_cpu[:, :r_init] = U0 * sqrt_s.reshape(1, -1)
            V_cpu[:, :r_init] = (Vt0.T) * sqrt_s.reshape(1, -1)

            # 剩下维度仍然用随机初始化（保持与原逻辑一致）
            if self.rank > r_init:
                U_cpu[:, r_init:] = (rng.randn(n_users, self.rank - r_init).astype(np.float32) * 0.01)
                V_cpu[:, r_init:] = (rng.randn(n_items, self.rank - r_init).astype(np.float32) * 0.01)

            print(f"SVD initialization successful (r_init={r_init}).")

        except Exception as e:
            print("SVD initialization failed, falling back to random init:", e)
            U_cpu = (rng.randn(n_users, self.rank).astype(np.float32) * 0.01)
            V_cpu = (rng.randn(n_items, self.rank).astype(np.float32) * 0.01)


        # 记录迭代历史
        self.iter_history = []
        fit_start_time = time.time()
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

            elapsed = time.time() - fit_start_time
            # 记录迭代历史：elapsed_seconds, observed_centered_MSE
            self.iter_history.append((elapsed, mse))
            print(f"[ALS] iter {it+1}/{self.max_iter}, centered MSE={mse:.6f}, elapsed={elapsed:.2f}s")


            if abs(prev_mse - mse) < self.tol:
                print(f"[ALS] converged at iter {it+1}")
                break
            prev_mse = mse

        # 训练结束，把U_cpu/V_cpu同步回torch用于predict
        self.U = torch.tensor(U_cpu, dtype=torch.float32, device=device)
        self.V = torch.tensor(V_cpu, dtype=torch.float32, device=device)

        # 记录最终metric与总耗时（在记录前同步GPU以保证计时准确）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.final_metric = mse
        self.total_fit_time = time.time() - fit_start_time

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

# --------------------------------------
# Soft-Impute (中心化 + 截断SVD + 软阈值)
# --------------------------------------
class PyTorchSoftImpute:
    """
    Soft-Impute 对中心化矩阵进行阶数k的截断SVD，并对奇异值做 soft-threshold (s - lambda)+，
    再用该低秩矩阵在未观察位置填充，于观察位置保持原值。
    增加改动：
      - 记录 self.iter_history：[(elapsed_seconds, observed_centered_MSE), ...]
      - 训练结束后记录 self.final_metric, self.total_fit_time
      - 每次迭代计算“已观察位置上的中心化MSE”（与ALS统一度量）
      - 在使用GPU时训练结束前同步 GPU，保证计时公平
    """
    def __init__(self, rank=250, lambda_reg=1e-4, max_iter=15, tol=1e-4,
                 randomized=True, n_iter_svd=7, chunk_size=CHUNK_SIZE):
        self.rank = int(rank)
        self.lambda_reg = float(lambda_reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.randomized = bool(randomized)
        self.n_iter_svd = int(n_iter_svd)
        self.chunk_size = int(chunk_size)

        # 存储中心化统计量
        self.mu = None
        self.b_u = None
        self.b_i = None

        # low-rank factor
        self.lowrank = (None, None, None)
        # 训练过程历史
        self.iter_history = []

    def _truncated_svd(self, X, k):
        """
        进行k阶截断SVD。若randomized=True则用sklearn's randomized_svd，
        否则使用scipy.sparse.linalg.svds。
        """
        if self.randomized:
            U, s, Vt = randomized_svd(
                X, n_components=k, n_iter=self.n_iter_svd, random_state=42
            )
            return U.astype(np.float32), s.astype(np.float32), Vt.astype(np.float32)
        else:
            # svds要求至少k+1 <= min(m,n)，且返回是s从小到大排序，需要颠倒
            X64 = X.astype(np.float64, copy=False)
            U64, s64, Vt64 = svds(X64, k=k)
            s_order = s64[::-1]
            U_order = U64[:, ::-1]
            Vt_order = Vt64[::-1, :]
            return (U_order.astype(np.float32),
                    s_order.astype(np.float32),
                    Vt_order.astype(np.float32))

    def fit(self, train_csr):
        """
        对训练稀疏矩阵进行中心化后，执行Soft-Impute。
        修复内容：
          - 修复“覆盖观测值导致 obs_mse 恒为0”的问题（核心问题）
          - 把观测位置MSE放在覆盖前计算（正确做法）
          - 增加奇异值 shrink 前后的 debug 打印
          - 记录 (elapsed, obs_mse_recon) 到 iter_history
          - 保持整体结构与原代码一致，不做重构
        """
        # ----- 1) 中心化统计 -----
        mu, b_u, b_i, _, _ = compute_global_user_item_means(train_csr)
        self.mu = float(mu)
        self.b_u = b_u
        self.b_i = b_i

        m, n = train_csr.shape
        k = min(self.rank, min(m, n) - 1)
        if k <= 0:
            self.lowrank = (None, None, None)
            self.iter_history = []
            self.final_metric = float('inf')
            self.total_fit_time = 0.0
            return self

        # 稀疏观测
        rows, cols = train_csr.nonzero()
        vals = train_csr.data.astype(np.float32)
        centered_vals = vals - (self.mu + b_u[rows] + b_i[cols])

        # ----- 2) 构造稠密填充矩阵 X -----
        try:
            X = np.zeros((m, n), dtype=np.float32)
            X[rows, cols] = centered_vals
        except Exception as e:
            raise MemoryError(
                "构建稠密矩阵 X 失败（内存不足）。需要改为稀疏+分块SVD实现。"
            ) from e

        # ----- 3) Soft-Impute 迭代 -----
        prev_frob = float('inf')
        self.iter_history = []
        fit_start_time = time.time()

        for it in range(self.max_iter):
            # === SVD ===
            U, s, Vt = self._truncated_svd(X, k=k)

            # --- Debug：打印前几项奇异值 ---
            topk = min(8, len(s))
            print(f"  raw singulars (top {topk}): {s[:topk]}")

            # === Soft-threshold ===
            s_shrunk = np.maximum(s - self.lambda_reg, 0.0)
            print(f"  shrunk singulars (top {topk}): {s_shrunk[:topk]} (lambda={self.lambda_reg})")

            nz = s_shrunk > 0

            # === 重构 X_recon ===
            if nz.sum() == 0:
                # 所有奇异值被削为0
                X_recon = np.zeros_like(X)
            else:
                U_k = U[:, nz]  # (m, r')
                s_k = s_shrunk[nz]  # (r',)
                Vt_k = Vt[nz, :]  # (r', n)

                # 计算 X_recon = U_k * diag(s_k) * Vt_k
                # 为提高效率，构造 V_k = Vt_k.T * s_k
                V_k = (Vt_k.T * s_k.reshape(1, -1))  # shape (n, r')
                X_recon = U_k @ V_k.T  # (m, n)

            # === 计算观测位置上的重构误差（覆盖前）===
            preds_obs_recon = X_recon[rows, cols]
            obs_mse_recon = float(np.mean((centered_vals - preds_obs_recon) ** 2))

            # === 覆盖观测值（Soft-Impute 必须）===
            X_new = X_recon.copy()
            X_new[rows, cols] = centered_vals

            # === 收敛判定 ===
            frob_diff = np.linalg.norm(X_new - X, ord='fro')

            elapsed = time.time() - fit_start_time
            self.iter_history.append((elapsed, obs_mse_recon))

            print(
                f"[SoftImpute] iter {it + 1}/{self.max_iter}, "
                f"frob_diff={frob_diff:.6f}, "
                f"observed_centered_MSE_recon={obs_mse_recon:.6f}, "
                f"kept_sv={nz.sum()}, "
                f"elapsed={elapsed:.2f}s"
            )

            # 更新 X
            X = X_new

            if frob_diff < self.tol:
                print(f"[SoftImpute] Converged at iter {it + 1}")
                break

        # ======= 保存最终低秩因子 =======
        if nz.sum() > 0:
            self.lowrank = (U_k, s_k, Vt_k)
        else:
            self.lowrank = (None, None, None)

        # GPU 同步（如有）
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.final_metric = obs_mse_recon
        self.total_fit_time = time.time() - fit_start_time

        return self


    def predict(self, user_idx, item_idx):
        """
        使用保留的 (U_k, s_k, Vt_k) 对新点做预测，并恢复中心化与偏置。
        """
        uu = np.array(user_idx, dtype=np.int64)
        vv = np.array(item_idx, dtype=np.int64)
        n = len(uu)

        U_k, s_k, Vt_k = self.lowrank
        if U_k is None or s_k is None:
            # 全0 => 预测=中心化+偏置=mu+bias
            preds = self.mu + self.b_u[uu] + self.b_i[vv]
            return np.clip(preds, 1.0, 5.0)

        # 分块
        preds_center = np.empty(n, dtype=np.float32)
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            up = uu[start:end]
            vp = vv[start:end]
            # (u,:) dot diag(s_k) dot (v,:)^T
            # = sum_j ( U_k[u,j] * s_k[j] * V_k[v,j] ),
            #   若V_k = Vt_k^T * diag(s_k)
            U_part = U_k[up, :]          # (p, r')
            V_part = Vt_k[:, vp].T       # (p, r')
            preds_center[start:end] = np.sum(U_part * (s_k * V_part), axis=1)

        preds = preds_center + self.mu + self.b_u[uu] + self.b_i[vv]
        return np.clip(preds, 1.0, 5.0)

# ------------------------
# 评估预测
# ------------------------
class MatrixCompletionEvaluator:
    def __init__(self, n_folds=5, chunk_size=CHUNK_SIZE, history_dir="history"):
        self.n_folds = int(n_folds)
        self.chunk_size = int(chunk_size)
        self.history_dir = history_dir
        os.makedirs(self.history_dir, exist_ok=True)

    def _save_history(self, model_name, fold, iter_history, final_metric, total_fit_time):
        """
        保存历史为 npz：包含 iter_history (N x 2), final_metric, total_fit_time
        """
        safe_name = model_name.replace(" ", "_").replace("/", "_")
        path = os.path.join(self.history_dir, f"{safe_name}_fold{fold}_history.npz")
        # iter_history -> np.array shape (N,2)
        if iter_history:
            arr = np.array(iter_history, dtype=np.float64)
        else:
            arr = np.zeros((0, 2), dtype=np.float64)
        np.savez_compressed(path, iter_history=arr, final_metric=float(final_metric), total_fit_time=float(total_fit_time))
        return path

    def evaluate(self, model_class, model_params, model_name=""):
        """
        原 evaluate 基本逻辑保持不变，但在计时前后进行 GPU 同步并保存模型训练历史。
        返回：rmse_list, avg_rmse, avg_time
        并且会在 history_dir 中保存每折对应的历史 npz 文件，文件名形如：
            history/{model_name}_fold{fold}_history.npz
        """
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

                # 建模与计时：在计时前后进行 GPU 同步，确保计时公平
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                model = model_class(**model_params)
                model.fit(train_matrix)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t = time.time() - start

                preds = model.predict(test_df['UserIdx'].values, test_df['MovieIdx'].values)
                test_df['Predicted'] = preds
                rmse = float(np.sqrt(np.mean((test_df['Rating'].values - preds) ** 2)))
                print(f"Fold {fold} RMSE: {rmse:.6f}")

                rmse_list.append(rmse)
                times.append(t)

                # 保存训练历史（如果模型提供 iter_history / final_metric / total_fit_time）
                iter_hist = getattr(model, 'iter_history', [])
                final_metric = getattr(model, 'final_metric', np.inf)
                total_fit_time = getattr(model, 'total_fit_time', t)
                saved_path = self._save_history(model_name, fold, iter_hist, final_metric, total_fit_time)
                print(f"Saved history to {saved_path}")

                # 清理显存
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

    @staticmethod
    def _time_to_target_from_history(iter_history, target_metric):
        """
        iter_history: list/ndarray of shape (N,2) or list of tuples [(elapsed, metric), ...]
        target_metric: 希望达到的 metric（越小越好）
        返回：第一次 elapsed where metric <= target_metric；若未达到返回 np.inf
        """
        if iter_history is None:
            return np.inf
        arr = np.array(iter_history, dtype=np.float64)
        if arr.size == 0:
            return np.inf
        # arr[:,1] 是 metric
        idxs = np.where(arr[:, 1] <= float(target_metric))[0]
        if idxs.size == 0:
            return np.inf
        return float(arr[idxs[0], 0])


# ------------------------
# 运行两种实验算法
# ------------------------
def run_experiments():
    evaluator = MatrixCompletionEvaluator(n_folds=5, chunk_size=CHUNK_SIZE, history_dir="history")

    results = []

    # Soft-Impute设置
    soft_configs = [
        {"name": "SoftImpute(svds_lambda=80_r=150)", "params": {"lambda_reg": 80, "rank": 150, "chunk_size": CHUNK_SIZE}},
        {"name": "SoftImpute(svds_lambda=100_r=200)", "params": {"lambda_reg": 100, "rank": 200, "chunk_size": CHUNK_SIZE}},
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
        {"name": "ALS(rank=150)", "params": {"rank": 150, "lambda_reg": 0.02, "max_iter": 20, "tol": 1e-4, "chunk_size": CHUNK_SIZE}},
        {"name": "ALS(rank=200)", "params": {"rank": 200, "lambda_reg": 0.02, "max_iter": 20, "tol": 1e-4, "chunk_size": CHUNK_SIZE}},
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