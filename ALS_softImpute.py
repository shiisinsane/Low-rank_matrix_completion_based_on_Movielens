# final_integrated_matrix_completion.py
# MovieLens 10/20M — Chunk-safe, center + ALS-WR + Safe Soft-Impute fallback
# Usage: replace your ALS_softImpute.py with this file and run.

import time
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz, csr_matrix
from scipy.sparse.linalg import svds

warnings.filterwarnings('ignore')

# Device and global chunk config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Chunk size for operations over all observed ratings or large test sets.
# Default tuned for ~16GB GPU; reduce if you have less memory.
CHUNK_SIZE = int(2_000_000)


# ------------------------
# Utilities: center / stats
# ------------------------
def compute_global_user_item_means(train_csr):
    """
    Compute global mean, user mean (bias), item mean (bias).
    We'll compute:
      mu = mean of all observed ratings
      b_u = mean_{i in I(u)} (r_ui) - mu
      b_i = mean_{u in U(i)} (r_ui) - mu
    We'll return these so we can center the matrix: r_center = r - mu - b_u[u] - b_i[i]
    (This is a one-pass, commonly used centering; iterative refinement is possible but not necessary.)
    """
    rows, cols = train_csr.nonzero()
    ratings = train_csr.data.astype(np.float32)
    mu = float(np.mean(ratings))

    # user means (relative to mu)
    n_users = train_csr.shape[0]
    b_u = np.zeros(n_users, dtype=np.float32)
    counts_u = np.zeros(n_users, dtype=np.int32)
    for r, rating in zip(rows, ratings):
        b_u[r] += rating
        counts_u[r] += 1
    nonzero_u = counts_u > 0
    b_u[nonzero_u] = b_u[nonzero_u] / counts_u[nonzero_u] - mu

    # item means (relative to mu)
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
# ALS (centered + chunk-safe + ALS-WR style)
# ------------------------
class PyTorchALS:
    """
    ALS for centered residuals with chunked operations to avoid OOM.
    We expect caller to pass the raw train sparse matrix; this class will compute center stats itself.
    """

    def __init__(self, rank=250, lambda_reg=0.02, max_iter=15, tol=1e-4, chunk_size=CHUNK_SIZE):
        self.rank = int(rank)
        self.lambda_reg = float(lambda_reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.chunk_size = int(chunk_size)

        # learned factors for centered matrix
        self.U = None  # torch tensor [n_users, rank]
        self.V = None  # torch tensor [n_items, rank]
        # stored centering stats
        self.global_mean = None
        self.user_bias = None  # numpy
        self.item_bias = None  # numpy

    def _initialize(self, n_users, n_items):
        self.U = torch.randn(n_users, self.rank, dtype=torch.float32, device=device) * 0.01
        self.V = torch.randn(n_items, self.rank, dtype=torch.float32, device=device) * 0.01

    def _chunked_preds(self, rows_t, cols_t):
        """Compute predictions for given index arrays (torch tensors) in chunks."""
        n = len(rows_t)
        preds = torch.empty(n, dtype=torch.float32, device=device)
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            ru = rows_t[start:end]
            rv = cols_t[start:end]
            preds[start:end] = torch.sum(self.U[ru] * self.V[rv], dim=1)
        return preds

    def fit(self, train_csr):
        # compute centering stats
        mu, b_u, b_i, counts_u, counts_i = compute_global_user_item_means(train_csr)
        self.global_mean = float(mu)
        self.user_bias = b_u  # np array
        self.item_bias = b_i  # np array
        n_users, n_items = train_csr.shape
        self._initialize(n_users, n_items)

        # prepare observed arrays
        rows, cols = train_csr.nonzero()
        ratings = train_csr.data.astype(np.float32)
        # centered ratings (r - mu - b_u - b_i)
        centered = ratings - (mu + b_u[rows] + b_i[cols])

        # tensors
        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        cols_t = torch.tensor(cols, dtype=torch.long, device=device)
        centered_t = torch.tensor(centered, dtype=torch.float32, device=device)

        prev_mse = float('inf')
        # iterative ALS
        for it in range(self.max_iter):
            # update users
            unique_users = torch.unique(rows_t)
            for user in unique_users:
                mask = (rows_t == user)
                item_idx = cols_t[mask]
                y = centered_t[mask]  # centered observed vector for user

                V_sub = self.V[item_idx]  # k x r
                # normal eq: (V_sub^T V_sub + λ * |I(u)| I) u = V_sub^T y
                VTV = V_sub.T @ V_sub
                reg_scale = float(max(1, int((mask.sum().item()))))  # scale reg by number of observations
                reg = self.lambda_reg * reg_scale * torch.eye(self.rank, device=device, dtype=torch.float32)
                b = V_sub.T @ y
                try:
                    self.U[user] = torch.linalg.solve(VTV + reg, b)
                except Exception:
                    self.U[user] = torch.linalg.pinv(VTV + reg) @ b

            # update items
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

            # compute mse on centered data
            preds = self._chunked_preds(rows_t, cols_t)
            mse = torch.mean((centered_t - preds) ** 2).item()
            print(f"[ALS] iter {it+1}/{self.max_iter}, centered MSE={mse:.6f}")
            if abs(prev_mse - mse) < self.tol:
                print(f"[ALS] converged at iter {it+1}")
                break
            prev_mse = mse

        return self

    def predict(self, user_idx, item_idx):
        # accepts numpy arrays / lists
        uu = np.array(user_idx, dtype=np.int64)
        vv = np.array(item_idx, dtype=np.int64)
        n = len(uu)
        preds = np.empty(n, dtype=np.float32)

        # batch in chunks
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            uu_chunk = torch.tensor(uu[start:end], dtype=torch.long, device=device)
            vv_chunk = torch.tensor(vv[start:end], dtype=torch.long, device=device)
            pred_chunk = torch.sum(self.U[uu_chunk] * self.V[vv_chunk], dim=1).cpu().numpy()
            preds[start:end] = pred_chunk

        # add back centers: mu + b_u + b_i
        preds = preds + self.global_mean + self.user_bias[uu] + self.item_bias[vv]
        preds = np.clip(preds, 1.0, 5.0)
        return preds


# ------------------------
# Soft-Impute fallback (safe low-rank svds + chunk predict)
# ------------------------
class PyTorchSoftImpute:
    """
       For large sparse matrix we do a one-shot svds and soft-threshold s:
        - compute svds (U,s,Vt)
        - shrink s := max(s - lambda, 0)
        - store (U_k, s_k, Vt_k) for prediction
       Prediction uses chunked row-wise dot products: pred_ij = (U_k[i,:] * s_k) dot Vt_k[:, j]
    """

    def __init__(self, lambda_reg=1e-4, rank=250, chunk_size=CHUNK_SIZE):
        self.lambda_reg = float(lambda_reg)
        self.rank = int(rank)
        self.chunk_size = int(chunk_size)
        self.lowrank = None  # (U_k, s_k, Vt_k) - numpy on CPU
        # centering stats
        self.mu = None
        self.b_u = None
        self.b_i = None

    def _svds(self, train_csr, k):
        # use scipy.sparse.linalg.svds (may be slow) and return sorted descending
        u, s, vt = svds(train_csr.astype(np.float64), k=k)
        idx = np.argsort(s)[::-1]
        return u[:, idx].astype(np.float32), s[idx].astype(np.float32), vt[idx, :].astype(np.float32)

    def fit(self, train_csr):
        # 1) compute centering stats (mu, b_u, b_i)
        mu, b_u, b_i, _, _ = compute_global_user_item_means(train_csr)
        self.mu = mu
        self.b_u = b_u
        self.b_i = b_i

        # 2) build centered sparse matrix (on CPU) for svds input: r_center = r - mu - b_u[u] - b_i[v]
        rows, cols = train_csr.nonzero()
        vals = train_csr.data.astype(np.float32)
        centered_vals = vals - (mu + b_u[rows] + b_i[cols])
        # build csr with centered values
        A_center = csr_matrix((centered_vals, (rows, cols)), shape=train_csr.shape)

        # 3) compute truncated svds
        k = min(self.rank, min(train_csr.shape) - 1)
        if k <= 0:
            self.lowrank = (None, None, None)
            return self
        U, s, Vt = self._svds(A_center, k=k)

        # 4) soft-threshold s
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
        # precompute V_k = Vt_k.T * s_k  -> shape (n_items, r)
        V_k = (Vt_k.T * s_k.reshape(1, -1))

        # chunked dot product
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            u_chunk = uu[start:end]
            v_chunk = vv[start:end]
            U_sub = U_k[u_chunk, :]    # chunk x r
            V_sub = V_k[v_chunk, :]    # chunk x r
            preds[start:end] = np.sum(U_sub * V_sub, axis=1)

        # add back centers
        if self.mu is not None:
            preds = preds + self.mu + self.b_u[uu] + self.b_i[vv]

        return np.clip(preds, 1.0, 5.0)

# ------------------------
# Evaluator
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
# Experiment runner
# ------------------------
def run_experiments():
    evaluator = MatrixCompletionEvaluator(n_folds=5, chunk_size=CHUNK_SIZE)

    results = []

    # Soft-Impute fallback configs
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


    # ALS configs
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