import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
import time
import warnings

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class PyTorchALS:
    """PyTorch实现的交替最小化（ALS）- GPU加速"""

    def __init__(self, rank=50, lambda_reg=0.01, max_iter=20, tol=1e-4):
        self.rank = rank
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.U = None
        self.V = None
        self.user_biases = None
        self.movie_biases = None
        self.global_mean = None
        self.n_users = None
        self.n_movies = None

    def _initialize_uv(self, n_users, n_movies):
        """初始化因子矩阵"""
        # 使用GPU张量
        self.U = torch.randn(n_users, self.rank, device=device) * 0.01
        self.V = torch.randn(n_movies, self.rank, device=device) * 0.01
        self.user_biases = torch.zeros(n_users, device=device)
        self.movie_biases = torch.zeros(n_movies, device=device)

    def fit(self, train_matrix):
        """训练ALS模型"""
        # 转换为稠密矩阵（GPU上）
        m, n = train_matrix.shape
        self.n_users, self.n_movies = m, n

        # 获取非零元素的索引和值
        rows, cols = train_matrix.nonzero()
        ratings = train_matrix.data

        # 计算全局均值
        self.global_mean = torch.tensor(
            np.mean(ratings) if len(ratings) > 0 else 3.0,
            device=device
        )

        # 转换为PyTorch张量
        rows_tensor = torch.tensor(rows, dtype=torch.long, device=device)
        cols_tensor = torch.tensor(cols, dtype=torch.long, device=device)
        ratings_tensor = torch.tensor(ratings, dtype=torch.float32, device=device)

        # 初始化因子矩阵
        self._initialize_uv(m, n)

        prev_error = float('inf')
        start_time = time.time()

        for iteration in range(self.max_iter):
            # 更新用户因子
            for i in range(m):
                # 获取用户i的评分
                mask = (rows_tensor == i)
                if mask.sum().item() == 0:
                    continue

                movie_indices = cols_tensor[mask]
                actual_ratings = ratings_tensor[mask]

                # 计算残差
                residuals = actual_ratings - self.global_mean - self.movie_biases[movie_indices]

                # 构建线性方程组
                V_sub = self.V[movie_indices]
                A = V_sub.T @ V_sub + self.lambda_reg * torch.eye(self.rank, device=device)
                b = V_sub.T @ residuals

                # 求解
                self.U[i] = torch.linalg.solve(A, b)

                # 更新用户偏置
                if len(residuals) > 0:
                    self.user_biases[i] = torch.mean(residuals - V_sub @ self.U[i])

            # 更新电影因子
            for j in range(n):
                mask = (cols_tensor == j)
                if mask.sum().item() == 0:
                    continue

                user_indices = rows_tensor[mask]
                actual_ratings = ratings_tensor[mask]

                # 计算残差
                residuals = actual_ratings - self.global_mean - self.user_biases[user_indices]

                # 构建线性方程组
                U_sub = self.U[user_indices]
                A = U_sub.T @ U_sub + self.lambda_reg * torch.eye(self.rank, device=device)
                b = U_sub.T @ residuals

                # 求解
                self.V[j] = torch.linalg.solve(A, b)

                # 更新电影偏置
                if len(residuals) > 0:
                    self.movie_biases[j] = torch.mean(residuals - U_sub @ self.V[j])

            # 计算误差
            if iteration % 5 == 0:
                pred_ratings = (self.global_mean +
                                self.user_biases[rows_tensor] +
                                self.movie_biases[cols_tensor] +
                                torch.sum(self.U[rows_tensor] * self.V[cols_tensor], dim=1))

                current_error = torch.mean((ratings_tensor - pred_ratings) ** 2).item()
                elapsed = time.time() - start_time
                print(f"Iteration {iteration + 1}/{self.max_iter}, MSE: {current_error:.4f}, Time: {elapsed:.2f}s")
                start_time = time.time()

                if abs(prev_error - current_error) < self.tol:
                    print(f"PyTorch ALS收敛于迭代{iteration + 1}")
                    break

                prev_error = current_error

        return self

    def predict(self, user_idx, movie_idx):
        """预测评分"""
        with torch.no_grad():
            # 确保索引是整数
            if isinstance(user_idx, (list, np.ndarray)):
                user_tensor = torch.tensor(user_idx, dtype=torch.long, device=device)
                movie_tensor = torch.tensor(movie_idx, dtype=torch.long, device=device)
            else:
                user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
                movie_tensor = torch.tensor([movie_idx], dtype=torch.long, device=device)

            prediction = (self.global_mean +
                          self.user_biases[user_tensor] +
                          self.movie_biases[movie_tensor] +
                          torch.sum(self.U[user_tensor] * self.V[movie_tensor], dim=1))

            # 限制在1-5范围内
            prediction = torch.clamp(prediction, 1.0, 5.0)

            return prediction.cpu().numpy().item() if prediction.numel() == 1 else prediction.cpu().numpy()


class PyTorchSoftImpute:
    """PyTorch实现的Soft-Impute - GPU加速"""

    def __init__(self, lambda_reg=0.1, max_iter=50, tol=1e-4, rank=100):
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.rank = rank  # 用于随机SVD的秩
        self.Z = None
        self.global_mean = None

    def _randomized_svd(self, X, k):
        """随机SVD实现 - 更高效"""
        m, n = X.shape

        # 随机投影
        Omega = torch.randn(n, k, device=device)
        Y = X @ Omega

        # QR分解
        Q, _ = torch.linalg.qr(Y)

        # 投影到低维空间
        B = Q.T @ X

        # 计算SVD
        U_b, S, Vt = torch.linalg.svd(B, full_matrices=False)
        U = Q @ U_b

        return U[:, :k], S[:k], Vt[:k, :]

    def fit(self, train_matrix):
        """训练Soft-Impute模型"""
        m, n = train_matrix.shape

        # 获取非零元素
        rows, cols = train_matrix.nonzero()
        ratings = train_matrix.data

        # 计算全局均值
        self.global_mean = torch.tensor(
            np.mean(ratings) if len(ratings) > 0 else 3.0,
            device=device
        )

        # 中心化数据
        centered_ratings = torch.tensor(ratings - self.global_mean.cpu().numpy(),
                                        dtype=torch.float32, device=device)

        # 创建稀疏张量（使用稠密矩阵，因为GPU上稀疏操作有限）
        Z = torch.zeros(m, n, device=device)

        # 填充观测值
        rows_tensor = torch.tensor(rows, dtype=torch.long, device=device)
        cols_tensor = torch.tensor(cols, dtype=torch.long, device=device)
        Z[rows_tensor, cols_tensor] = centered_ratings

        Z_old = Z.clone()

        start_time = time.time()

        for iteration in range(self.max_iter):
            # 对当前矩阵进行随机SVD
            k = min(self.rank, m, n)
            U, s, Vt = self._randomized_svd(Z, k)

            # 软阈值收缩
            s_shrunk = torch.clamp(s - self.lambda_reg, min=0)

            # 重建矩阵
            Z = U @ torch.diag(s_shrunk) @ Vt

            # 填充观测值（保持观测值不变）
            Z[rows_tensor, cols_tensor] = centered_ratings

            # 检查收敛
            diff = torch.norm(Z - Z_old, 'fro').item()
            if diff < self.tol:
                print(f"PyTorch Soft-Impute收敛于迭代{iteration + 1}")
                break

            Z_old = Z.clone()

            if iteration % 5 == 0:
                # 计算训练误差
                preds = Z[rows_tensor, cols_tensor] + self.global_mean
                actual = torch.tensor(ratings, dtype=torch.float32, device=device)
                mse = torch.mean((actual - preds) ** 2).item()
                elapsed = time.time() - start_time
                print(f"Iteration {iteration + 1}/{self.max_iter}, MSE: {mse:.4f}, Time: {elapsed:.2f}s")
                start_time = time.time()

        # 保存结果矩阵
        self.Z = Z + self.global_mean
        return self

    def predict(self, user_idx, movie_idx):
        """预测评分"""
        with torch.no_grad():
            # 确保索引是整数
            if isinstance(user_idx, (list, np.ndarray)):
                user_tensor = torch.tensor(user_idx, dtype=torch.long, device=device)
                movie_tensor = torch.tensor(movie_idx, dtype=torch.long, device=device)
            else:
                user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
                movie_tensor = torch.tensor([movie_idx], dtype=torch.long, device=device)

            prediction = self.Z[user_tensor, movie_tensor]
            prediction = torch.clamp(prediction, 1.0, 5.0)

            return prediction.cpu().numpy().item() if prediction.numel() == 1 else prediction.cpu().numpy()


class MatrixCompletionEvaluator:
    """矩阵填充模型评估器"""

    def __init__(self, n_folds=5, use_gpu=True):
        self.n_folds = n_folds
        self.use_gpu = use_gpu

    def evaluate(self, model_class, model_params, model_name=""):
        """评估模型并返回各折RMSE和平均RMSE"""
        rmse_list = []
        fold_times = []

        for fold in range(1, self.n_folds + 1):
            print(f"\n{'=' * 60}")
            print(f"处理第{fold}折 - {model_name}")
            print(f"{'=' * 60}")

            try:
                # 加载数据
                train_matrix = load_npz(f"train_matrix_fold{fold}.npz")
                test_df = pd.read_csv(f"test_set_fold{fold}.csv")

                # 检查数据
                print(f"训练集: {train_matrix.shape[0]}用户, {train_matrix.shape[1]}电影")
                print(f"非零元素: {train_matrix.nnz}")
                print(f"测试集大小: {len(test_df)}")

                # 训练模型
                print("开始训练...")
                fold_start_time = time.time()
                model = model_class(**model_params)
                model.fit(train_matrix)
                train_time = time.time() - fold_start_time

                print(f"训练完成，耗时: {train_time:.2f}秒")

                # 批量预测（更高效）
                print("开始预测...")
                test_users = test_df['UserIdx'].values
                test_movies = test_df['MovieIdx'].values

                # 批量处理预测
                batch_size = 10000
                predictions = []

                for i in range(0, len(test_users), batch_size):
                    end_idx = min(i + batch_size, len(test_users))
                    batch_users = test_users[i:end_idx]
                    batch_movies = test_movies[i:end_idx]

                    batch_preds = model.predict(batch_users, batch_movies)
                    predictions.extend(batch_preds.flatten())

                test_df['Predicted'] = predictions

                # 计算RMSE
                rmse = np.sqrt(np.mean((test_df['Rating'] - test_df['Predicted']) ** 2))
                rmse_list.append(rmse)
                fold_times.append(train_time)

                print(f"第{fold}折RMSE: {rmse:.4f}")
                print(f"预测值范围: [{test_df['Predicted'].min():.2f}, {test_df['Predicted'].max():.2f}]")
                print(f"预测值均值: {test_df['Predicted'].mean():.2f}")
                print(f"预测值标准差: {test_df['Predicted'].std():.2f}")

                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"处理第{fold}折时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                rmse_list.append(float('inf'))
                fold_times.append(float('inf'))

        # 计算平均结果
        valid_rmse = [r for r in rmse_list if r != float('inf')]
        avg_rmse = np.mean(valid_rmse) if valid_rmse else float('inf')

        valid_times = [t for t in fold_times if t != float('inf')]
        avg_time = np.mean(valid_times) if valid_times else float('inf')

        return rmse_list, avg_rmse, avg_time


def run_als_experiment():
    """运行ALS实验"""
    print("\n" + "=" * 70)
    print("PyTorch ALS 矩阵填充实验 (GPU加速)")
    print("=" * 70)

    evaluator = MatrixCompletionEvaluator(n_folds=5)

    # 测试不同的ALS配置
    als_configs = [
        {
            "name": "ALS (rank=10)",
            "params": {"rank": 10, "lambda_reg": 0.1, "max_iter": 15, "tol": 1e-3}
        },
        {
            "name": "ALS (rank=20)",
            "params": {"rank": 20, "lambda_reg": 0.05, "max_iter": 20, "tol": 1e-4}
        },
        {
            "name": "ALS (rank=50)",
            "params": {"rank": 50, "lambda_reg": 0.01, "max_iter": 25, "tol": 1e-4}
        }
    ]

    results = []

    for config in als_configs:
        print(f"\n{'-' * 60}")
        print(f"测试配置: {config['name']}")
        print(f"参数: {config['params']}")
        print(f"{'-' * 60}")

        rmse_list, avg_rmse, avg_time = evaluator.evaluate(
            PyTorchALS, config['params'], config['name']
        )

        results.append({
            'Model': config['name'],
            'Avg RMSE': avg_rmse,
            'Avg Time (s)': avg_time,
            'Fold 1 RMSE': rmse_list[0] if len(rmse_list) > 0 else None,
            'Fold 2 RMSE': rmse_list[1] if len(rmse_list) > 1 else None,
            'Fold 3 RMSE': rmse_list[2] if len(rmse_list) > 2 else None,
            'Fold 4 RMSE': rmse_list[3] if len(rmse_list) > 3 else None,
            'Fold 5 RMSE': rmse_list[4] if len(rmse_list) > 4 else None,
        })

        print(f"\n{config['name']} 5折平均RMSE: {avg_rmse:.4f}")
        print(f"各折RMSE: {[round(r, 4) for r in rmse_list]}")
        print(f"平均训练时间: {avg_time:.2f}秒")

    return results


def run_softimpute_experiment():
    """运行Soft-Impute实验"""
    print("\n" + "=" * 70)
    print("PyTorch Soft-Impute 矩阵填充实验 (GPU加速)")
    print("=" * 70)

    evaluator = MatrixCompletionEvaluator(n_folds=5)

    # 测试不同的Soft-Impute配置
    softimpute_configs = [
        {
            "name": "Soft-Impute (λ=0.5)",
            "params": {"lambda_reg": 0.5, "max_iter": 30, "tol": 1e-4, "rank": 100}
        },
        {
            "name": "Soft-Impute (λ=0.1)",
            "params": {"lambda_reg": 0.1, "max_iter": 40, "tol": 1e-4, "rank": 100}
        },
        {
            "name": "Soft-Impute (λ=0.01)",
            "params": {"lambda_reg": 0.01, "max_iter": 50, "tol": 1e-4, "rank": 100}
        }
    ]

    results = []

    for config in softimpute_configs:
        print(f"\n{'-' * 60}")
        print(f"测试配置: {config['name']}")
        print(f"参数: {config['params']}")
        print(f"{'-' * 60}")

        rmse_list, avg_rmse, avg_time = evaluator.evaluate(
            PyTorchSoftImpute, config['params'], config['name']
        )

        results.append({
            'Model': config['name'],
            'Avg RMSE': avg_rmse,
            'Avg Time (s)': avg_time,
            'Fold 1 RMSE': rmse_list[0] if len(rmse_list) > 0 else None,
            'Fold 2 RMSE': rmse_list[1] if len(rmse_list) > 1 else None,
            'Fold 3 RMSE': rmse_list[2] if len(rmse_list) > 2 else None,
            'Fold 4 RMSE': rmse_list[3] if len(rmse_list) > 3 else None,
            'Fold 5 RMSE': rmse_list[4] if len(rmse_list) > 4 else None,
        })

        print(f"\n{config['name']} 5折平均RMSE: {avg_rmse:.4f}")
        print(f"各折RMSE: {[round(r, 4) for r in rmse_list]}")
        print(f"平均训练时间: {avg_time:.2f}秒")

    return results


def main():
    """主函数"""
    print("=" * 80)
    print("MovieLens矩阵填充实验 - PyTorch GPU版本")
    print("=" * 80)
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # 运行ALS实验
    als_results = run_als_experiment()

    # 运行Soft-Impute实验
    softimpute_results = run_softimpute_experiment()

    # 合并结果
    all_results = als_results + softimpute_results

    # 创建结果DataFrame
    results_df = pd.DataFrame(all_results)

    # 结果汇总
    print("\n" + "=" * 80)
    print("最终结果汇总")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # 保存结果到CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"matrix_completion_results_pytorch_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"\n结果已保存到: {filename}")

    # 找出最佳模型
    best_idx = results_df['Avg RMSE'].idxmin()
    best_model = results_df.loc[best_idx]
    print(f"\n最佳模型: {best_model['Model']}")
    print(f"最佳平均RMSE: {best_model['Avg RMSE']:.4f}")
    print(f"平均训练时间: {best_model['Avg Time (s)']:.2f}秒")

    return results_df


if __name__ == "__main__":
    # 设置PyTorch随机种子以保证可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    try:
        results = main()
    except KeyboardInterrupt:
        print("\n实验被用户中断")
    except Exception as e:
        print(f"\n实验过程中发生错误: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()