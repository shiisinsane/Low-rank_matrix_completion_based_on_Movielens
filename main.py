import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz  # 存储稀疏矩阵

"""
 基于完整ratings.dat构建全局ID映射
 目的：将原始UserID/MovieID转为连续索引，避免矩阵空行/空列
"""

ratings_full = pd.read_csv(
    'ratings.dat',
    sep='::',
    engine='python',
    names=['UserID', 'MovieID', 'Rating', 'Timestamp']
)

# 提取所有唯一的UserID/MovieID，构建映射字典
user_ids = ratings_full['UserID'].unique()
movie_ids = ratings_full['MovieID'].unique()
user_map = {uid: i for i, uid in enumerate(user_ids)}  # 原始UserID转换成连续UserIdx
movie_map = {mid: j for j, mid in enumerate(movie_ids)}  # 原始MovieID转换成连续MovieIdx

n_users = len(user_ids)  # 全局用户总数
n_movies = len(movie_ids)  # 全局电影总数


"""
读取官方train/test文件并转换为连续索引
"""
def read_official_file(file_path):
    """
    读取划分好的rX.train/rX.test文件，转换为带连续索引（UserIdx/MovieIdx）的DataFrame
    """
    df = pd.read_csv(
        file_path,
        sep='::',
        engine='python',
        names=['UserID', 'MovieID', 'Rating', 'Timestamp']
    )
    # 将原始ID转为连续索引
    df['UserIdx'] = df['UserID'].map(user_map)
    df['MovieIdx'] = df['MovieID'].map(movie_map)
    return df


"""
遍历每个fold（r1~r5），构建训练矩阵+保存测试集
"""

for fold in range(1, 6):  # 5个fold
    print(f"\n===== 处理第{fold}折数据 =====")

    # 读取当前fold的训练/测试文件
    train_file = f'k-fold/r{fold}.train'
    test_file = f'k-fold/r{fold}.test'

    train_df = read_official_file(train_file)  # 训练集数据
    test_df = read_official_file(test_file)  # 测试集数据


    # 构建当前fold的训练稀疏矩阵，仅包含训练集的评分
    def build_train_matrix(df):
        rows = df['UserIdx'].values  # 训练集的用户
        cols = df['MovieIdx'].values  # 训练集的电影
        vals = df['Rating'].values  # 训练集的评分
        return csr_matrix(
            (vals, (rows, cols)),
            shape=(n_users, n_movies)
        )


    train_matrix = build_train_matrix(train_df)

    # 存当前fold的训练矩阵和测试集，供后续模型使用，.npz格式
    save_npz(f"train_matrix_fold{fold}.npz", train_matrix)
    test_df[['UserIdx', 'MovieIdx', 'Rating']].to_csv(
        f"test_set_fold{fold}.csv",
        index=False
    )

    print(f"训练集评分数：{len(train_df)} | 测试集评分数：{len(test_df)}")
    print(f"训练矩阵形状：{train_matrix.shape} | 训练矩阵非零元素数：{train_matrix.nnz}")


"""
保存全局ID映射，因为后续模型可能需要还原原始ID
"""
np.savez(
    "id_maps.npz",
    user_map=user_map,
    movie_map=movie_map,
    n_users=n_users,
    n_movies=n_movies
)