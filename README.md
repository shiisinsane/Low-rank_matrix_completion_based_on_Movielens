# 低秩矩阵填充：凸方法与非凸方法在MovieLens数据集上的实现与对比

## 一、项目概述

本项目实现了两种经典的低秩矩阵填充算法, **Soft-Impute（凸方法）** 和**交替最小二乘（ALS，非凸方法）**，用于处理MovieLens 10M/20M数据集中的用户-物品评分预测问题。通过5倍交叉验证的均方根误差（RMSE）评估模型性能，对比两种方法在稀疏评分矩阵补全任务中的效果。

## 二、数据集说明

数据集来源: https://grouplens.org/datasets/movielens/

使用**MovieLens 10M**数据集，包含用户对电影的评分记录，格式为稀疏矩阵：

- 行：用户（`UserIdx`）
- 列：电影（`MovieIdx`）
- 非零元素：用户对电影的评分
- 矩阵高度稀疏，因为大部分用户未评分大部分电影，需通过低秩假设补全缺失值

## 三、方法原理与公式推导

### 1. 数据预处理：评分中心化

为消除用户和物品的固有偏差，比如某些用户天生喜欢打高分，先对评分进行中心化处理：

- 全局均值： $\mu = \text{mean}(r_{ui})$（所有观察到的评分均值）
  
- 用户偏差： $b_u = \text{mean}(r_{ui} \mid i \in I(u)) - \mu$（用户u的平均评分减去全局均值）
  
- 物品偏差： $b_i = \text{mean}(r_{ui} \mid u \in U(i)) - \mu$（物品i的平均评分减去全局均值）
  中心化评分定义为： 
  $\hat{r}_ {ui}  = r_{ui} - \mu - b_u - b_i$
  
  代码中用`compute_global_user_item_means`函数计算 $\mu, b_u, b_i$，并在模型拟合前转换原始评分为 $\hat{r}_{ui}$
  

### 2. 凸方法：Soft-Impute

#### 原理

Soft-Impute基于**凸优化**，假设中心化后的评分矩阵 $\hat{R}$可由低秩矩阵 $X$近似，通过奇异值阈值化实现低秩约束，目标函数为：

$$
\min_{X} \frac{1}{2} | P_\Omega(\hat{R} - X) |_F^2 + \lambda | X |_*
$$

其中：

- $P_\Omega$为投影算子，仅保留观测到的评分位置
  
- $\lambda$为正则化参数，控制低秩程度
  

#### 求解步骤`PyTorchSoftImpute`

1. **截断奇异值分解（SVD）**：对中心化稀疏矩阵 $\hat{R}$进行截断SVD，保留前k个奇异值： 
  $\hat{R} \approx U S V^T$ 其中 $U \in \mathbb{R}^{m \times k}$， $S \in \mathbb{R}^{k \times k}$， $V^T \in \mathbb{R}^{k \times n}$
  
2. **软阈值处理**：对奇异值施加阈值化，实现核范数正则化: $S'_i = \max(S_i - \lambda, 0)$
  
3. **低秩近似矩阵**：保留非零奇异值对应的子矩阵，得到低秩近似： $X = U_k S'_k V_k^T $
  
  其中 $U_k, S'_k, V_k^T$为阈值化后保留的奇异向量和奇异值
  
4. **预测公式**：
  
  对用户 i 和物品 j 的评分预测为： $\hat{r}_{ij} = (U_k[i,:] \cdot S'_k) \cdot V_k^T[:,j] $
  
  最终评分需加回中心化偏差： $pred_{ij} = \hat{r}_{ij} + \mu + b_u[i] + b_i[j]$ 
  
  代码中的`predict`方法通过分块点积实现，`U_sub * V_sub`对应上述内积
  

### 3. 非凸方法：交替最小二乘ALS

#### 原理

ALS通过将低秩矩阵X分解为用户因子 $U \in \mathbb{R}^{m \times r}$和物品因子 $V \in \mathbb{R}^{n \times r}（X \approx UV^T）$，目标函数为非凸优化问题：

$$
\min_{U, V} \frac{1}{2} \sum_{(u,i) \in \Omega} (\hat{r}_{ui} - U_u V_i^T)^2 + \frac{\lambda}{2} \left( \sum_u c_u | U_u |^2 + \sum_i c_i | V_i |^2 \right)
$$

其中：

- $c_u, c_i$为正则化权重，与用户/物品的评分数量正相关，这里用到了WR（带权重的正则化）思想，算法实质上是ALS-WR
  
- $\lambda$为正则化参数
  

#### 求解步骤`PyTorchALS`

1. **迭代优化**：固定一个因子矩阵，求解另一个因子矩阵，交替进行：
  
  - **固定V，更新U**：对每个用户 u，求解正规方程: $(V_{\Omega(u)}^T V_{\Omega(u)} + \lambda c_u I) U_u = V_{\Omega(u)}^T \hat{r}_u $
    
    其中： $\Omega(u)$为用户 u 评分过的物品集合， $c_u = \max(1, |\Omega(u)| )$
    
  - **固定U，更新V**：对每个物品 i，求解对称的正规方程： 
    $(U_{\Omega(i)}^T U_{\Omega(i)} + \lambda c_i I) V_i = U_{\Omega(i)}^T \hat{r}_i$
    
2. **预测公式**：
  
  用户i对物品j的中心化评分预测为： $\hat{r}_{ij} = U_i \cdot V_j$
  
  最终评分加回偏差： $pred_{ij} = \hat{r}_{ij} + \mu + b_u[i] + b_i[j]$
  

## 四、项目结构

```

├── /
├── get_data.py # 稀疏矩阵存储
├── test_cuda.py # 初版代码，验证之后发现精度不太行
└── ALS_softImpute.py # 分别实现凸/非凸方法的正式代码文件
```

### 1. 预处理数据：get_data.py

该脚本运行前需通过官方脚本`split_ratings.sh` 划分好了5折交叉验证文件，这些文件的格式此时仍是`.dat`

完成划分后，需确保以下文件存在于指定路径：

| 文件/目录 | 路径，与脚本同目录 | 格式说明 |
| --- | --- | --- |
| `ratings.dat` | 根目录 | ::分隔的4列数据，列名：UserID, MovieID, Rating, Timestamp |
| `k-fold/` | 根目录 | 存放5折交叉验证文件，包含：r1.train、r1.test、r2.train…r5.train、r5.test |
| `k-fold/r{fold}.train` | k-fold子目录 | 同`ratings.dat`格式，为第fold折的训练集 |
| `k-fold/r{fold}.test` | k-fold子目录 | 同`ratings.dat`格式，为第fold折的测试集 |

需安装以下Python库：

```bash
pip install pandas numpy scipy
```

确认输入文件目录结构符合上述要求，直接运行脚本：

```bash
python get_data.py 
```

脚本运行后会生成以下文件，均在脚本同级目录：

#### 1）训练稀疏矩阵

- 文件名：`train_matrix_fold{fold}.npz`（fold=1-5）
- 格式：以`Scipy CSR`**稀疏矩阵格式**进行存储，文件后缀为`.npz`
- 矩阵形状为`(n_users, n_movies)`，非零元素为对应用户-电影的评分，仅包含训练集数据。

#### 2）测试集文件

- 文件名：`test_set_fold{fold}.csv`（fold=1-5）
  
- 格式：CSV文件，无索引列，包含3列：
  
  | 列名  | 说明  |
  | --- | --- |
  | UserIdx | 连续化用户索引（从0开始） |
  | MovieIdx | 连续化电影索引（从0开始） |
  | Rating | 原始评分（1-5） |
  

#### 3）全局ID映射文件

- 文件名：`id_maps.npz`
  
- 包含4个变量：
  
  | 变量名 | 类型  | 说明  |
  | --- | --- | --- |
  | user_map | 字典  | 原始UserID -->连续UserIdx |
  | movie_map | 字典  | 原始MovieID -->连续MovieIdx |
  | n_users | 整数  | 全局唯一用户总数 |
  | n_movies | 整数  | 全局唯一电影总数 |
  
  仅作为备用，后续可通过该文件将模型输出的连续索引还原为原始ID。
  

### 2. 实验脚本：ALS_softImpute.py

#### 1）工具函数

`compute_global_user_item_means`：计算全局均值、用户偏差、物品偏差，用于评分中心化

#### 2）模型类

`PyTorchSoftImpute`类、`PyTorchALS`类

均有`fit`和`predict`类方法

#### 3）评估与实验

`MatrixCompletionEvaluator`：使用**5折交叉验证**进行评估，加载训练和测试数据，计算**RMSE**

`run_experiments`：对比不同参数的Soft-Impute和ALS模型，输出平均RMSE和运行时间

#### 4）使用方法

执行此脚本前需提前通过`get_data.py`将MovieLens数据集按5折划分并保存为`train_matrix_fold{1-5}.npz`（CSR格式）和`test_set_fold{1-5}.csv`

运行该脚本：

```bash
python ALS_softImpute.py
```

结果输出包含各模型RMSE和运行时间的CSV文件`matrix_completion_results_final_xxxx.csv`

## 五、实验设置

- 评价指标：RMSE（均方根误差）：  
  $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{(u,i) \in \text{test}} (r_{ui} - \text{pred}_{ui})^2}$
  
- 参数设置：
  
  Soft-Impute： $\lambda \in {1e-4, 5e-5}$， $r=250$
  
  ALS：秩 $r \in {200, 250}$， $\lambda=0.02$，最大迭代30次


## 附录1: 参考资料
在完成本实验的过程中, 除了参考最优化课程ppt以外, 还参考了以下资料:

**(1)** 实现soft-Impute算法的时候参考了以下网站: http://web.stanford.edu/~hastie/swData.htm

**(2)** 最优化课程推荐阅读文献:

[1] Spectral Regularization Algorithms for Learning Large Incomplete Matrices, 
2010, J Mach Learn Res. 

[2] Matrix Completion has No Spurious Local Minimum, NIPS, 2016. 

[3] No Spurious Local Minima in Nonconvex Low Rank Problems: A Unified 
Geometric Analysis, ICML, 2017.

[4] Low-rank Matrix Completion using Alternating Minimization, NIPS, 2013.

[5] Matrix Factorization Techniques for Recommender Systems, J of Computer, 
2009.

**(3)** 针对推荐系统常用算法的批判性综述文献:

Rendle, S., Zhang, L., & Koren, Y. (2019). On the Difficulty of Evaluating Baselines: A Study on Recommender Systems. ArXiv, abs/1905.01395.

**(4)** 将ALS改进为ALS-WR以降低RMSE到可容忍标准, 参考了以下文献 (是文献(3)的connected paper): 

Florian Strub, Romaric Gaudel, and Jérémie Mary. 2016. Hybrid Recommender System based on Autoencoders. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems (DLRS 2016). Association for Computing Machinery, New York, NY, USA, 11–16. https://doi.org/10.1145/2988450.2988456

**（5）** 历史baseline参考及其参数信息来源：

维基百科：https://en.wikipedia.org/wiki/Netflix_Prize

文献：Koren, Y. (2009). The BellKor Solution to the Netflix Grand Prize.

## 附录2: 实验日志

#### 12.2--->利用movielens 10M官方数据集的split_ratings.sh脚本划分5折交叉验证集

官方明确说明：

> A Unix shell script, ,is provided that, if desired, can be used to split the ratings data for five-fold cross-validation of rating predictions. It depends on a second script, allbut.pl, which is also included and is written in Perl. They should run without modification under Linux, Mac OS X, Cygwin or other Unix like systems. `split_ratings.sh`

> Running will use as input, and produce the fourteen output files described below. Multiple runs of the script will produce identical results. `split_ratings.sh`, `ratings.dat`

注意Windows系统没有原生的perl环境。我们在Linux（Ubuntu）系统上执行脚本。

把`split_ratings.sh`, `allbut.pl`, `ratings.dat`文件放到同一目录下，切换到当前目录，然后依次执行：

```bash
chmod +x split_ratings.sh
chmod +x allbut.pl
./split_ratings.sh
```

在当前目录下得到划分好的数据集：r1.train, r2.train, r3.train, r4.train, r5.train , r1.test, r2.test, r3.test, r4.test, r5.test； ra.train, rb.train , ra.test, rb.test

将这些文件放到`k-fold`目录下备用。


#### 12.3--->第一次尝试, RMSE均在1以上, 精度不能接受
| Model                     | Avg RMSE | Avg Time (s) | Fold 1 RMSE | Fold 2 RMSE | Fold 3 RMSE | Fold 4 RMSE | Fold 5 RMSE |
| ------------------------- | -------- | ------------ | ----------- | ----------- | ----------- | ----------- | ----------- |
| ALS (rank=80)             | 1.060435 | 862.8742     | 1.062899    | 1.056557    | 1.061779    | 1.058958    | 1.061982    |
| ALS (rank=100)            | 1.060469 | 908.2947     | 1.062928    | 1.056602    | 1.061807    | 1.059009    | 1.061998    |
| Soft-Impute (lambda=0.05) | 1.06043  | 2.563487     | 1.062888    | 1.056572    | 1.061762    | 1.058941    | 1.061987    |
| Soft-Impute (lambda=0.01) | 1.06043  | 2.54833      | 1.062888    | 1.056573    | 1.061759    | 1.058941    | 1.061987    |

#### 12.6 ---> 解决了一系列bug, 产生新问题

--->遇到CUDA OOM的问题, 原因可能是在ALS中同时把所有评分(U[user], V[movie])全部拉到GPU做大规模点积, 导致显存爆掉 

---> 解决方式: ALS中改用小批量方式(chunks/batches)计算residual

---> 结果: ALS可以正常运行, soft-Impute还是爆显存 ---> 并非chunk的问题, 可能是公式写错了

---> 进一步修改结果: 两种算法均可正常运行, ALS精度达到0.94的级别, 但soft-Impute精度失常

| Model                                | Avg RMSE | Avg Time (s) | Fold 1 RMSE | Fold 2 RMSE | Fold 3 RMSE | Fold 4 RMSE | Fold 5 RMSE |
| ------------------------------------ | -------- | ------------ | ----------- | ----------- | ----------- | ----------- | ----------- |
| ALS (rank=250)                       | 0.944064 | 1642.629     | 0.945687    | 0.944656    | 0.942666    | 0.944987    | 0.942322    |
| ALS (rank=200)                       | 0.944078 | 1498.227     | 0.945725    | 0.944677    | 0.942698    | 0.944979    | 0.942309    |
| SoftImpute (svds lambda=1e-4, r=250) | 2.727029 | 13.97935     | 2.723761    | 2.72417     | 2.71954     | 2.742078    | 2.725596    |
| SoftImpute (svds lambda=5e-5, r=250) | 2.727029 | 12.52122     | 2.723761    | 2.72417     | 2.71954     | 2.742078    | 2.725596    |

#### 12.7 ---> 解决soft-Impute精度失常的问题

--->解决方式: 对soft-Impute做中心化操作

--->结果: ALS和soft-Impute精度均到0.94的级别

| Model                                | Avg RMSE | Avg Time (s) | Fold 1 RMSE | Fold 2 RMSE | Fold 3 RMSE | Fold 4 RMSE | Fold 5 RMSE |
| ------------------------------------ | -------- | ------------ | ----------- | ----------- | ----------- | ----------- | ----------- |
| SoftImpute (svds lambda=1e-4, r=250) | 0.943758 | 42.71561     | 0.945405    | 0.944342    | 0.942377    | 0.944654    | 0.94201     |
| SoftImpute (svds lambda=5e-5, r=250) | 0.943758 | 40.64163     | 0.945405    | 0.944342    | 0.942377    | 0.944654    | 0.94201     |
| ALS (rank=250)                       | 0.944064 | 1635.687     | 0.945687    | 0.944656    | 0.942666    | 0.944987    | 0.942322    |
| ALS (rank=200)                       | 0.944078 | 1509.731     | 0.945725    | 0.944677    | 0.942698    | 0.944979    | 0.942309    |

#### 12.8 ---> 检查了一下soft-Impute的实现, 发现不太严谨的地方; ALS的运算时间好像不够快, 尝试改进

---> 解决方式: 对soft-Impute又参考了一下Spectral Regularization Algorithms for Learning Large Incomplete Matrices这篇文献, 进行了算法修改;
ALS在GPU上一次求解很多小型的线性问题, 有大量的GPU kernel调用, 但搬到CPU上求解可能反而更快, 使用GPU并没有实质上的提速, 改进了一下device的利用

---> 结果: 精度好像没什么变化；运行时间还是都很长

#### 12.10 ---> 继续解决了一系列bug：

#### （1）添加了更加公平的时间对比方式：两种算法达到同一精度所用时间；

#### （2）两种算法均使用“随机SVD”代替原始的SVD，以加快时间；

#### （3）调整softImpute正则化系数数量级，适配其奇异值的数量级以便做软阈值操作；

#### （4）修改ALS初始化方式：将小随机初始化改为SVD初始化以加速

#### （5）降低两种算法的rank参数

----> 结果：算法运行时间明显快了一些；精度倒是没有什么改变

| Model                             | Avg RMSE | Avg Time (s) | Fold 1 RMSE | Fold 2 RMSE | Fold 3 RMSE | Fold 4 RMSE | Fold 5 RMSE |
| --------------------------------- | -------- | ------------ | ----------- | ----------- | ----------- | ----------- | ----------- |
| SoftImpute(svds_lambda=80_r=150)  | 0.943757 | 364.1642     | 0.945404    | 0.944342    | 0.942373    | 0.944654    | 0.94201     |
| SoftImpute(svds_lambda=100_r=200) | 0.943757 | 339.0217     | 0.945404    | 0.944342    | 0.942373    | 0.944654    | 0.94201     |
| ALS(rank=150)                     | 0.943757 | 348.0491     | 0.945405    | 0.944343    | 0.942371    | 0.944655    | 0.942011    |
| ALS(rank=200)                     | 0.943757 | 595.8006     | 0.945404    | 0.944342    | 0.942372    | 0.944654    | 0.94201     |

各模型平均达到目标精度的时间（target MSE = 0.85）：

| model_name                        | avg  time（s）   | std count |
| --------------------------------- | -------- | --------- |
| ALS(rank=150)                     | 16.37189 | 3.146436  |
| ALS(rank=200)                     | 29.67463 | 3.142044  |
| SoftImpute(svds lambda=100 r=200) | 23.06343 | 3.74411   |
| SoftImpute(svds lambda=80 r=150)  | 23.42607 | 1.046526  |






