## 开发日志

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

---> 结果: ALS可以正常运行, soft-Impute还是爆显存 ---> 并非chunk的问题, 可能是fallback预测公式写错了

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
| ALS (rank=250)                       | 0.944064 | 1642.629     | 0.945687    | 0.944656    | 0.942666    | 0.944987    | 0.942322    |
| ALS (rank=200)                       | 0.944078 | 1498.227     | 0.945725    | 0.944677    | 0.942698    | 0.944979    | 0.942309    |
| SoftImpute (svds lambda=1e-4, r=250) | 0.943758 | 43.50782     | 0.945405    | 0.944342    | 0.942377    | 0.944654    | 0.94201     |
| SoftImpute (svds lambda=5e-5, r=250) | 0.943758 | 42.26582     | 0.945405    | 0.944342    | 0.942377    | 0.944654    | 0.94201     |




