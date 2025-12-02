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
