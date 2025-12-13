import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob


class HistoryTimeAnalyzer:
    def __init__(self, history_dir="history"):
        self.history_dir = history_dir
        self._check_history_dir()
        self.history_files = glob(os.path.join(history_dir, "*_fold*_history.npz"))
        if not self.history_files:
            raise FileNotFoundError(f"在{history_dir}目录下未找到历史文件")

    def _check_history_dir(self):
        """检查历史文件目录是否存在"""
        if not os.path.exists(self.history_dir):
            raise NotADirectoryError(f"历史文件目录{self.history_dir}不存在")

    def _parse_model_info(self, filename):
        """解析文件名，提取模型名称和折数（与主训练文件中_save_history的命名对应）"""
        # 文件名格式{model_name}_fold{fold}_history.npz
        base_name = os.path.basename(filename)
        model_part, fold_part = base_name.split("_fold")
        model_name = model_part.replace("_", " ")  # 恢复保存时替换的空格
        fold = int(fold_part.split("_")[0])
        return model_name, fold

    def _get_time_to_target(self, iter_history, target_mse):
        """
        计算达到目标精度的时间
        :param iter_history: 从历史文件加载的迭代记录，shape=(n_iter, 2)，列=[时间, MSE]
        :param target_mse: 目标精度（MSE）
        :return: 达到目标的时间（秒），未达到则返回np.nan
        """
        if len(iter_history) == 0:
            return np.nan

        # 遍历迭代历史，找到首次MSE<=目标值的时间
        for time_elapsed, mse in iter_history:
            if mse <= target_mse:
                return time_elapsed
        return np.nan  # 未达到目标精度

    def _extract_rank(self, model_name):
        """
        精准提取模型名中的rank数值（适配ALS含rank=、SoftImpute含r=）
        示例匹配：
        - ALS rank=5 → 5
        - ALS_rank=10 → 10
        - SoftImpute r=8 → 8
        - SoftImpute_r=20 → 20
        :param model_name: 模型名称
        :return: rank数值（int），未找到则返回0
        """
        # 匹配ALS的rank=xxx（忽略大小写/下划线/空格）
        als_rank_match = re.search(r'rank\s*=\s*(\d+)', model_name, re.IGNORECASE)
        if als_rank_match:
            return int(als_rank_match.group(1))

        # 匹配SoftImpute的r=xxx（忽略大小写/下划线/空格）
        soft_impute_rank_match = re.search(r'r\s*=\s*(\d+)', model_name, re.IGNORECASE)
        if soft_impute_rank_match:
            return int(soft_impute_rank_match.group(1))

        return 0  # 无rank时默认0

    def analyze(self, target_mse, output_csv="time_analysis.csv"):
        """
        分析所有模型达到目标精度的时间
        :param target_mse: 统一的目标精度（MSE）
        :param output_csv: 分析结果保存路径
        :return: 分析结果DataFrame
        """
        results = []
        for file_path in self.history_files:
            # 解析模型信息
            model_name, fold = self._parse_model_info(file_path)

            # 加载历史数据（与test1中_save_history的保存格式对应）
            data = np.load(file_path)
            iter_history = data["iter_history"]  # 迭代历史：(时间, MSE)
            final_metric = data["final_metric"]  # 最终MSE
            total_time = data["total_fit_time"]  # 总训练时间

            # 计算达到目标精度的时间
            time_to_target = self._get_time_to_target(iter_history, target_mse)

            results.append({
                "model_name": model_name,
                "fold": fold,
                "time_to_target": time_to_target,
                "final_mse": final_metric,
                "total_training_time": total_time,
                "reached_target": not np.isnan(time_to_target)
            })

        # 整理结果
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_csv, index=False)
        print(f"分析结果已保存至 {output_csv}")
        return result_df

    def visualize(self, result_df, target_mse, fig_path="time_comparison.png", rank_sort_asc=True):
        """
        可视化不同模型达到目标精度的时间对比
        :param result_df: 分析结果DataFrame
        :param target_mse: 目标MSE（用于图表标注）
        :param fig_path: 图表保存路径
        :param rank_sort_asc: rank是否升序排列（True=升序，False=降序）
        """
        plt.figure(figsize=(10, 6))

        # ========== 核心逻辑：按模型类型+rank排序 ==========
        all_models = result_df["model_name"].unique()

        # 1. 分离ALS和SoftImpute模型（精准匹配关键词）
        als_models = [m for m in all_models if "ALS" in m]
        soft_impute_models = [m for m in all_models if "SoftImpute" in m]

        # 2. 按rank数值排序（升序/降序可自定义）
        # ALS模型按rank排序
        als_models_sorted = sorted(
            als_models,
            key=lambda x: self._extract_rank(x),
            reverse=not rank_sort_asc
        )
        # SoftImpute模型按rank排序
        soft_impute_models_sorted = sorted(
            soft_impute_models,
            key=lambda x: self._extract_rank(x),
            reverse=not rank_sort_asc
        )

        # 3. 合并顺序：ALS（按rank排）在前，SoftImpute（按rank排）在后
        sorted_models = als_models_sorted + soft_impute_models_sorted

        # 4. 为不同模型分配颜色：ALS用skyblue，SoftImpute用orange
        colors = []
        avg_times = []
        for model in sorted_models:
            model_data = result_df[result_df["model_name"] == model]
            avg_time = model_data["time_to_target"].mean()
            avg_times.append(avg_time)
            # 分配颜色
            if "ALS" in model:
                colors.append("skyblue")
            elif "SoftImpute" in model:
                colors.append("orange")

        # 绘制条形图（指定颜色）
        plt.bar(sorted_models, avg_times, color=colors)

        # ========== 图表标注优化 ==========
        plt.ylabel(f"Average Time to Reach Target Accuracy (s)")
        plt.title("Comparison of Time Taken by Different Models to Reach the Same Accuracy")
        plt.xticks(rotation=45, ha="right")  # ha=right 让标签更贴合坐标轴
        plt.tight_layout()  # 自动调整布局，防止标签被截断
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")  # 高清保存，防止标签截断
        print(f"可视化结果已保存至 {fig_path}")
        plt.show()


if __name__ == "__main__":
    # 目标精度（根据实际需求调整，例如取所有模型最终MSE的中位数）
    target_mse = 0.85  # 示例值，需根据实际历史数据调整

    # 初始化分析器
    analyzer = HistoryTimeAnalyzer()

    # 执行分析
    analysis_result = analyzer.analyze(target_mse)

    # 打印关键统计结果
    print("\n各模型平均达到目标精度的时间：")
    print(analysis_result.groupby("model_name")["time_to_target"].agg(["mean", "std", "count"]))

    # 可视化：ALS和SoftImpute内部按rank升序排列（如需降序，设置rank_sort_asc=False）
    analyzer.visualize(analysis_result, target_mse, rank_sort_asc=True)