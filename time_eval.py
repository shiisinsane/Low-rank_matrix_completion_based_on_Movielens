import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob


class HistoryTimeAnalyzer:
    def __init__(self, history_dir="history"):
        self.history_dir = history_dir
        self._check_history_dir()
        # 复用test1中定义的历史文件命名规则
        self.history_files = glob(os.path.join(history_dir, "*_fold*_history.npz"))
        if not self.history_files:
            raise FileNotFoundError(f"在{history_dir}目录下未找到历史文件")

    def _check_history_dir(self):
        """检查历史文件目录是否存在（复用test1中的目录创建逻辑）"""
        if not os.path.exists(self.history_dir):
            raise NotADirectoryError(f"历史文件目录{self.history_dir}不存在")

    def _parse_model_info(self, filename):
        """解析文件名，提取模型名称和折数（与test1中_save_history的命名对应）"""
        # 文件名格式：{model_name}_fold{fold}_history.npz
        base_name = os.path.basename(filename)
        model_part, fold_part = base_name.split("_fold")
        model_name = model_part.replace("_", " ")  # 恢复保存时替换的空格
        fold = int(fold_part.split("_")[0])
        return model_name, fold

    def _get_time_to_target(self, iter_history, target_mse):
        """
        计算达到目标精度的时间（复用test1中潜在的时间计算逻辑）
        :param iter_history: 从历史文件加载的迭代记录，shape=(n_iter, 2)，列=[时间, MSE]
        :param target_mse: 目标精度（MSE）
        :return: 达到目标的时间（秒），未达到则返回np.nan
        """
        if len(iter_history) == 0:
            return np.nan

        # 遍历迭代历史，找到首次MSE <= 目标值的时间
        for time_elapsed, mse in iter_history:
            if mse <= target_mse:
                return time_elapsed
        return np.nan  # 未达到目标精度

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

    def visualize(self, result_df, fig_path="time_comparison.png"):
        """可视化不同模型达到目标精度的时间对比"""
        plt.figure(figsize=(10, 6))
        model_names = result_df["model_name"].unique()

        # 按模型分组计算平均时间
        avg_times = []
        for model in model_names:
            model_data = result_df[result_df["model_name"] == model]
            avg_time = model_data["time_to_target"].mean()
            avg_times.append(avg_time)

        # 绘制条形图
        plt.bar(model_names, avg_times, color=['skyblue', 'orange'])
        plt.ylabel(f"Average time to reach target accuracy (MSE={target_mse}s)")
        plt.title("Comparison of the time taken by different models to reach the same accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fig_path)
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

    # 可视化
    analyzer.visualize(analysis_result)