import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# 读取日志数据
with open("output.log") as f:
    log_data = f.read()

# 正则表达式用于提取矩阵大小和完成时间
regex = r"matrix of size (\d+)x(\d+) completed in: ([\d.]+)s"

# 使用字典存储每个矩阵大小的总时间和计数
time_data = defaultdict(lambda: {"total_time": 0, "count": 0})

# 遍历日志数据，提取和累加时间
for match in re.finditer(regex, log_data):
    rows, cols, time = match.groups()
    mn_value = int(rows) * int(cols)  # 计算 m*n
    time_data[mn_value]["total_time"] += float(time)
    time_data[mn_value]["count"] += 1

# 计算每个 m*n 值的平均时间
average_times = {mn: data["total_time"] / data["count"] for mn, data in time_data.items()}

# 将 m*n 值进行排序
sorted_mn_values = sorted(average_times.keys())

# 计算排序后的平均时间
sorted_avg_times = [average_times[mn] for mn in sorted_mn_values]

# 转换 m*n 为实际的 m 和 n 比例
ratios = [np.sqrt(mn) for mn in sorted_mn_values]  # Assuming square matrices

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(ratios, sorted_avg_times, marker='o', color='darkblue')
plt.xlabel('Square Root of Matrix Size Product (sqrt(m*n))')
plt.ylabel('Average Time (s)')
plt.title('Average Time for SVD Computation by Matrix Size Ratio')
plt.grid(True)
plt.tight_layout()

# 保存图片 
plt.savefig("plot.png")
