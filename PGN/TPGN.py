import pandas as pd

# 读取 CSV 文件（修改为你的实际路径）
csv_path = "/dev/sda1/dataset/BirdsData_16000/birds_data.csv"
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"错误: 文件 {csv_path} 不存在！")
    exit()
except Exception as e:
    print(f"读取 CSV 失败: {e}")
    exit()

# 获取类别首次出现的顺序
class_order = df['class'].unique()

# 按原始顺序统计每个类别的样本数量
label_counts = df['class'].value_counts().reindex(class_order)

# 打印结果
print(label_counts)