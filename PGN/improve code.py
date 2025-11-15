import pandas as pd

# # 加载 CSV 文件
csv_path = "D:/data/ESC-50-master/audio/esc50.csv"  # 替换为你的 CSV 文件路径
df = pd.read_csv(csv_path)
# # 查看列名
# print(df.columns)
# #修改列名
# df.rename(columns={"target": "classID"}, inplace=True)
# df.rename(columns={"category": "class"}, inplace=True)
# # 删除不需要的列
# # df.drop(columns=["url"], inplace=True)
# # df.drop(columns=["author"], inplace=True)
# # df.drop(columns=["license"], inplace=True)
# # 保存为新的 CSV 文件
# output_path = "D:/data/ESC-50-master/ESC-50-master/audio/esc501.csv"
# df.to_csv(output_path, index=False)
#
# print(f"已保存修改后的文件到 {output_path}")
# 添加一个'id'列，值从0开始按顺序编号
df.insert(0, 'id', range(0, len(df)))

# 将修改后的数据保存到新的CSV文件
output_file = "esc50.csv"
df.to_csv(output_file, index=False)

print(f"已将添加了'id'列（从0开始编号）的CSV文件保存为 {output_file}")