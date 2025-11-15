# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
#
#
#
# # 设置中文字体支持（仅用于处理中文数据，图表中将显示英文）
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 使用 SimHei
# # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac 使用 Arial Unicode MS
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
#
# def plot_sample_distribution(csv_path="D:/data/new_birdsData_id.csv", output_image="sample_distribution.png"):
#     # 1. 读取 CSV 文件
#     try:
#         df = pd.read_csv(csv_path)
#     except FileNotFoundError:
#         print(f"错误: 文件 {csv_path} 不存在！")
#         return
#     except Exception as e:
#         print(f"读取 CSV 失败: {e}")
#         return
#
#     # 2. 定义中英文映射
#     name_mapping = {
#         '灰雁': 'Greylag Goose',
#         '大天鹅': 'Mute Swan',
#         '绿头鸭': 'Mallard',
#         '绿翅鸭': 'Common Teal',
#         '灰山鹑': 'Grey Partridge',
#         '西鹌鹑': 'Common Quail',
#         '雉鸡': 'Common Pheasant',
#         '红喉潜鸟': 'Red-throated Loon',
#         '苍鹭': 'Grey Heron',
#         '普通鸬鹚': 'Great Cormorant',
#         '苍鹰': 'Northern Goshawk',
#         '欧亚鵟': 'Common Buzzard',
#         '西方秧鸡': 'Water Rail',
#         '骨顶鸡': 'Common Coot',
#         '黑翅长脚鹬': 'Black-winged Stilt',
#         '凤头麦鸡': 'Northern Lapwing',
#         '白腰草鹬': 'Common Greenshank',
#         '红脚鹬': 'Common Redshank',
#         '林鹬': 'Wood Sandpiper',
#         '麻雀': 'House Sparrow'
#     }
#
#     # 将中文鸟类名称转换为英文
#     df['class_en'] = df['class'].map(name_mapping)
#
#     # 3. 统计每个类别的样本数量，保持原始顺序
#     class_order = df['class_en'].unique()  # 获取英文类别首次出现的顺序
#     class_counts = df['class_en'].value_counts().reindex(class_order).reset_index()
#     class_counts.columns = ['class_en', 'count']
#     total_samples = class_counts['count'].sum()
#
#     # 创建 3D 柱状图
#     fig = plt.figure(figsize=(14, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 5. 设置图表样式
#     plt.title(f'Sample Distribution (Total Samples: {total_samples})', fontsize=16, pad=15)
#     plt.xlabel('Bird Species', fontsize=12)
#     plt.ylabel('Sample Count', fontsize=12)
#     plt.xticks(rotation=45, ha='right', fontsize=10)  # 旋转X轴标签
#     plt.yticks(fontsize=10)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加Y轴网格线
#
#     # 在柱子上显示具体数量
#     for i, count in enumerate(class_counts['count']):
#         plt.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10)
#
#     # 6. 调整布局，避免标签被裁剪
#     plt.tight_layout()
#
#     # 7. 保存和显示
#     try:
#         plt.savefig(output_image, dpi=300, bbox_inches='tight')
#         print(f"柱状图已保存至: {output_image}")
#     except Exception as e:
#         print(f"保存图片失败: {e}")
#     plt.show()
#
#
# if __name__ == "__main__":
#     plot_sample_distribution()
#


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 数据
data = {
    'class': ['灰雁', '大天鹅', '绿头鸭', '绿翅鸭', '灰山鹑', '西鹌鹑', '雉鸡', '红喉潜鸟', '苍鹭', '普通鸬鹚',
              '苍鹰', '欧亚鵟', '西方秧鸡', '骨顶鸡', '黑翅长脚鹬', '凤头麦鸡', '白腰草鹬', '红脚鹬', '林鹬', '麻雀'],
    'count': [759, 800, 766, 602, 29, 738, 797, 835, 850, 852, 733, 290, 680, 460, 786, 814, 710, 790, 825, 1195]
}
df = pd.DataFrame(data)

# 中英文映射
name_mapping = {
    '灰雁': 'Greylag Goose', '大天鹅': 'Mute Swan', '绿头鸭': 'Mallard', '绿翅鸭': 'Common Teal',
    '灰山鹑': 'Grey Partridge', '西鹌鹑': 'Common Quail', '雉鸡': 'Common Pheasant',
    '红喉潜鸟': 'Red-throated Loon', '苍鹭': 'Grey Heron', '普通鸬鹚': 'Great Cormorant',
    '苍鹰': 'Northern Goshawk', '欧亚鵟': 'Common Buzzard', '西方秧鸡': 'Water Rail',
    '骨顶鸡': 'Common Coot', '黑翅长脚鹬': 'Black-winged Stilt', '凤头麦鸡': 'Northern Lapwing',
    '白腰草鹬': 'Common Greenshank', '红脚鹬': 'Common Redshank', '林鹬': 'Wood Sandpiper',
    '麻雀': 'House Sparrow'
}

# 将中文转换为英文
df['class_en'] = df['class'].map(name_mapping)

# 创建 3D 柱状图
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# 数据准备
xpos = np.arange(len(df))  # X轴：类别索引
ypos = np.zeros(len(df))   # Y轴：固定为0（单组柱状图）
zpos = np.zeros(len(df))   # Z轴底部：0
dx = 0.5  # 柱子宽度
dy = 0.5 # 柱子深度
dz = df['count'].values      # 柱子高度（样本数量）

# 绘制 3D 柱状图
colors = ['#2E8B57' for _ in range(len(df))]  # 统一使用图片中的绿色 (#2E8B57 - SeaGreen)
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.8)

# 生成地面纹理（使用 2D 网格）
x_ground = np.linspace(min(xpos) - 1, max(xpos) + 1, 100)
y_ground = np.linspace(-1, 1, 100)
x_ground, y_ground = np.meshgrid(x_ground, y_ground)
z_ground = np.sin(x_ground * 0.5) * 10 - 50  # 波浪状纹理，位于 Z 轴下方

# 绘制地面
ax.plot_surface(x_ground, y_ground, z_ground, cmap='Greens', alpha=0.3)

# 设置轴标签
ax.set_xlabel('Bird Species', fontsize=12, labelpad=20)
ax.set_ylabel('', fontsize=12)  # Y轴无意义，留空
ax.set_zlabel('Sample Count', fontsize=12, labelpad=10)

# 设置 X 轴刻度和标签
ax.set_xticks(xpos)
ax.set_xticklabels(df['class_en'], rotation=45, ha='right', fontsize=10)

# 设置标题
total_samples = df['count'].sum()
ax.set_title(f'Sample Distribution (Total Samples: {total_samples})', fontsize=16, pad=20)

# 调整视角
ax.view_init(elev=20, azim=30)  # 模仿图片中的视角

# 在柱子上方显示数量
for i, count in enumerate(df['count']):
    ax.text(xpos[i], ypos[i], count + 20, str(count), ha='center', va='bottom', fontsize=8)

# 调整布局
plt.tight_layout()

# 保存和显示
output_image = "3d_sample_distribution_seagreen.png"
try:
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"3D 柱状图已保存至: {output_image}")
except Exception as e:
    print(f"保存图片失败: {e}")
plt.show()