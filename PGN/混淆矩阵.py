import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



# 真实标签（行顺序）
true_labels = [
    'Greylag Goose', 'Mute Swan', 'Mallard', 'Common Teal',
    'Grey Partridge', 'Common Quail', 'Common Pheasant',
    'Red-throated Loon', 'Grey Heron',  'Great Cormorant',
    'Northern Goshawk',  'Common Buzzard',  'Water Rail',
    'Common Coot', 'Black-winged Stilt',  'Northern Lapwing',
    'Common Greenshank', 'Common Redshank', 'Wood Sandpiper',
    'House Sparrow'
]
# 预测标签（列顺序，与上方真实标签列对应）
pred_labels = true_labels
# 混淆矩阵数值（二维列表，行对应 true_labels，列对应 pred_labels ）
confusion_matrix_data = [
    [139, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 158, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 147, 3, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 2, 116, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 133, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 154, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [2, 0, 2, 0, 0, 0, 0, 0, 166, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 1, 0, 167, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 3, 0, 0, 0, 0, 0, 1, 1, 51, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0, 2, 148, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 2, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0, 86, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 162, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 157, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2, 0, 132, 0, 3, 1],
    [3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 156, 4, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 0],
    [1, 0, 1, 0, 0, 2, 2, 0, 0, 0, 1, 0, 2, 0, 0, 2, 2, 0, 0, 211]
]

# 设置图形大小
plt.figure(figsize=(12, 10))

# 绘制热图
sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues',
            xticklabels=true_labels, yticklabels=true_labels)

# 设置标题和标签
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# 旋转标签以提高可读性
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# 调整布局以防止标签被截断
plt.tight_layout()

# 显示图形
plt.savefig('confusion_matrix.png')
plt.show()