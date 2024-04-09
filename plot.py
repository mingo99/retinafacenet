import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# log_path = "./log/train.log"
# # plot_lr(log_path, save_path="lr.png")
# plot_loss(log_path, save_path="loss.png")


font_path = '/mnt/d/Acc/software/nets/facenet-retinaface-pytorch/model_data/simhei.ttf'  # 替换为你本地的SimHei字体路径
font_prop = FontProperties(fname=font_path, size=12)

plt.figure(figsize=(15, 5), dpi=200)
# 绘制数据集分布饼状图
labels = ['Train', 'Test']
sizes = [35322,3550]  # 每个部分的百分比
plt.subplot(1, 2, 1)
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, explode=(0,0.1))
plt.title('图片', fontproperties=font_prop)  

labels2 = ['Train', 'Test']
sizes2 = [222554,21782]  # 每个部分的百分比
plt.subplot(1, 2, 2)
plt.pie(sizes2, labels=labels2, autopct='%1.1f%%', startangle=90, explode=(0, 0.1))
plt.title('标注', fontproperties=font_prop)
plt.suptitle('训练集与测试集的比例', fontproperties=font_prop)

# # 创建第一个子图
# labels = ['Body', 'Face']
# sizes = [156065,88271]  # 每个部分的百分比
# plt.subplot(1, 3, 1)
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, explode=(0,0.1))
# plt.title('数据集', fontproperties=font_prop)  
#
# # 创建第二个子图
# labels2 = ['Body', 'Face']
# sizes2 = [141900,80564]  # 每个部分的百分比
# plt.subplot(1, 3, 2)
# plt.pie(sizes2, labels=labels2, autopct='%1.1f%%', startangle=90, explode=(0,0.1))
# plt.title('训练集', fontproperties=font_prop)  
#
# # 创建第三个子图
# labels3 = ['Body', 'Face']
# sizes3 = [14075,7707]  # 每个部分的百分比
# plt.subplot(1, 3, 3)
# plt.pie(sizes3, labels=labels3, autopct='%1.1f%%', startangle=90, explode=(0, 0.1))
# plt.title('验证集', fontproperties=font_prop)
# plt.suptitle('人体与人脸标注的比例', fontproperties=font_prop)
# 调整布局
plt.tight_layout()
# 显示图
plt.show()
