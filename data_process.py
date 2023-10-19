import numpy as np
import tarfile
import pickle
from PIL import Image
import json
import csv

# # 解压缩文件
# with tarfile.open('cifar-100-python.tar.gz', 'r:gz') as tar:
#     tar.extractall()

# 加载数据
def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

data_batch = unpickle('cifar-100-python/train')

# 获取图像数据和标签
images = data_batch[b'data']
labels = data_batch[b'fine_labels']

# 将图像数据转换为向量
vectors = np.array(images) / 255.0  # 将像素值缩放到 [0, 1] 范围内
print(len(vectors))
# 打印示例
print('图像向量示例:')
for i in range(5):
    print(vectors[i])
    print(len(vectors[i]),labels[i])

# 将向量保存到向量数据库中
# 这里可以使用适合你的向量数据库的方法将向量保存起来


# # 构建 JSON 数据
# json_data = {"rows": []}
# for vector in vectors[:8000]:
#     json_data["rows"].append({"vector": vector.tolist()})
#
# # 保存为 JSON 文件
# with open("vectors.json", "w") as json_file:
#     json.dump(json_data, json_file)

def display_vector_as_image(vector, width=32, height=32):
    # 将一维向量转换为三维图像形状
    reshaped_vector = np.reshape(vector, (3,height, width))
    reshaped_vector=np.transpose(reshaped_vector, (1, 2, 0))
    print(reshaped_vector)
    # 缩放像素值到 [0, 255] 范围内
    scaled_vector = (reshaped_vector * 255).astype(np.uint8)

    # 创建图像对象
    image = Image.fromarray(scaled_vector)

    # 显示图像
    image.show()

# display_vector_as_image(vectors[0])
class_names_dict = {}
with open('class_names.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        label = int(row['Label'])
        class_name = row['Class Name']
        class_names_dict[label] = class_name
