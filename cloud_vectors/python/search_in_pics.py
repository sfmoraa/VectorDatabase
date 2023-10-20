import configparser
import time
import random
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
import numpy as np
from PIL import Image
import pickle
import csv

def display_vector_as_image(vector, width=32, height=32):
    # 将一维向量转换为三维图像形状
    reshaped_vector = np.reshape(vector, (3,height, width))
    reshaped_vector=np.transpose(reshaped_vector, (1, 2, 0))
    # 缩放像素值到 [0, 255] 范围内
    scaled_vector = (reshaped_vector * 255).astype(np.uint8)

    # 创建图像对象
    image = Image.fromarray(scaled_vector)

    # 显示图像
    image.show()


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def generate_vector(num=1):
    class_names_dict = {}
    with open('../../class_names.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label = int(row['Label'])
            class_name = row['Class Name']
            class_names_dict[label] = class_name

    data_batch = unpickle('../../cifar-100-python/train')
    images = data_batch[b'data']
    labels = data_batch[b'fine_labels']
    vectors = np.array(images) / 255.0  # 将像素值缩放到 [0, 1] 范围内

    random_numbers = []
    for _ in range(num):
        random_index = random.randint(0, len(vectors) - 1)
        random_numbers.append(random_index)
        print("current label:",class_names_dict[labels[random_index]])

    display_vector_as_image(vectors[random_numbers])
    return vectors[random_numbers]


dim=3072
cfp = configparser.RawConfigParser()
cfp.read('config_serverless.ini')
milvus_uri = cfp.get('example', 'uri')
token = cfp.get('example', 'token')
connections.connect("default",
                        uri=milvus_uri,
                        token=token)

collection_name="pics"
id=FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
pic_vector=FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)

schema = CollectionSchema(fields=[id,pic_vector],
                          auto_id=True
                          )

collection = Collection(name=collection_name)#, schema=schema
t0 = time.time()
print("Loading collection...")
collection.load()
t1 = time.time()
print(f"Succeed in {round(t1-t0, 4)} seconds!")
nq = 1
search_params = {"metric_type": "L2",  "params": {"level": 2}}


topk = 5
for i in range(1):
    # search_vec = [[random.random() for _ in range(dim)] for _ in range(nq)]
    search_vec = generate_vector(1)
    print(f"Searching vector: {search_vec[0][0:5]}...")
    t0 = time.time()
    results = collection.search(search_vec,
                            anns_field=pic_vector.name,
                            param=search_params,
                            limit=topk,
                            output_fields=[pic_vector.name])
    t1 = time.time()
    print(f"Result:{type(results)}")
    print(f"search {i} latency: {round(t1-t0, 4)} seconds!")
    print(results[0][0].entity.to_dict()['entity']['vector'])
    display_vector_as_image(results[0][0].entity.to_dict()['entity']['vector'])


connections.disconnect("default")



