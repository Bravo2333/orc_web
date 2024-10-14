import re

import cv2
import requests
from flask import Flask, request, jsonify, send_from_directory
import os
import base64
import numpy as np
from dec_rec import getrec_result
from flask_cors import CORS
import uuid
app = Flask(__name__)
CORS(app)
# 设置上传文件的静态目录
UPLOAD_FOLDER = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# 配置数据集存储路径
DATASET_FOLDER = 'datasets'

# 确保 datasets 文件夹存在
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
# 模拟表格识别处理函数

def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


def extract_base64_prefix(base64_string):
    """
    提取并移除 Base64 编码中的前缀，返回前缀和去除前缀后的Base64数据
    """
    # 使用正则表达式提取前缀
    match = re.match(r'^(data:.*?;base64,)', base64_string)

    if match:
        # 如果有前缀，提取前缀
        prefix = match.group(1)
        # 移除前缀
        base64_data = re.sub(r'^data:.*?;base64,', '', base64_string)
    else:
        # 如果没有前缀
        prefix = None
        base64_data = base64_string

    return prefix, base64_data




# 创建数据集
@app.route('/api/datasets/create', methods=['POST'])
def create_dataset():
    data = request.get_json()
    dataset_name = data.get('name')

    if not dataset_name:
        return jsonify({"error": "数据集名称不能为空"}), 400

    dataset_path = os.path.join(DATASET_FOLDER, dataset_name)
    images_folder = os.path.join(dataset_path, 'images')

    if os.path.exists(dataset_path):
        return jsonify({"error": "数据集已存在"}), 400

    # 创建数据集文件夹和 images 文件夹
    os.makedirs(images_folder)

    # 创建空的 annotations.txt 文件
    annotation_file = os.path.join(dataset_path, 'annotations.txt')
    with open(annotation_file, 'w') as f:
        pass  # 创建空文件

    return jsonify({"message": f"数据集 '{dataset_name}' 创建成功"}), 201


# 获取数据集列表
@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    datasets = [d for d in os.listdir(DATASET_FOLDER) if os.path.isdir(os.path.join(DATASET_FOLDER, d))]
    return jsonify(datasets), 200


# 删除数据集
@app.route('/api/datasets/delete', methods=['POST'])
def delete_dataset():
    data = request.get_json()
    dataset_name = data.get('name')

    dataset_path = os.path.join(DATASET_FOLDER, dataset_name)

    if not os.path.exists(dataset_path):
        return jsonify({"error": "数据集不存在"}), 400

    # 删除整个数据集文件夹
    os.system(f'rm -rf {dataset_path}')

    return jsonify({"message": f"数据集 '{dataset_name}' 已删除"}), 200


# 上传图片并自动化标注（模拟）
@app.route('/api/annotate', methods=['POST'])
def annotate_image():
    dataset_name = request.json.get('dataset')
    image_data = request.json.get('image')

    if not dataset_name or not image_data:
        return jsonify({"error": "数据集和图片数据不能为空"}), 400

    dataset_path = os.path.join(DATASET_FOLDER, dataset_name)
    images_folder = os.path.join(dataset_path, 'images')
    annotation_file = os.path.join(dataset_path, 'annotations.txt')

    if not os.path.exists(dataset_path):
        return jsonify({"error": "数据集不存在"}), 400

    # 模拟图片保存
    image_id = len(os.listdir(images_folder)) + 1
    image_path = os.path.join(images_folder, f'image{image_id}.png')

    with open(image_path, 'wb') as f:
        f.write(image_data.encode())  # 保存图片 (Base64解码可以使用base64模块)

    # 模拟标注信息 (这是一个简单的例子，实际可以根据需求生成标注数据)
    annotation_data = f"image{image_id}.png: 标注信息\n"

    # 将标注信息追加到 annotations.txt 文件中
    with open(annotation_file, 'a') as f:
        f.write(annotation_data)

    # 返回标注后的数据 (模拟)
    annotated_image_data = image_data  # 实际上应该是带有标注的图片

    return jsonify({
        "originalImage": image_data,
        "annotatedImage": annotated_image_data
    }), 200


# 获取数据集的标注信息
@app.route('/api/datasets/<dataset_name>/annotations', methods=['GET'])
def get_annotations(dataset_name):
    dataset_path = os.path.join(DATASET_FOLDER, dataset_name)
    annotation_file = os.path.join(dataset_path, 'annotations.txt')

    if not os.path.exists(annotation_file):
        return jsonify({"error": "数据集或标注信息不存在"}), 400

    annotations = []
    with open(annotation_file, 'r') as f:
        for line in f.readlines():
            annotations.append(line.strip())

    return jsonify(annotations), 200


# 提供图片访问服务
@app.route('/datasets/<dataset_name>/images/<filename>', methods=['GET'])
def get_image(dataset_name, filename):
    images_folder = os.path.join(DATASET_FOLDER, dataset_name, 'images')
    return send_from_directory(images_folder, filename)

# API 接口: 处理图像识别请求
@app.route('/api/recognize', methods=['POST'])
def recognize_table():
    data = request.json
    base64_image = data.get('image')
    random_filename = f"{uuid.uuid4()}.png"

    prefix,base64_image = extract_base64_prefix(base64_image)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], random_filename)
    image_data = base64.b64decode(base64_image)
    print(image_path)

    # 将解码后的二进制数据写入文件
    with open(image_path, "wb") as image_file:
        image_file.write(image_data)
    # with open('base64.txt', "wb") as image_file:
    #     image_file.write(base64_image)
    if base64_image:
        # 保存上传的图片
        output_text_path = 'recognized_texts.txt'
        base64_image = image_to_base64(image_path)

        # 准备POST请求的Payload
        payload = {
            "image": base64_image
        }
        # 发送POST请求到Flask服务
        response = requests.post('http://127.0.0.1:5000/api/recognize', json=payload)
        data = response.json()
        base64_result_image = prefix+data['image_base64']
        # 假设有两个多边形，4个点定义一个多边形
        polygons_list = data['polygons']
        print(polygons_list)
        polygons = [[(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)] for polygon in polygons_list]
        polygons = [np.array(polygon) for polygon in polygons]  # 将其转换为numpy格式

        # 执行主流程
        recognition_results = getrec_result(image_path, polygons, output_text_path)
        # 处理图片，模拟表格识别并返回结果
        result ={}
        result['base64'] = base64_result_image
        result['recognition_results'] = recognition_results
        print(type(result))
        # 返回识别的图片和文本结果
        return jsonify(result)
    return jsonify({"error": "Something went wrong"}), 500


# 提供静态文件（图片）
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True,port=3000)
