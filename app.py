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
