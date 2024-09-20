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
def fix_base64_padding(base64_string):
    # 计算需要的填充数，使得字符串长度是4的倍数
    missing_padding = len(base64_string) % 4
    if missing_padding != 0:
        base64_string += '=' * (4 - missing_padding)
    return base64_string
# API 接口: 处理图像识别请求
@app.route('/api/recognize', methods=['POST'])
def recognize_table():
    data = request.json
    base64_image = data.get('image')
    random_filename = f"{uuid.uuid4()}.jpg"
    
    base64_image = fix_base64_padding(base64_image)
    print(base64_image)
    # 将Base64编码的图片转换为OpenCV图像
    image = base64_to_image(base64_image)

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    if base64_image:
        # 保存上传的图片
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], random_filename)
        cv2.imwrite(image_path, image)
        output_text_path = 'recognized_texts.txt'
        base64_image = image_to_base64(image_path)

        # 准备POST请求的Payload
        payload = {
            "image": base64_image
        }
        # 发送POST请求到Flask服务
        response = requests.post('http://127.0.0.1:5000/api/recognize', json=payload)
        data = response.json()
        base64_result_image = data['image_base64']
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
