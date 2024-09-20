import requests
from flask import Flask, request, jsonify, send_from_directory
import os
import base64
import numpy as np
from dec_rec import getrec_result
app = Flask(__name__)

# 设置上传文件的静态目录
UPLOAD_FOLDER = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
# 模拟表格识别处理函数


# API 接口: 处理图像识别请求
@app.route('/api/recognize', methods=['POST'])
def recognize_table():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # 保存上传的图片
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        image_path = file_path
        output_text_path = 'recognized_texts.txt'
        base64_image = image_to_base64(image_path)

        # 准备POST请求的Payload
        payload = {
            "image": base64_image
        }
        # 发送POST请求到Flask服务
        response = requests.post('http://127.0.0.1:3000/api/recognize', json=payload)
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

        # 返回识别的图片和文本结果
        return jsonify({
            "image": {
                "base64": base64_result_image
            },
            "result": recognition_results
        })

    return jsonify({"error": "Something went wrong"}), 500


# 提供静态文件（图片）
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
